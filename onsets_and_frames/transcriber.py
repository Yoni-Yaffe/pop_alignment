import torch
import torch.nn.functional as F
from torch import nn
from .mel import melspectrogram
from .lstm import BiLSTM
from onsets_and_frames.constants import *

class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x
    

class ModulatedOnsetStack(nn.Module):
    def __init__(self, input_features, output_features, model_size,
                 mlp_input_features):
        super().__init__()
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
        self.model_size = model_size
        self.modulation1 = LinearModulation(model_size, mlp_input_features)
        self.modulation2 = LinearModulation(model_size, mlp_input_features)
        self.conv_stack = ConvStack(input_features, model_size)
        self.seq_model = sequence_model(model_size, model_size)
        self.fc = nn.Sequential(
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )

    def forward(self, x, inst_id):
        x = self.conv_stack(x)
        x = self.modulation1(x, inst_id)
        x = self.seq_model(x)
        x = self.modulation2(x, inst_id)
        x = self.fc(x)
        return x


class LinearModulation(nn.Module):
    def __init__(self, input_features, mlp_input_features):
        nn.Module.__init__(self)
        self.mlp = torch.nn.Linear(mlp_input_features, 2 * input_features)

    def forward(self, x, inst_id):
        inst_id = self.mlp(inst_id)
        scale, shift = inst_id.chunk(chunks=2, dim=1)
        x = (1 + scale) * x + shift
        return x



class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48,
                 onset_complexity=1,
                 n_instruments=13):
        nn.Module.__init__(self)
        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        onset_model_size = int(onset_complexity * model_size)
        self.onset_stack = nn.Sequential(
            ConvStack(input_features, onset_model_size),
            
            sequence_model(onset_model_size, onset_model_size),
            
            nn.Linear(onset_model_size, output_features * n_instruments),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features * n_instruments)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)

        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch, parallel_model=None, multi=False, positive_weight=2., inv_positive_weight=2.):
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)

        if not parallel_model:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel)

        if multi:
            onset_pred = onset_pred[..., : N_KEYS]
            offset_pred = offset_pred[..., : N_KEYS]
            frame_pred = frame_pred[..., : N_KEYS]
            velocity_pred = velocity_pred[..., : N_KEYS]

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        if 'velocity' in batch:
            predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)

        losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label, reduction='none'),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label, reduction='none'),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label, reduction='none'),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
        if 'velocity' in batch:
            losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], velocity_label, onset_label)

        onset_mask = 1. * onset_label
        onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
        onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
        onset_mask += 1
        if 'onset_mask' in batch:
            onset_mask = onset_mask * batch['onset_mask']

        offset_mask = 1. * offset_label
        offset_positive_weight = 2.
        offset_mask *= (offset_positive_weight - 1)
        offset_mask += 1.

        frame_mask = 1. * frame_label
        frame_positive_weight = 2.
        frame_mask *= (frame_positive_weight - 1)
        frame_mask += 1.

        for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, offset_mask, frame_mask]):
            losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator


class ModulatedOnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48,
                 onset_complexity=1,
                 n_instruments=13):
        super().__init__(input_features, output_features, model_complexity,
                         onset_complexity, n_instruments)
        model_size = model_complexity * 16
        onset_model_size = int(onset_complexity * model_size)
        self.onset_stack = ModulatedOnsetStack(input_features, output_features, onset_model_size, n_instruments)
        self.mlp = nn.Sequential(
            nn.Linear(n_instruments, onset_model_size),
            nn.LeakyReLU(),
            nn.Linear(onset_model_size, onset_model_size),
            nn.LeakyReLU(),
        )
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features * n_instruments)
        )

    def forward(self, mel, inst_id):
        inst_id = self.mlp(inst_id)
        onset_pred = self.onset_stack(mel, inst_id)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)

        onset_detached = onset_pred.detach()
        shape = onset_detached.shape
        keys = MAX_MIDI - MIN_MIDI + 1
        new_shape = shape[: -1] + (shape[-1] // keys, keys)
        onset_detached = onset_detached.reshape(new_shape)
        onset_detached, _ = onset_detached.max(axis=-2)

        offset_detached = offset_pred.detach()

        combined_pred = torch.cat([onset_detached, offset_detached, activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch, instruments_list, parallel_model=None, multi=False, positive_weight=2., inv_positive_weight=2.):
        n_instruments = len(instruments_list)
        audio_label = batch['audio']

        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        if 'velocity' in batch:
            velocity_label = batch['velocity']
        total_losses = {
            'loss/onset': 0,
            'loss/offset': 0,
            'loss/frame': 0,
            'loss/velocity': 0
        }
        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        one_hot_list = F.one_hot(torch.arange(n_instruments + 1).roll(-1, 0))
        for i, inst in enumerate(instruments_list):
            inst_id = one_hot_list[i]
            curr_onset_label = onset_label[:, N_KEYS * i: N_KEYS * (i + 1)]
            curr_offset_label = offset_label[:, N_KEYS * i: N_KEYS * (i + 1)]
            curr_frame_label = frame_label[:, N_KEYS * i: N_KEYS * (i + 1)]
            if 'velocity' in batch:
                curr_velocity_label = velocity_label[:, N_KEYS * i: N_KEYS * (i + 1)]
            if not parallel_model:
                onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel, inst_id)
            else:
                onset_pred, offset_pred, _, frame_pred, velocity_pred = parallel_model(mel, inst_id)

            if multi:
                onset_pred = onset_pred[..., : N_KEYS]
                offset_pred = offset_pred[..., : N_KEYS]
                frame_pred = frame_pred[..., : N_KEYS]
                velocity_pred = velocity_pred[..., : N_KEYS]

            predictions = {
                'onset': onset_pred.reshape(*curr_onset_label.shape),
                'offset': offset_pred.reshape(*curr_offset_label.shape),
                'frame': frame_pred.reshape(*curr_frame_label.shape),
            }
            if 'velocity' in batch:
                predictions['velocity'] = velocity_pred.reshape(*velocity_label.shape)
            if i == 0:
                total_predictions = predictions
            else:
                for key in total_predictions.keys():
                    total_predictions[key] = torch.cat(total_predictions[key], predictions[key])
            losses = {
                'loss/onset': F.binary_cross_entropy(predictions['onset'], curr_onset_label, reduction='none'),
                'loss/offset': F.binary_cross_entropy(predictions['offset'], curr_offset_label, reduction='none'),
                'loss/frame': F.binary_cross_entropy(predictions['frame'], curr_frame_label, reduction='none'),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
            if 'velocity' in batch:
                losses['loss/velocity'] = self.velocity_loss(predictions['velocity'], curr_velocity_label, curr_onset_label)

            onset_mask = 1. * onset_label
            onset_mask[..., : -N_KEYS] *= (positive_weight - 1)
            onset_mask[..., -N_KEYS:] *= (inv_positive_weight - 1)
            onset_mask += 1
            if 'onset_mask' in batch:
                onset_mask = onset_mask * batch['onset_mask']

            offset_mask = 1. * curr_offset_label
            offset_positive_weight = 2.
            offset_mask *= (offset_positive_weight - 1)
            offset_mask += 1.

            frame_mask = 1. * curr_frame_label
            frame_positive_weight = 2.
            frame_mask *= (frame_positive_weight - 1)
            frame_mask += 1.

            for loss_key, mask in zip(['onset', 'offset', 'frame'], [onset_mask, offset_mask, frame_mask]):
                losses['loss/' + loss_key] = (mask * losses['loss/' + loss_key]).mean()

            for key in ['onset', 'offset', 'frame']:
                total_losses[f'loss/{key}'] += losses[f'loss/{key}']
            if 'velocity' in batch:
                total_losses['loss/velocity'] += losses['loss/velocity']
        return total_predictions, total_losses



def duplicate_linear(linear, n):
    A, b = linear.parameters()
    in_features, out_features = linear.in_features, linear.out_features
    layer_new = torch.nn.Linear(in_features, n * out_features)
    A_new, b_new = layer_new.parameters()
    A_new.requires_grad, b_new.requires_grad = False, False
    for j in range(n):
        A_new[j * out_features: (j + 1) * out_features, :] = A.detach().clone()
        b_new[j * out_features: (j + 1) * out_features] = b.detach().clone()
    A_new.requires_grad, b_new.requires_grad = True, True
    return layer_new

def duplicate_linear_pop(linear, n):
    A, b = linear.parameters()
    num_nots = MAX_MIDI - MIN_MIDI + 1
    assert A.shape[0] % (num_nots) == 0
    num_prev_inst = A.shape[0] // num_nots - 1
    A_pitch = A.detach()[-num_nots:, :]
    b_pitch = b.detach()[-num_nots:]
    in_features, out_features = linear.in_features, linear.out_features
    layer_new = torch.nn.Linear(in_features, n * num_nots)
    A_new, b_new = layer_new.parameters()
    A_new.requires_grad, b_new.requires_grad = False, False
    for j in range(n):
        if j < num_prev_inst:
            A_new[j * num_nots: (j + 1) * num_nots, :] = A.detach()[j * num_nots: (j + 1) * num_nots, :].clone()
            b_new[j * num_nots: (j + 1) * num_nots] = b.detach()[j * num_nots: (j + 1) * num_nots].clone()
        else:
            A_new[j * num_nots: (j + 1) * num_nots, :] = A_pitch.clone()
            b_new[j * num_nots: (j + 1) * num_nots] = b_pitch.detach().clone()
    A_new.requires_grad, b_new.requires_grad = True, True
    return layer_new



def load_weights(model, old_model, n_instruments):
    for i in range(len(model.onset_stack)):
        if i < len(model.onset_stack) - 2:
            model.onset_stack[i].load_state_dict(old_model.onset_stack[i].state_dict())
        elif i < len(model.onset_stack) - 1:
            linear = old_model.onset_stack[i]
            layer_new = duplicate_linear(linear, n_instruments)
            model.onset_stack[i].load_state_dict(layer_new.state_dict())

    for i in range(len(model.frame_stack)):
        if i < len(model.frame_stack) - 1:
            model.frame_stack[i].load_state_dict(old_model.frame_stack[i].state_dict())

    for i in range(len(model.combined_stack)):
        if i < len(model.combined_stack) - 1:
            model.combined_stack[i].load_state_dict(old_model.combined_stack[i].state_dict())

    for i in range(len(model.offset_stack)):
        if i < len(model.offset_stack) - 1:
            model.offset_stack[i].load_state_dict(old_model.offset_stack[i].state_dict())

    for i in range(len(model.velocity_stack)):
        if i < len(model.velocity_stack) - 1:
            model.velocity_stack[i].load_state_dict(old_model.velocity_stack[i].state_dict())
        elif i < len(model.velocity_stack):
            linear = old_model.velocity_stack[i]
            layer_new = duplicate_linear(linear, n_instruments)
            model.velocity_stack[i].load_state_dict(layer_new.state_dict())


def load_weights_pop(model, old_model, n_instruments):
    for i in range(len(model.onset_stack)):
        if i < len(model.onset_stack) - 2:
            model.onset_stack[i].load_state_dict(old_model.onset_stack[i].state_dict())
        elif i < len(model.onset_stack) - 1:
            linear = old_model.onset_stack[i]
            layer_new = duplicate_linear_pop(linear, n_instruments)
            model.onset_stack[i].load_state_dict(layer_new.state_dict())

    for i in range(len(model.frame_stack)):
        if i < len(model.frame_stack) - 1:
            model.frame_stack[i].load_state_dict(old_model.frame_stack[i].state_dict())

    for i in range(len(model.combined_stack)):
        if i < len(model.combined_stack) - 1:
            model.combined_stack[i].load_state_dict(old_model.combined_stack[i].state_dict())

    for i in range(len(model.offset_stack)):
        if i < len(model.offset_stack) - 1:
            model.offset_stack[i].load_state_dict(old_model.offset_stack[i].state_dict())

    for i in range(len(model.velocity_stack)):
        if i < len(model.velocity_stack) - 1:
            model.velocity_stack[i].load_state_dict(old_model.velocity_stack[i].state_dict())
        elif i < len(model.velocity_stack):
            linear = old_model.velocity_stack[i]
            layer_new = duplicate_linear_pop(linear, n_instruments)
            model.velocity_stack[i].load_state_dict(layer_new.state_dict())