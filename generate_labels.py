from onsets_and_frames import *
import soundfile
from torch.nn import DataParallel
from onsets_and_frames.transcriber import load_weights
from onsets_and_frames.midi_utils import frames2midi
import sys
import yaml
import os
import torch.nn.functional as F
from einops import rearrange
from onsets_and_frames.transcriber import ModulatedOnsetsAndFrames, ModulatedOnsetsAndFramesGroup


def set_diff(model, diff=True):
    for layer in model.children():
        for p in layer.parameters():
            p.requires_grad = diff


def load_audio(flac):
    audio, sr = soundfile.read(flac, dtype='int16')
    if len(audio.shape) == 2:
        audio = audio.astype(float).mean(axis=1)
    else:
        audio = audio.astype(float)
    audio = audio.astype(np.int16)
    print('audio len', len(audio))
    assert sr == SAMPLE_RATE
    audio = torch.ShortTensor(audio)
    return audio


def inference_single_flac(transcriber, flac_path, inst_mapping, out_dir, modulated_transcriber=False):
    if isinstance(transcriber, DataParallel):
        transcriber_module = transcriber.module
    else:
        transcriber_module = transcriber
    modulated_inst = isinstance(transcriber_module, ModulatedOnsetsAndFrames)
    modulated_inst_and_group = isinstance(transcriber_module, ModulatedOnsetsAndFramesGroup)
    print("modulated inst", modulated_inst)
    print("modulated inst and group", modulated_inst_and_group)
    if modulated_inst_and_group:
        n_groups = transcriber_module.mlp_group[1].in_features
    audio = load_audio(flac_path)
    audio_inp = audio.float() / 32768.
    MAX_TIME = 5 * 60 * SAMPLE_RATE
    audio_inp_len = len(audio_inp)
    if audio_inp_len > MAX_TIME:
        n_segments = 3 if audio_inp_len > 2 * MAX_TIME else 2
        print('long audio, splitting to {} segments'.format(n_segments))
        seg_len = audio_inp_len // n_segments
        onsets_preds = []
        offset_preds = []
        frame_preds = []
        vel_preds = []
        for i_s in range(n_segments):
            curr = audio_inp[i_s * seg_len: (i_s + 1) * seg_len].unsqueeze(0).cuda()
            curr_mel = melspectrogram(curr.reshape(-1, curr.shape[-1])[:, :-1]).transpose(-1, -2)
            if modulated_inst or modulated_inst_and_group:
                batch_mel = curr_mel.repeat(len(inst_mapping) + 1, 1, 1)
                instruments = F.one_hot(torch.arange(len(inst_mapping) + 1)).to(torch.float32).to('cuda')
                if modulated_inst:
                    curr_onset_pred, curr_offset_pred, _, curr_frame_pred, curr_velocity_pred = transcriber(batch_mel, instruments)
                else:
                    groups = torch.zeros((len(inst_mapping) + 1, n_groups)).to(torch.float32).to('cuda')
                    curr_onset_pred, curr_offset_pred, _, curr_frame_pred, curr_velocity_pred = transcriber(batch_mel, instruments, groups)
                    
                curr_onset_pred = rearrange(curr_onset_pred, 'i t n -> 1 t (i n)')
                curr_frame_pred = rearrange(curr_frame_pred.max(axis=0)[0], 't n -> 1 t n')
                curr_offset_pred, curr_velocity_pred = curr_offset_pred[:1], curr_velocity_pred[:1]
                    
                
            else:
                curr_onset_pred, curr_offset_pred, _, curr_frame_pred, curr_velocity_pred = transcriber(curr_mel)
            onsets_preds.append(curr_onset_pred)
            offset_preds.append(curr_offset_pred)
            frame_preds.append(curr_frame_pred)
            vel_preds.append(curr_velocity_pred)
        onset_pred = torch.cat(onsets_preds, dim=1)
        offset_pred = torch.cat(offset_preds, dim=1)
        frame_pred = torch.cat(frame_preds, dim=1)
        velocity_pred = torch.cat(vel_preds, dim=1)
    else:
        print("didn't have to split")
        audio_inp = audio_inp.unsqueeze(0).cuda()
        mel = melspectrogram(audio_inp.reshape(-1, audio_inp.shape[-1])[:, :-1]).transpose(-1, -2)
        if modulated_inst or modulated_inst_and_group:
            batch_mel = mel.repeat(len(inst_mapping) + 1, 1, 1)
            instruments = F.one_hot(torch.arange(len(inst_mapping) + 1)).to(torch.float32).to('cuda')
            if modulated_inst:
                onset_pred, offset_pred, _, frame_pred, velocity_pred = transcriber(batch_mel, instruments)
            else:
                groups = torch.zeros((len(inst_mapping) + 1, n_groups)).to(torch.float32).to('cuda')
                onset_pred, offset_pred, _, frame_pred, velocity_pred = transcriber(batch_mel, instruments, groups)
            offset_pred, velocity_pred = offset_pred[:1], velocity_pred[:1]
            onset_pred = rearrange(onset_pred, 'i t n -> 1 t (i n)')
            frame_pred = rearrange(frame_pred.max(axis=0)[0], 't n -> 1 t n')
        else:
            onset_pred, offset_pred, _, frame_pred, velocity_pred = transcriber(mel)

    onset_pred = onset_pred.detach().squeeze().cpu()
    frame_pred = frame_pred.detach().squeeze().cpu()

    peaks = get_peaks(onset_pred, 3)  # we only want local peaks, in a 7-frame neighborhood, 3 to each side.
    onset_pred[~peaks] = 0

    onset_pred_np = onset_pred.numpy()
    # onset_pred_np[:, :-N_KEYS * 5] = 0
    frame_pred_np = frame_pred.numpy()

    # save_path = 'Champions_League.mid'
    save_path = os.path.join(out_dir, os.path.basename(flac_path).replace('.flac', '.mid'))

    inst_only = len(inst_mapping) * N_KEYS
    frames2midi(save_path,
                onset_pred_np[:, : inst_only], frame_pred_np[:, : inst_only],
                64. * onset_pred_np[:, : inst_only],
                inst_mapping=inst_mapping)
    print(f"saved midi to {save_path}")
    return save_path

def generate_labels(transcriber_ckpt, flac_dir, config):
    # inst_mapping = [0, 68, 70, 71, 40, 73, 41, 42, 43, 45, 6, 60]
    # inst_mapping = [0, 68, 70, 71, 40, 73, 41, 42, 45, 6, 60]
    inst_mapping = config['inst_mapping']

    num_instruments = len(inst_mapping)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device {device}")
    transcriber = torch.load(transcriber_ckpt).to(device)

    set_diff(transcriber.frame_stack, False)
    set_diff(transcriber.offset_stack, False)
    set_diff(transcriber.combined_stack, False)
    set_diff(transcriber.velocity_stack, False)

    parallel_transcriber = DataParallel(transcriber)

    transcriber.zero_grad()
    print("cuda mem info:")
    print(torch.cuda.mem_get_info())
    torch.cuda.empty_cache()
    print("cuda mem info:")
    print(torch.cuda.mem_get_info())
    flac_path_list = [os.path.join(flac_dir, f) for f in os.listdir(flac_dir) if f.endswith('.flac')]
    results_dir = os.path.join(config['out_dir'], 'results')
    os.makedirs(results_dir, exist_ok=True)
    with torch.no_grad():
        for flac_path in flac_path_list:
            inference_single_flac(transcriber=parallel_transcriber,
                                  flac_path=flac_path,
                                  inst_mapping=inst_mapping, 
                                  out_dir=results_dir,
                                  modulated_transcriber=config['modulated_transcriber'])


def generate_labels_wrapper(yaml_config: dict):
    config = yaml_config['inference_params']
    config['out_dir'] = yaml_config['logdir']
    ckpt = 'ckpts/transcriber_iteration_60001.pt'
    ckpt = config['ckpt']
    flac_dir = config['audio_files_dir']
    # flac_path = 'Champions_League#0.flac'
    flac_path = 'Westworld#0.flac'
    generate_labels(ckpt, flac_dir, config)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        yaml_path = 'config.yaml'
    else:
        logdir = sys.argv[1]
        yaml_path = os.path.join(logdir, 'run_config.yaml')

    with open(yaml_path, 'r') as fp:
        yaml_config = yaml.load(fp, Loader=yaml.FullLoader)
    generate_labels_wrapper(yaml_config)