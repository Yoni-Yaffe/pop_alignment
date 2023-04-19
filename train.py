import os
from datetime import datetime
import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from onsets_and_frames import *
from onsets_and_frames.dataset import EMDATASET
from torch.nn import DataParallel
from onsets_and_frames.transcriber import load_weights
import time
import shutil
import pop_conversion_map

def set_diff(model, diff=True):
    for layer in model.children():
        for p in layer.parameters():
            p.requires_grad = diff


ex = Experiment('train_transcriber')



def config():
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S') # ckpts and midi will be saved here
    transcriber_ckpt = 'ckpts/model_512.pt'
    multi_ckpt = False # Flag if the ckpt was trained on pitch only or instrument-sensitive. The provided checkpoints were trained on pitch only.

    # transcriber_ckpt = 'ckpts/'
    # multi_ckpt = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_interval = 6 # how often to save checkpoint
    batch_size = 8
    sequence_length = SEQ_LEN #if HOP_LENGTH == 512 else 3 * SEQ_LEN // 4

    iterations = 1000 # per epoch
    learning_rate = 0.0001
    learning_rate_decay_steps = 10000
    clip_gradient_norm = False #3
    epochs = 15

    ex.observers.append(FileStorageObserver.create(logdir))

def append_to_file(path, msg):
    with open(path, 'a') as fp:
        fp.write(msg + '\n')


def train(logdir, device, iterations, checkpoint_interval, batch_size, sequence_length, learning_rate, learning_rate_decay_steps,
          clip_gradient_norm, epochs, transcriber_ckpt, multi_ckpt):
    total_run_1 = time.time()
    print(f"device {device}")
    # print_config(ex.current_run)
    os.makedirs(logdir, exist_ok=True)
    n_weight = 3 if HOP_LENGTH == 512 else 2
    # train_data_path = '/disk4/ben/UnalignedSupervision/NoteEM_audio'
    # labels_path = '/disk4/ben/UnalignedSupervision/NoteEm_labels'
    dataset_name = 'no_solo_group1'
    train_data_path = f'/vol/scratch/jonathany/{dataset_name}/noteEM_audio'
    labels_path = f'/vol/scratch/jonathany/{dataset_name}/NoteEm_labels'
    # labels_path = '/disk4/ben/UnalignedSupervision/NoteEm_512_labels'

    os.makedirs(labels_path, exist_ok=True)
    score_log_path = os.path.join(logdir, "score_log.txt")
    with open(os.path.join(logdir, "score_log.txt"), 'a') as fp:
        fp.write(f"Parameters:\ndevice: {device}, iterations: {iterations}, checkpoint_interval: {checkpoint_interval},"
                 f" batch_size: {batch_size}, sequence_length: {sequence_length}, learning_rate: {learning_rate}, "
                 f"learning_rate_decay_steps: {learning_rate_decay_steps}, clip_gradient_norm: {clip_gradient_norm}, "
                 f"epochs: {epochs}, transcriber_ckpt: {transcriber_ckpt}, multi_ckpt: {multi_ckpt}, n_weight: {n_weight}\n")
    # train_groups = ['Bach Brandenburg Concerto 1 A']
    # train_groups = ['MusicNetSamples', 'new_samples']
    train_groups = [dataset_name]

    conversion_map = None
    if 'pop' in dataset_name:
        conversion_map = pop_conversion_map.conversion_map
    instrument_map = None
    print("Conversion map:", conversion_map)
    print("Instrument map:", instrument_map)
    dataset = EMDATASET(audio_path=train_data_path,
                           labels_path=labels_path,
                           groups=train_groups,
                            sequence_length=sequence_length,
                            seed=42,
                           device=DEFAULT_DEVICE,
                            instrument_map=instrument_map,
                            conversion_map=conversion_map
                        )
    print('len dataset', len(dataset), len(dataset.data))
    append_to_file(score_log_path, f'Dataset instruments: {dataset.instruments}')
    append_to_file(score_log_path, f'Total: {len(dataset.instruments)} instruments')

    #####
    if not multi_ckpt:
        model_complexity = 64 if '64' in transcriber_ckpt else 48
        saved_transcriber = torch.load(transcriber_ckpt).cpu()
        # We create a new transcriber with N_KEYS classes for each instrument:
        transcriber = OnsetsAndFrames(N_MELS, (MAX_MIDI - MIN_MIDI + 1),
                                              model_complexity,
                                    onset_complexity=1., n_instruments=len(dataset.instruments) + 1).to(device)
        # We load weights from the saved pitch-only checkkpoint and duplicate the final layer as an initialization:
        load_weights(transcriber, saved_transcriber, n_instruments=len(dataset.instruments) + 1)
    else:
        # The checkpoint is already instrument-sensitive
        transcriber = torch.load(transcriber_ckpt).to(device)

    # We recommend to train first only onset detection. This will already give good note durations because the combined stack receives
    # information from the onset stack
    set_diff(transcriber.frame_stack, False)
    set_diff(transcriber.offset_stack, False)
    set_diff(transcriber.combined_stack, False)
    set_diff(transcriber.velocity_stack, False)

    parallel_transcriber = DataParallel(transcriber)
    optimizer = torch.optim.Adam(list(transcriber.parameters()), lr=learning_rate, weight_decay=1e-5)
    transcriber.zero_grad()
    optimizer.zero_grad()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            ghost = torch.ones((100, 100), dtype=float)#.to('cuda:1') # occupy another gpu until transcriber training begins
        print('epoch', epoch)
        if epoch > 1:
            del loader
            del batch
        torch.cuda.empty_cache()

        POS = 1.1 # Pseudo-label positive threshold (value > 1 means no pseudo label).
        NEG = -0.1 # Pseudo-label negative threshold (value < 0 means no pseudo label).
        # POS = 0.7 # Pseudo-label positive threshold (value > 1 means no pseudo label).
        # NEG = -0.1 # Pseudo-label negative threshold (value < 0 means no pseudo label).
        with torch.no_grad():
            dataset.update_pts(parallel_transcriber,
                               POS=POS,
                               NEG=NEG,
                               to_save=logdir + '/alignments', # MIDI alignments and predictions will be saved here
                               first=epoch == 1,
                               update=True,
                               BEST_BON=epoch > 5  # after 5 epochs, update label only if bag of notes distance improved
                               )
        loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

        total_loss = []
        transcriber.train()

        onset_total_tp = 0.
        onset_total_pp = 0.
        onset_total_p = 0.

        if epoch == 1:
            del ghost
            torch.cuda.empty_cache()

        loader_cycle = cycle(loader)
        time_start = time.time()
        for _ in tqdm(range(iterations)):
            curr_loader = loader_cycle
            batch = next(curr_loader)
            optimizer.zero_grad()

            transcription, transcription_losses = transcriber.run_on_batch(batch, parallel_transcriber,
                                                                           positive_weight=n_weight,
                                                                           inv_positive_weight=n_weight,
                                                                           )
            onset_pred = transcription['onset'].detach() > 0.5
            onset_total_pp += onset_pred
            onset_tp = onset_pred * batch['onset'].detach()
            onset_total_tp += onset_tp
            onset_total_p += batch['onset'].detach()

            onset_recall = (onset_total_tp.sum() / onset_total_p.sum()).item()
            onset_precision = (onset_total_tp.sum() / onset_total_pp.sum()).item()

            pitch_onset_recall = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_p[..., -N_KEYS:].sum()).item()
            pitch_onset_precision = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_pp[..., -N_KEYS:].sum()).item()

            transcription_loss = sum(transcription_losses.values())
            loss = transcription_loss
            loss.backward()

            if clip_gradient_norm:
                clip_grad_norm_(transcriber.parameters(), clip_gradient_norm)

            optimizer.step()
            total_loss.append(loss.item())
            print('loss:', sum(total_loss) / len(total_loss), 'Onset Precision:', onset_precision, 'Onset Recall', onset_recall,
                                                            'Pitch Onset Precision:', pitch_onset_precision, 'Pitch Onset Recall', pitch_onset_recall)
        time_end = time.time()
        score_msg = f'epoch {epoch} loss: {sum(total_loss) / len(total_loss)} Onset Precision:  {onset_precision} ' \
                    f'Onset Recall {onset_recall} Pitch Onset Precision:  {pitch_onset_precision}' \
                    f'  Pitch Onset Recall  {pitch_onset_recall} time label update: {time.strftime('%M:%S', time.gmtime(time_end - time_start))}\n'

        save_condition = epoch % checkpoint_interval == 1
        if save_condition:
            torch.save(transcriber, os.path.join(logdir, 'transcriber_{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
            torch.save({'instrument_mapping': dataset.instruments},
                       os.path.join(logdir, 'instrument_mapping.pt'.format(epoch)))
        
        with open(os.path.join(logdir, "score_log.txt"), 'a') as fp:
            fp.write(score_msg)
    total_run_2 = time.time()
    with open(os.path.join(logdir, "score_log.txt"), 'a') as fp:
        fp.write(f"Total Runtime: {time.strftime('%H:%M:%S', time.gmtime(total_run_2 - total_run_1))}\n")
    shutil.copy("slurmlog.out", os.path.join(logdir, "full_log_slurm.txt"))
    


if __name__ == '__main__':
    run_name = "no_solo_group1"
    logdir = f"/vol/scratch/jonathany/runs/{run_name}_transcriber-{datetime.now().strftime('%y%m%d-%H%M%S')}" # ckpts and midi will be saved here
    transcriber_ckpt = '/vol/scratch/jonathany/ckpts/model_64.pt'
    # transcriber_ckpt = 'ckpts/model_64.pt'
    multi_ckpt = False # Flag if the ckpt was trained on pitch only or instrument-sensitive. The provided checkpoints were trained on pitch only.

    # transcriber_ckpt = 'ckpts/'
    # multi_ckpt = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_interval = 6 # how often to save checkpoint
    batch_size = 8
    sequence_length = SEQ_LEN if HOP_LENGTH == 512 else 3 * SEQ_LEN // 4

    iterations = 1000 # per epoch
    learning_rate = 0.0001
    learning_rate_decay_steps = 10000
    clip_gradient_norm = 3
    epochs = 15

    train(logdir, device, iterations, checkpoint_interval, batch_size, sequence_length, learning_rate, learning_rate_decay_steps,
          clip_gradient_norm, epochs, transcriber_ckpt, multi_ckpt)
