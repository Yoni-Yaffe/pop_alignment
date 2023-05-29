import os
from datetime import datetime
import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from onsets_and_frames import *
from onsets_and_frames.dataset import EMDATASET
from torch.nn import DataParallel
from onsets_and_frames.transcriber import load_weights, load_weights_pop, modulated_load_weights
import time
from conversion_maps import pop_conversion_map, classic_conversion_map
import sys
import yaml
import random
import soundfile
from onsets_and_frames import midi_utils

def set_diff(model, diff=True):
    for layer in model.children():
        for p in layer.parameters():
            p.requires_grad = diff



# def config():
#     logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S') # ckpts and midi will be saved here
#     transcriber_ckpt = 'ckpts/model_512.pt'
#     multi_ckpt = False # Flag if the ckpt was trained on pitch only or instrument-sensitive. The provided checkpoints were trained on pitch only.

#     # transcriber_ckpt = 'ckpts/'
#     # multi_ckpt = True

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     checkpoint_interval = 6 # how often to save checkpoint
#     batch_size = 8
#     sequence_length = SEQ_LEN #if HOP_LENGTH == 512 else 3 * SEQ_LEN // 4

#     iterations = 1000 # per epoch
#     learning_rate = 0.0001
#     learning_rate_decay_steps = 10000
#     clip_gradient_norm = False #3
#     epochs = 15

#     ex.observers.append(FileStorageObserver.create(logdir))

def append_to_file(path, msg):
    with open(path, 'a') as fp:
        fp.write(msg + '\n')


def train(logdir, device, iterations, checkpoint_interval, batch_size, sequence_length, learning_rate, learning_rate_decay_steps,
          clip_gradient_norm, epochs, transcriber_ckpt, multi_ckpt, config):
    
    print(f"config -  {config}")
    # Place holders
    onset_precision = None
    onset_recall = None
    pitch_onset_precision = None
    pitch_onset_recall = None

    total_run_1 = time.time()
    print(f"device {device}")
    # print_config(ex.current_run)
    # os.makedirs(logdir, exist_ok=True)
    # n_weight = 1 if HOP_LENGTH == 512 else 2
    n_weight = config['n_weight']
    # group = "group9"
    # train_data_path = '/disk4/ben/UnalignedSupervision/NoteEM_audio'
    # labels_path = '/disk4/ben/UnalignedSupervision/NoteEm_labels'
    # dataset_name = 'full_musicnet_groups_of_20'
    dataset_name = config['dataset_name']
    train_data_path = f'/vol/scratch/jonathany/datasets/{dataset_name}/noteEM_audio'
    print(f"train_data_path: {train_data_path}")
    # labels_path = f'/vol/scratch/jonathany/datasets/{dataset_name}//NoteEm_labels'
    if config['use_labels_in_dataset_dir']:
        labels_path = f'/vol/scratch/jonathany/datasets/{dataset_name}/NoteEm_labels'
    else:
        labels_path = os.path.join(logdir, 'NoteEm_labels')
    # labels_path = '/disk4/ben/UnalignedSupervision/NoteEm_512_labels'
    debug_dir = None
    if config['debug_segments']:
        debug_dir = os.path.join(logdir, 'debug_files')
        os.makedirs(debug_dir, exist_ok=True)
        
    os.makedirs(labels_path, exist_ok=True)
    score_log_path = os.path.join(logdir, "score_log.txt")
    with open(os.path.join(logdir, "score_log.txt"), 'a') as fp:
        fp.write(f"Parameters:\ndevice: {device}, iterations: {iterations}, checkpoint_interval: {checkpoint_interval},"
                 f" batch_size: {batch_size}, sequence_length: {sequence_length}, learning_rate: {learning_rate}, "
                 f"learning_rate_decay_steps: {learning_rate_decay_steps}, clip_gradient_norm: {clip_gradient_norm}, "
                 f"epochs: {epochs}, transcriber_ckpt: {transcriber_ckpt}, multi_ckpt: {multi_ckpt}, n_weight: {n_weight}\n")
    # train_groups = ['Bach Brandenburg Concerto 1 A']
    # train_groups = ['MusicNetSamples', 'new_samples']
    # train_groups = [f'em_group{i}' for i in range(1, 10)]
    # train_groups = ['full_musicnet_with_piano']
    train_groups = config['groups']
    
    conversion_map = None
    if 'use_pop_conversion_map' in config and config['use_pop_conversion_map']:
        conversion_map = pop_conversion_map.conversion_map
    elif 'use_classic_conversion_map' in config and config['use_classic_conversion_map']:
        conversion_map = classic_conversion_map.conversion_map
        
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
                            conversion_map=conversion_map,
                            pitch_shift=config['pitch_shift']
                        )
    print('len dataset', len(dataset), len(dataset.data))
    append_to_file(score_log_path, f'Dataset instruments: {dataset.instruments}')
    append_to_file(score_log_path, f'Total: {len(dataset.instruments)} instruments')

    #####
    if not multi_ckpt:
        model_complexity = 48 if '48' in transcriber_ckpt else 64
        onset_complexity = 1.5 if '70' in transcriber_ckpt else 1.0
        saved_transcriber = torch.load(transcriber_ckpt).cpu()
        # We create a new transcriber with N_KEYS classes for each instrument:

        prev_instruments = saved_transcriber.onset_stack[2].out_features // 88
        if config['modulated_transcriber']:
            transcriber = ModulatedOnsetsAndFrames(N_MELS, (MAX_MIDI - MIN_MIDI + 1),
                                        model_complexity,
                                    onset_complexity=onset_complexity, n_instruments=len(dataset.instruments) + prev_instruments).to(device)
            # We load weights from the saved pitch-only checkkpoint and duplicate the final layer as an initialization:
            modulated_load_weights(transcriber, saved_transcriber, n_instruments=len(dataset.instruments) + 1)
        else:
            
            transcriber = OnsetsAndFrames(N_MELS, (MAX_MIDI - MIN_MIDI + 1),
                                        model_complexity,
                                    onset_complexity=onset_complexity, n_instruments=len(dataset.instruments) + prev_instruments).to(device)
            # We load weights from the saved pitch-only checkkpoint and duplicate the final layer as an initialization:
            load_weights(transcriber, saved_transcriber, n_instruments=len(dataset.instruments) + 1)
            # load_weights_pop(transcriber, saved_transcriber, n_instruments=len(dataset.instruments) + prev_instruments)
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
        if config['psuedo_labels']:
            POS = 0.7
        # if epoch == 1 we do not want to make alignment
        if epochs > 1:
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
        for iteration in tqdm(range(1, iterations + 1)):
            curr_loader = loader_cycle
            batch = next(curr_loader)
            optimizer.zero_grad()
            if config['debug_segments'] and iteration % 1000 == 1:
                inst_only = len(dataset.instruments) * N_KEYS
                b_ind = random.randint(0, batch_size - 1)
                save_path = os.path.join(debug_dir, f"iteration_{iteration}_piece_{os.path.basename(batch['path'][b_ind])}")
                audio_array = batch['audio'][b_ind].detach().cpu().numpy()
                onset_array = batch['onset'][b_ind][:,:inst_only].detach().cpu().numpy()
                frame_array = batch['frame'][b_ind][:,:inst_only].detach().cpu().numpy()
                soundfile.write(f"{save_path}.flac", audio_array, SAMPLE_RATE, format='flac', subtype='PCM_24')
                midi_utils.frames2midi(f"{save_path}.mid", onset_array, frame_array, onset_array * 64, inst_mapping=dataset.instruments)
                
            if config['modulated_transcriber']:
                n_instruments = len(dataset.instruments)
                b, t, n = batch['onset'].shape
                
                active_instruments = np.arange(n_instruments)[np.array(
                    batch['onset'].any(dim=1).reshape(b, n // N_KEYS, N_KEYS).any(2).any(0)[:-1].cpu())]
                instruments = np.full(b, n_instruments)
                np.random.shuffle(active_instruments)
                num_instruments_in_batch = min(b // 2, len(active_instruments))
                instruments[:num_instruments_in_batch] = active_instruments[:num_instruments_in_batch]
                rand_instruments = np.random.choice(active_instruments, b)
                instruments[num_instruments_in_batch: b // 2] = rand_instruments[num_instruments_in_batch: b // 2]
                np.random.shuffle(instruments)
                instruments_tensor = torch.tensor(instruments, dtype=torch.int64)
                instruments_one_hot_tensor = F.one_hot(instruments_tensor).to(torch.float32)
                new_onset_label = torch.zeros((b, t, N_KEYS), dtype=torch.float32)
                for i, inst in enumerate(instruments):
                    new_onset_label[i] = batch['onset'][i, :, inst * N_KEYS: (inst + 1) * N_KEYS]
                batch['onset'] = new_onset_label.to(device)
                batch['instruments_one_hots'] = instruments_one_hot_tensor.to(device)
            transcription, transcription_losses = transcriber.run_on_batch(batch, parallel_model=parallel_transcriber,
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

            # transcription_loss = sum(transcription_losses.values())
            transcription_loss = transcription_losses['loss/onset']
            loss = transcription_loss
            loss.backward()

            if clip_gradient_norm:
                clip_grad_norm_(transcriber.parameters(), clip_gradient_norm)

            optimizer.step()
            total_loss.append(loss.item())
            print(f"loss: {sum(total_loss) / len(total_loss):.2f} Onset Precision: {onset_precision:.2f} Onset Recall {onset_recall:.2f} "
                  f"Pitch Onset Precision: {pitch_onset_precision:.2f} Pitch Onset Recall {pitch_onset_recall:.2f}")
            if epochs == 1 and iteration % 20000 == 1:
                torch.save(transcriber, os.path.join(logdir, 'transcriber_iteration_{}.pt'.format(iteration)))
                torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
                torch.save({'instrument_mapping': dataset.instruments},
                       os.path.join(logdir, 'instrument_mapping.pt'.format(iteration)))
            
            if epochs == 1 and iteration % 5000 == 1:
                score_msg = f"iteration {iteration:06d} loss: {sum(total_loss) / len(total_loss):.2f} Onset Precision:  {onset_precision:.2f} " \
                    f"Onset Recall {onset_recall:.2f} Pitch Onset Precision:  {pitch_onset_precision:.2f} " \
                    f"Pitch Onset Recall  {pitch_onset_recall:.2f}\n"
                with open(os.path.join(logdir, "score_log.txt"), 'a') as fp:
                    fp.write(score_msg)



        time_end = time.time()
        score_msg = f"epoch {epoch:02d} loss: {sum(total_loss) / len(total_loss):.2f} Onset Precision:  {onset_precision:.2f} " \
                    f"Onset Recall {onset_recall:.2f} Pitch Onset Precision:  {pitch_onset_precision:.2f} " \
                    f"Pitch Onset Recall  {pitch_onset_recall:.2f} time label update: {time.strftime('%M:%S', time.gmtime(time_end - time_start))}\n"

        save_condition = epoch % checkpoint_interval == 1
        if save_condition and epochs != 1:
            torch.save(transcriber, os.path.join(logdir, 'transcriber_{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
            torch.save({'instrument_mapping': dataset.instruments},
                       os.path.join(logdir, 'instrument_mapping.pt'.format(epoch)))
        
        with open(os.path.join(logdir, "score_log.txt"), 'a') as fp:
            fp.write(score_msg)
    total_run_2 = time.time()
    with open(os.path.join(logdir, "score_log.txt"), 'a') as fp:
        fp.write(f"Total Runtime: {time.strftime('%H:%M:%S', time.gmtime(total_run_2 - total_run_1))}\n")
    
    # keep last optimized state
    torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
    torch.save({'instrument_mapping': dataset.instruments},
                       os.path.join(logdir, 'instrument_mapping.pt'.format(epoch)))
    # shutil.copy(f"slurm_logs/slurmlog.out", os.path.join(logdir, "full_log_slurm.txt"))
    

if __name__ == '__main__':
    if len(sys.argv) == 1:
        yaml_path = 'config.yaml'
    else:
        logdir = sys.argv[1]
        yaml_path = os.path.join(logdir, 'run_config.yaml')
    
    with open(yaml_path, 'r') as fp:
        yaml_config = yaml.load(fp, Loader=yaml.FullLoader)
        
    if 'logdir' not in yaml_config:
        print('did not find a log dir')
        logdir = f"/vol/scratch/jonathany/runs/{yaml_config['run_name']}_transcriber-{datetime.now().strftime('%y%m%d-%H%M%S')}" # ckpts and midi will be saved here
        os.makedirs(logdir, exist_ok=True)
    # transcriber_ckpt = 'ckpts/model-70.pt'
    # multi_ckpt = False # Flag if the ckpt was trained on pitch only or instrument-sensitive. The provided checkpoints were trained on pitch only.
    config = yaml_config['train_params']
    transcriber_ckpt = config['transcriber_ckpt']
    multi_ckpt = config['multi_ckpt']
    checkpoint_interval = config['checkpoint_interval']
    batch_size = config['batch_size']
    iterations = config['iterations']
    learning_rate = config['learning_rate']
    learning_rate_decay_steps = config['learning_rate_decay_steps']
    clip_gradient_norm = config['clip_gradient_norm']
    epochs = config['epochs']
    
    # transcriber_ckpt = 'ckpts/'
    # multi_ckpt = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # checkpoint_interval = 6 # how often to save checkpoint
    # batch_size = 8
    sequence_length = SEQ_LEN if HOP_LENGTH == 512 else 3 * SEQ_LEN // 4

    # iterations = 1000 # per epoch
    # iterations = 100_000
    # learning_rate = 0.0001
    # learning_rate_decay_steps = 10000
    # clip_gradient_norm = 3
    # epochs = 15
    # epochs = 1

    train(logdir, device, iterations, checkpoint_interval, batch_size, sequence_length, learning_rate, learning_rate_decay_steps,
          clip_gradient_norm, epochs, transcriber_ckpt, multi_ckpt, config)
    