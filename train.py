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
from conversion_maps import pop_conversion_map, classic_conversion_map, constant_conversion_map
import sys
import yaml
import random
import soundfile
from onsets_and_frames import midi_utils
import networkx as nx
from evaluate_multi_inst import evaluate_file
from generate_labels import inference_single_flac
from contextlib import redirect_stdout
from scripts.get_runs_metadata import add_run_to_metadata_dir

def set_diff(model, diff=True):
    for layer in model.children():
        for p in layer.parameters():
            p.requires_grad = diff

META_DATA_DIR = '../../runs_metadata'

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
        
def get_max_matching(batch_onset_array):
    b, inst = batch_onset_array.shape
    adj = np.zeros((b + inst, b + inst))
    adj[:b, b:] = batch_onset_array
    adj[b:, :b] = batch_onset_array.transpose()
    G = nx.from_numpy_array(adj)
    matching = nx.max_weight_matching(G)
    d = {min(e): max(e) - b for e in matching}
    res = np.zeros(b)
    for i in range(b):
        if i in d:
            res[i] = d[i]
        else:
            if np.sum(batch_onset_array[i]) == 0:
                np.random.choice(np.arange(inst))
            else:
                res[i] = np.random.choice(np.arange(inst)[batch_onset_array[i]])
    return res

    


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
        if 'use_constant_conversion_map' in config and config['use_constant_conversion_map']:
            labels_path += '_pitch'
        else:
            labels_path += '_inst'
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
    elif 'use_constant_conversion_map' in config and config['use_constant_conversion_map']:
        conversion_map = constant_conversion_map.conversion_map
        parallel_reference_transcriber = None
    parallel_reference_transcriber = None 
    if 'reference_transcriber' in config and config['reference_transcriber']:
        reference_transcriber = torch.load(config['reference_transcriber']).to(device)
        set_diff(reference_transcriber.frame_stack, False)
        set_diff(reference_transcriber.offset_stack, False)
        set_diff(reference_transcriber.combined_stack, False)
        if hasattr(reference_transcriber, 'velocity_stack'):
            set_diff(reference_transcriber.velocity_stack, False)
        parallel_reference_transcriber = DataParallel(reference_transcriber)
    parallel_reference_inst_transcriber = None
    if 'reference_inst_transcriber' in config and config['reference_inst_transcriber']:
        reference_inst_transcriber = torch.load(config['reference_inst_transcriber']).to(device)
        set_diff(reference_inst_transcriber.frame_stack, False)
        set_diff(reference_inst_transcriber.offset_stack, False)
        set_diff(reference_inst_transcriber.combined_stack, False)
        if hasattr(reference_inst_transcriber, 'velocity_stack'):
            set_diff(reference_inst_transcriber.velocity_stack, False)
        parallel_reference_inst_transcriber = DataParallel(reference_inst_transcriber)
    
    instrument_map = None
    if 'instrument_mapping' in config:
        instrument_map = config['instrument_mapping']
    print("Conversion map:", conversion_map)
    print("Instrument map:", instrument_map)
    dataset = EMDATASET(audio_path=train_data_path,
                        tsv_path=config['tsv_dir'],
                           labels_path=labels_path,
                           groups=train_groups,
                            sequence_length=sequence_length,
                            seed=42,
                           device=DEFAULT_DEVICE,
                            instrument_map=instrument_map,
                            conversion_map=conversion_map,
                            pitch_shift=config['pitch_shift'],
                            prev_inst_mapping=config['prev_inst_mapping'],
                            keep_eval_files=config['make_evaluation'],
                            evaluation_list=config['evaluation_list'],
                            only_eval= (iterations == 0)
                            # reference_pitch_transcriber=parallel_reference_transcriber,
                            # reference_instrument_transcriber=parallel_reference_inst_transcriber
                        )
    # del parallel_reference_inst_transcriber
    # print('len dataset', len(dataset), len(dataset.data))
    append_to_file(score_log_path, f'Dataset instruments: {dataset.instruments}')
    append_to_file(score_log_path, f'Dataset groups: {train_groups}')
    append_to_file(score_log_path, f'Total: {len(dataset.instruments)} instruments')

    #####
    if not multi_ckpt:
        model_complexity = 48 if '48' in transcriber_ckpt else 64
        onset_complexity = 1.5  if 'model-70' in transcriber_ckpt else 1.0
        saved_transcriber = torch.load(transcriber_ckpt).cpu()
        # We create a new transcriber with N_KEYS classes for each instrument:

        # prev_instruments = saved_transcriber.onset_stack[2].out_features // 88
        if config['modulated_transcriber']:
            if config['group_modulation']:
                transcriber = ModulatedOnsetsAndFramesGroup(N_MELS, (MAX_MIDI - MIN_MIDI + 1),
                                            model_complexity,
                                        onset_complexity=onset_complexity, n_instruments=len(dataset.instruments) + 1,#).to(device)
                                        n_groups=len(train_groups)).to(device)
            else:

                transcriber = ModulatedOnsetsAndFrames(N_MELS, (MAX_MIDI - MIN_MIDI + 1),
                                            model_complexity,
                                        onset_complexity=onset_complexity, n_instruments=len(dataset.instruments) + 1).to(device)
            print("transcriber", transcriber)
            # We load weights from the saved pitch-only checkkpoint and duplicate the final layer as an initialization:
            modulated_load_weights(transcriber, saved_transcriber, n_instruments=len(dataset.instruments) + 1)
        else:
            # print("len dataset instruments", len(dataset.instruments))
            # print("len prev instruments", prev_instruments)
            transcriber = OnsetsAndFrames(N_MELS, (MAX_MIDI - MIN_MIDI + 1),
                                        model_complexity,
                                    onset_complexity=onset_complexity, n_instruments=len(dataset.instruments) + 1).to(device)
            # We load weights from the saved pitch-only checkkpoint and duplicate the final layer as an initialization:
            # load_weights(transcriber, saved_transcriber, n_instruments=len(dataset.instruments) + 1)
            # print("transcriber", transcriber)
            # print("saved transcriber", saved_transcriber)
            load_weights_pop(transcriber, saved_transcriber, n_instruments=len(dataset.instruments) + 1)
            
            # # a code to verify that the linear layer was loaded successfuly
            # A_old, b_old = saved_transcriber.onset_stack[-2].parameters()
            # A_new, b_new = transcriber.onset_stack[-2].parameters()
            # assert torch.all(A_new[:prev_instruments * N_KEYS].detach().cpu().eq(A_old.detach().cpu()))
            # assert torch.all(b_new[:prev_instruments * N_KEYS].detach().cpu().eq(b_old.detach().cpu()))
            # for i in range(len(dataset.instruments)):
            #     assert torch.all(A_new[N_KEYS * (i + prev_instruments): N_KEYS * (i + 1 + prev_instruments)].detach().cpu().eq(A_old.detach().cpu()[-N_KEYS:]))
            #     assert torch.all(b_new[N_KEYS * (i + prev_instruments): N_KEYS * (i + 1 + prev_instruments)].detach().cpu().eq(b_old.detach().cpu()[-N_KEYS:]))
            # print("passed tests")
    else:
        # The checkpoint is already instrument-sensitive
        transcriber = torch.load(transcriber_ckpt).to(device)

    # We recommend to train first only onset detection. This will already give good note durations because the combined stack receives
    # information from the onset stack
    set_diff(transcriber.onset_stack, True)
    if 'train_only_frame_stack' in config and config['train_only_frame_stack']:
        print("train_only_frame stack")
        set_diff(transcriber.onset_stack, False)
        set_diff(transcriber.frame_stack, True)
        
    else:
        set_diff(transcriber.frame_stack, False)
    set_diff(transcriber.offset_stack, False)
    set_diff(transcriber.combined_stack, False)
    if hasattr(transcriber, 'velocity_stack'):
        set_diff(transcriber.velocity_stack, False)

    parallel_transcriber = DataParallel(transcriber)

        
    print("parallel transcriber", parallel_transcriber)
    optimizer = torch.optim.Adam(list(transcriber.parameters()), lr=learning_rate, weight_decay=0)
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
            POS = 0.5
            NEG = 0.01
        # if epoch == 1 we do not want to make alignment
        if config['update_pts']:
            with torch.no_grad():
                
                dataset.update_pts(parallel_transcriber,
                                POS=POS,
                                NEG=NEG,
                                to_save=logdir + '/alignments', # MIDI alignments and predictions will be saved here
                                first=epoch == 1,
                                update=True,
                                BEST_BON=epoch > 5,  # after 5 epochs, update label only if bag of notes distance improved
                                reference_transcriber=parallel_reference_transcriber,
                                reference_inst_transcriber=parallel_reference_inst_transcriber
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
                
                # active_instruments = np.arange(n_instruments)[np.array(
                #     batch['onset'][: b // 2].any(dim=1).reshape(b // 2, n // N_KEYS, N_KEYS).any(2).any(0)[:-1].cpu())]
                           
                instruments = np.full(b, n_instruments)
                # num_instruments_in_batch = min(b // 2, len(active_instruments))
                # np.random.shuffle(active_instruments)
                # instruments[:num_instruments_in_batch] = active_instruments[:num_instruments_in_batch]
                # rand_instruments = np.random.choice(active_instruments, b)
                # instruments[num_instruments_in_batch: b // 2] = rand_instruments[num_instruments_in_batch: b // 2]
                onset_array = batch['onset']
                onset_array_per_inst = onset_array[: b // 2, :, :-N_KEYS].any(1).reshape(b // 2, n // N_KEYS - 1, N_KEYS).any(2).cpu().numpy()
                instruments[: b // 2] = get_max_matching(onset_array_per_inst)
                instruments_tensor = torch.tensor(instruments, dtype=torch.int64)
                instruments_one_hot_tensor = F.one_hot(instruments_tensor).to(torch.float32)
                new_onset_label = torch.zeros((b, t, N_KEYS), dtype=torch.float32)
                for i, inst in enumerate(instruments):
                    new_onset_label[i] = batch['onset'][i, :, inst * N_KEYS: (inst + 1) * N_KEYS]
                batch['onset'] = new_onset_label.to(device)
                batch['instruments_one_hots'] = instruments_one_hot_tensor.to(device)
                if config['group_modulation']:
                    batch_groups = [train_groups.index(g) for g in batch['group']]
                    group_one_hot_tensor = F.one_hot(torch.tensor(batch_groups), num_classes=len(train_groups)).to(torch.float32)
                    batch['group_one_hots'] = group_one_hot_tensor.to(device)
                # print('active_instruments: ', active_instruments)
                # print("instruments: ", instruments)
                # count = [i.detach().cpu().sum() for i in batch['onset']]
                # print("count onsets ", count)
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
            if config['modulated_transcriber']:
                print("Modulated transcriber flag")
                pitch_onset_recall = (onset_total_tp[batch_size // 2:].sum() / onset_total_p[batch_size // 2:].sum()).item()
                pitch_onset_precision = (onset_total_tp[batch_size // 2:].sum() / onset_total_pp[batch_size // 2:].sum()).item()
            else:
                pitch_onset_recall = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_p[..., -N_KEYS:].sum()).item()
                pitch_onset_precision = (onset_total_tp[..., -N_KEYS:].sum() / onset_total_pp[..., -N_KEYS:].sum()).item()
                # print("onset_total_tp shape", onset_total_tp.shape)
                # print("check 1, ", torch.equal(onset_total_tp[..., -N_KEYS:], onset_total_tp))
                # print("check 2, ", torch.equal(onset_total_pp[..., -N_KEYS:], onset_total_pp))
                # print("check 3, ", torch.equal(onset_total_p[..., -N_KEYS:], onset_total_p))
                
            # transcription_loss = sum(transcription_losses.values())
            if 'train_only_frame_stack' in config and config['train_only_frame_stack']:
                transcription_loss = transcription_losses['loss/frame']
                print("frame_Stack_loss")
            else:
                transcription_loss = transcription_losses['loss/onset']
            
            loss = transcription_loss
            loss.backward()

            if clip_gradient_norm:
                clip_grad_norm_(transcriber.parameters(), clip_gradient_norm)

            optimizer.step()
            total_loss.append(loss.item())
            print(f"avg loss: {sum(total_loss) / len(total_loss):.5f} current loss: {total_loss[-1]:.5f} Onset Precision: {onset_precision:.3f} Onset Recall {onset_recall:.3f} "
                  f"Pitch Onset Precision: {pitch_onset_precision:.3f} Pitch Onset Recall {pitch_onset_recall:.3f}")
            if epochs == 1 and iteration % 20000 == 1:
                torch.save(transcriber, os.path.join(logdir, 'transcriber_iteration_{}.pt'.format(iteration)))
                torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))
                torch.save({'instrument_mapping': dataset.instruments},
                       os.path.join(logdir, 'instrument_mapping.pt'.format(iteration)))
            
            if epochs == 1 and iteration % 5000 == 1:
                score_msg = f"iteration {iteration:06d} loss: {sum(total_loss) / len(total_loss):.5f} Onset Precision:  {onset_precision:.3f} " \
                    f"Onset Recall {onset_recall:.3f} Pitch Onset Precision:  {pitch_onset_precision:.3f} " \
                    f"Pitch Onset Recall  {pitch_onset_recall:.3f}\n"
                with open(os.path.join(logdir, "score_log.txt"), 'a') as fp:
                    fp.write(score_msg)
                    
                onset_total_tp = 0.
                onset_total_pp = 0.
                onset_total_p = 0.
            if epochs == 1 and iteration % 1000 == 1:
                torch.save(transcriber, os.path.join(logdir, 'transcriber_ckpt.pt'.format(iteration)))



        time_end = time.time()
        score_msg = f"epoch {epoch:02d} loss: {sum(total_loss) / len(total_loss):.5f} Onset Precision:  {onset_precision:.3f} " \
                    f"Onset Recall {onset_recall:.3f} Pitch Onset Precision:  {pitch_onset_precision:.3f} " \
                    f"Pitch Onset Recall  {pitch_onset_recall:.3f} time label update: {time.strftime('%M:%S', time.gmtime(time_end - time_start))}\n"

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
                       os.path.join(logdir, 'instrument_mapping.pt'))
    

    if config['make_evaluation']:
        transcriber.zero_grad()
        print("cuda mem info:")
        print(torch.cuda.mem_get_info())
        torch.cuda.empty_cache()
        eval_inference_dir = os.path.join(logdir, 'eval_inference')
        os.makedirs(eval_inference_dir, exist_ok=True)
        tsv_list = []
        midi_transcribed_list = []
        with torch.no_grad():
            file_list = dataset.eval_file_list
            if 'eval_all' in config and config['eval_all']:
                file_list = dataset.file_list 
            for flac, tsv in file_list:
                if '#0' not in  flac:
                    continue
                midi_path = inference_single_flac(parallel_transcriber, flac, dataset.instruments, eval_inference_dir, config['modulated_transcriber'])
                midi_transcribed_list.append(midi_path)
                tsv_list.append(tsv)
        output_path_evaluation = os.path.join(logdir, "evaluation_results.txt")
        with open(output_path_evaluation, 'w') as f:
            with redirect_stdout(f):
                evaluate_file(midi_transcribed_list, tsv_list, dataset.instruments, conversion_map)
        add_run_to_metadata_dir(logdir, META_DATA_DIR)
    if 'train_inference' in config and config['train_inference']:
        train_inference_path = os.path.join(logdir, 'train_inference')
        os.makedirs(train_inference_path)
        for f, _ in dataset.file_list:
            inference_single_flac(parallel_transcriber, f, dataset.instruments, train_inference_path, config['modulated_transcriber'])
        
            
    
    
    
def train_wrapper(yaml_config: dict, logdir):
    # if len(sys.argv) == 1:
    #     yaml_path = 'config.yaml'
    # else:
    #     logdir = sys.argv[1]
    #     yaml_path = os.path.join(logdir, 'run_config.yaml')
    # print("yaml path:", yaml_path)
    # with open(yaml_path, 'r') as fp:
    #     yaml_config = yaml.load(fp, Loader=yaml.FullLoader)
        
    # if 'logdir' not in yaml_config:
    #     print('did not find a log dir')
    #     logdir = f"/vol/scratch/jonathany/runs/{yaml_config['run_name']}_transcriber-{datetime.now().strftime('%y%m%d-%H%M%S')}" # ckpts and midi will be saved here
    #     os.makedirs(logdir, exist_ok=True)
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
        
    train_wrapper(yaml_config, logdir)
    