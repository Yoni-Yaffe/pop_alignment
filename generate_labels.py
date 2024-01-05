from onsets_and_frames import *
import soundfile
from torch.nn import DataParallel
from onsets_and_frames.transcriber import load_weights
from onsets_and_frames.midi_utils import frames2midi, extract_notes_np, extract_notes_np_rescaled
import sys
import yaml
import os
import torch.nn.functional as F
from einops import rearrange
from onsets_and_frames.transcriber import ModulatedOnsetsAndFrames, ModulatedOnsetsAndFramesGroup
from onsets_and_frames.utils import max_inst

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


def inference_single_flac(transcriber, flac_path, inst_mapping, out_dir, modulated_transcriber=False, use_max_inst=True, pitch_transcriber=None, mask=None, save_onsets_and_frames=False, onset_threshold_vec=None):
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
        
        pitch_onsets_preds = []
        pitch_offset_preds = []
        pitch_frame_preds = []
        pitch_vel_preds = []
        for i_s in range(n_segments):
            curr = audio_inp[i_s * seg_len: (i_s + 1) * seg_len].unsqueeze(0).cuda()
            curr_mel = melspectrogram(curr.reshape(-1, curr.shape[-1])[:, :-1]).transpose(-1, -2)
            if modulated_inst or modulated_inst_and_group:
                batch_mel = curr_mel.repeat(len(inst_mapping) + 1, 1, 1)
                instruments = F.one_hot(torch.arange(len(inst_mapping) + 1)).to(torch.float32).to('cuda')
                if modulated_inst:
                    curr_onset_pred, curr_offset_pred, _, curr_frame_pred, *_ = transcriber(batch_mel, instruments)
                else:
                    groups = torch.zeros((len(inst_mapping) + 1, n_groups)).to(torch.float32).to('cuda')
                    curr_onset_pred, curr_offset_pred, _, curr_frame_pred, *_ = transcriber(batch_mel, instruments, groups)
                    
                curr_onset_pred = rearrange(curr_onset_pred, 'i t n -> 1 t (i n)')
                curr_frame_pred = rearrange(curr_frame_pred.max(axis=0)[0], 't n -> 1 t n')
                curr_offset_pred= curr_offset_pred[:1]
                    
                
            else:
                curr_onset_pred, curr_offset_pred, _, curr_frame_pred, *_ = transcriber(curr_mel)
            if pitch_transcriber is not None:
                pitch_curr_onset_pred, pitch_curr_offset_pred, _, pitch_curr_frame_pred, *_ = pitch_transcriber(curr_mel)
                pitch_onsets_preds.append(pitch_curr_onset_pred)
                print("pitch curr onset pred shape", pitch_curr_onset_pred.shape)
                print("pitch curr frame pred shape", pitch_curr_frame_pred.shape)
                
                pitch_offset_preds.append(pitch_curr_offset_pred)
                pitch_frame_preds.append(pitch_curr_frame_pred)
                # pitch_vel_preds.append(pitch_curr_velocity_pred)
            onsets_preds.append(curr_onset_pred)
            offset_preds.append(curr_offset_pred)
            frame_preds.append(curr_frame_pred)
            # vel_preds.append(curr_velocity_pred)
        onset_pred = torch.cat(onsets_preds, dim=1)
        offset_pred = torch.cat(offset_preds, dim=1)
        frame_pred = torch.cat(frame_preds, dim=1)
        # velocity_pred = torch.cat(vel_preds, dim=1)
        if pitch_transcriber is not None:
            pitch_onset_pred = torch.cat(pitch_onsets_preds, dim=1)
            pitch_offset_pred = torch.cat(pitch_offset_preds, dim=1)
            pitch_frame_pred = torch.cat(pitch_frame_preds, dim=1)
            # pitch_velocity_pred = torch.cat(pitch_vel_preds, dim=1)
            
    else:
        print("didn't have to split")
        audio_inp = audio_inp.unsqueeze(0).cuda()
        mel = melspectrogram(audio_inp.reshape(-1, audio_inp.shape[-1])[:, :-1]).transpose(-1, -2)
        if modulated_inst or modulated_inst_and_group:
            batch_mel = mel.repeat(len(inst_mapping) + 1, 1, 1)
            instruments = F.one_hot(torch.arange(len(inst_mapping) + 1)).to(torch.float32).to('cuda')
            if modulated_inst:
                onset_pred, offset_pred, _, frame_pred, *_ = transcriber(batch_mel, instruments)
            else:
                groups = torch.zeros((len(inst_mapping) + 1, n_groups)).to(torch.float32).to('cuda')
                onset_pred, offset_pred, _, frame_pred, *_ = transcriber(batch_mel, instruments, groups)
            offset_pred = offset_pred[:1]
            onset_pred = rearrange(onset_pred, 'i t n -> 1 t (i n)')
            frame_pred = rearrange(frame_pred.max(axis=0)[0], 't n -> 1 t n')
        else:
            onset_pred, offset_pred, _, frame_pred, *_ = transcriber(mel)
            if pitch_transcriber is not None:
                pitch_onset_pred, pitch_offset_pred, _, pitch_frame_pred, *_ = pitch_transcriber(mel)
                
                
                

    onset_pred = onset_pred.detach().squeeze().cpu()
    frame_pred = frame_pred.detach().squeeze().cpu()
    

    onset_pred_np = onset_pred.numpy()
    frame_pred_np = frame_pred.numpy()

    if mask is not None:
        mask_with_pitch = mask + [1]
        mask_list = [np.full((onset_pred_np.shape[0], N_KEYS), i) for i in mask_with_pitch]
        mask_array = np.hstack(mask_list)
        print("mask shape", mask_array.shape)
        print("mask array:", mask_array)
        assert onset_pred_np.shape == mask_array.shape
        onset_pred_np = mask_array * onset_pred_np

    if pitch_transcriber is not None:
        pitch_onset_pred = pitch_onset_pred.detach().squeeze().cpu()
        pitch_frame_pred = pitch_frame_pred.detach().squeeze().cpu()
        pitch_onset_pred_np = pitch_onset_pred.numpy()
        onset_pred_np[:, -88:] = pitch_onset_pred_np[:, -88:]
        frame_pred_np = pitch_frame_pred.numpy()
        
        

    # save_path = 'Champions_League.mid'
    save_path = os.path.join(out_dir, os.path.basename(flac_path).replace('.flac', '.mid'))
    
    inst_only = len(inst_mapping) * N_KEYS
    print("inst only")
    print("onset pred shape", onset_pred_np.shape)
    if use_max_inst and len(inst_mapping) > 1:
        print("used max inst !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        onset_pred_np = max_inst(onset_pred_np, threshold_vec=onset_threshold_vec)
        onset_threshold_vec = None
        # onset_pred_np = np.maximum(onset_pred_np, max_inst(onset_pred_np))
    if len(inst_mapping) == 1:
        print("onset_pred_np_shape_before", onset_pred_np.shape)
        onset_pred_np = onset_pred_np[:,-88:] 
        print("onset_pred_np_shape_after", onset_pred_np.shape)
    frames2midi(save_path,
                onset_pred_np[:, : inst_only], frame_pred_np[:, : inst_only],
                64. * onset_pred_np[:, : inst_only],
                inst_mapping=inst_mapping, onset_threshold_vec=onset_threshold_vec)

    if save_onsets_and_frames:
        onset_save_path = os.path.join(out_dir, os.path.basename(flac_path).replace('.flac', '_onset_pred.npy'))
        frame_save_path = os.path.join(out_dir, os.path.basename(flac_path).replace('.flac', '_frame_pred.npy'))
        np.save(onset_save_path, onset_pred_np)
        np.save(frame_save_path, frame_pred_np)
    
    
    print(f"saved midi to {save_path}")
    return save_path

def generate_labels(transcriber_ckpt, flac_dir, config, pitch_ckpt=None, mask=None, onset_threshold_vec=None, save_onsets_and_frames=False):
    # inst_mapping = [0, 68, 70, 71, 40, 73, 41, 42, 43, 45, 6, 60]
    # inst_mapping = [0, 68, 70, 71, 40, 73, 41, 42, 45, 6, 60]
    if onset_threshold_vec is not None:
        print("using threshold vec - ")
        print(onset_threshold_vec)
        
    pitch_transcriber = None
    pitch_parallel_transcriber = None
    inst_mapping = config['inst_mapping']

    num_instruments = len(inst_mapping)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device {device}")
    transcriber = torch.load(transcriber_ckpt).to(device)

    set_diff(transcriber.frame_stack, False)
    set_diff(transcriber.offset_stack, False)
    set_diff(transcriber.combined_stack, False)
    if hasattr(transcriber, 'velocity_stack'):
        set_diff(transcriber.velocity_stack, False)
    print("transcriber: ", transcriber)

    parallel_transcriber = DataParallel(transcriber)

    transcriber.zero_grad()
    if pitch_ckpt:
        pitch_transcriber = torch.load(pitch_ckpt).to(device)
        set_diff(pitch_transcriber.frame_stack, False)
        set_diff(pitch_transcriber.offset_stack, False)
        set_diff(pitch_transcriber.combined_stack, False)
        if hasattr(pitch_transcriber, 'velocity_stack'):
            set_diff(pitch_transcriber.velocity_stack, False)
    

        pitch_parallel_transcriber = DataParallel(pitch_transcriber)

        pitch_transcriber.zero_grad()
        pitch_transcriber.eval()
        pitch_parallel_transcriber.eval()
    transcriber.eval()
    parallel_transcriber.eval()
    print("pitch transcriber", pitch_parallel_transcriber)
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
                                  modulated_transcriber=config['modulated_transcriber'],
                                  pitch_transcriber=pitch_parallel_transcriber,
                                  mask=mask,
                                  onset_threshold_vec=onset_threshold_vec,
                                  save_onsets_and_frames=save_onsets_and_frames)


def generate_labels_wrapper(yaml_config: dict):
    config = yaml_config['inference_params']
    config['out_dir'] = yaml_config['logdir']
    ckpt = 'ckpts/transcriber_iteration_60001.pt'
    ckpt = config['ckpt']
    if 'pitch_ckpt' in config:
        pitch_ckpt = config['pitch_ckpt']
    else:
        pitch_ckpt = None
    if 'mask' in config:
        mask = config['mask']
    else:
        mask = None
    flac_dir = config['audio_files_dir']
    onset_threshold_vec = config.get('onset_threshold_vec')
    save_onsets_and_frames = config.get('save_onsets_and_frames', False)
    # flac_path = 'Champions_League#0.flac'
    flac_path = 'Westworld#0.flac'
    generate_labels(ckpt, flac_dir, config, pitch_ckpt=pitch_ckpt, mask=mask, onset_threshold_vec=onset_threshold_vec, save_onsets_and_frames=save_onsets_and_frames)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        yaml_path = 'config.yaml'
    else:
        logdir = sys.argv[1]
        yaml_path = os.path.join(logdir, 'run_config.yaml')

    with open(yaml_path, 'r') as fp:
        yaml_config = yaml.load(fp, Loader=yaml.FullLoader)
    generate_labels_wrapper(yaml_config)