from onsets_and_frames import *
import soundfile
from torch.nn import DataParallel
from onsets_and_frames.transcriber import load_weights
from onsets_and_frames.midi_utils import frames2midi
from train import set_diff

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


def generate_labels(transcriber_ckpt, flac_path):
    inst_mapping = [0, 68, 70, 71, 40, 73, 41, 42, 43, 45, 6, 60]
    num_instruments = len(inst_mapping)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model_complexity = 64
    # saved_transcriber = torch.load(transcriber_ckpt).cpu()
    # # We create a new transcriber with N_KEYS classes for each instrument:
    # transcriber = OnsetsAndFrames(N_MELS, (MAX_MIDI - MIN_MIDI + 1),
    #                               model_complexity,
    #                               onset_complexity=1.5, n_instruments=num_instruments + 1).to(device)
    # # We load weights from the saved pitch-only checkkpoint and duplicate the final layer as an initialization:
    # load_weights(transcriber, saved_transcriber, n_instruments=num_instruments + 1)
    transcriber = torch.load(transcriber_ckpt).to(device)

    set_diff(transcriber.frame_stack, False)
    set_diff(transcriber.offset_stack, False)
    set_diff(transcriber.combined_stack, False)
    set_diff(transcriber.velocity_stack, False)

    parallel_transcriber = DataParallel(transcriber)
    audio = load_audio(flac_path)
    transcriber.zero_grad()

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
            curr = audio_inp[i_s * seg_len: (i_s + 1) * seg_len].unsqueeze(0)  # .cuda()
            curr_mel = melspectrogram(curr.reshape(-1, curr.shape[-1])[:, :-1]).transpose(-1, -2)
            curr_onset_pred, curr_offset_pred, _, curr_frame_pred, curr_velocity_pred = parallel_transcriber(curr_mel)
            onsets_preds.append(curr_onset_pred)
            offset_preds.append(curr_offset_pred)
            frame_preds.append(curr_frame_pred)
            vel_preds.append(curr_velocity_pred)
        onset_pred = torch.cat(onsets_preds, dim=1)
        offset_pred = torch.cat(offset_preds, dim=1)
        frame_pred = torch.cat(frame_preds, dim=1)
        velocity_pred = torch.cat(vel_preds, dim=1)

        onset_pred = onset_pred.detach().squeeze().cpu()
        frame_pred = frame_pred.detach().squeeze().cpu()

        peaks = get_peaks(onset_pred, 3)  # we only want local peaks, in a 7-frame neighborhood, 3 to each side.
        onset_pred[~peaks] = 0

        onset_pred_np = onset_pred.numpy()
        frame_pred_np = frame_pred.numpy()

        save_path = 'test_predict.mid'

        inst_only = len(inst_mapping) * N_KEYS
        frames2midi(save_path,
                    onset_pred_np[:, : inst_only], frame_pred_np[:, : inst_only],
                    64. * onset_pred_np[:, : inst_only],
                    inst_mapping=inst_mapping)

if __name__ == '__main__':
    ckpt = 'ckpts/transcriber_iteration_1.pt'
    flac_path = '1789#0.flac'
    generate_labels(ckpt, flac_path)
