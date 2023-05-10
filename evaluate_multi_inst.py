import sys
from collections import defaultdict
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval_multi import precision_recall_f1_overlap as evaluate_notes_with_instrument
from scipy.stats import hmean
from tqdm import tqdm
from onsets_and_frames import *
from onsets_and_frames.midi_utils import *
import os

eps = sys.float_info.epsilon


def midi2hz(midi):
    res = 440.0 * (2.0 ** ((midi - 69.0)/12.0))
    return res


def evaluate(synthsized_midis, reference_midis, instruments=None):
    metrics = defaultdict(list)

    counter = 0

    for transcribed, reference in tqdm(zip(synthsized_midis, reference_midis)):
        print('eval for', transcribed.split('/')[-1], reference.split('/')[-1])
        counter += 1
        transcribed_events = parse_midi_multi(transcribed)
        reference_events = parse_midi_multi(reference)

        max_time = int(reference_events[:, 1].max() + 5)
        audio_length = max_time * SAMPLE_RATE
        n_keys = MAX_MIDI - MIN_MIDI + 1

        n_steps = (audio_length - 1) // HOP_LENGTH + 1
        n_steps_transcriber = (audio_length - 1) // HOP_LENGTH + 1

        p_est, i_est, v_est, ins_est = transcribed_events[:, 2], transcribed_events[:, 0: 2], transcribed_events[:, 3], transcribed_events[:, 4]
        p_ref, i_ref, v_ref, ins_ref = reference_events[:, 2], reference_events[:, 0: 2], reference_events[:, 3], reference_events[:, 4]

        p_ref_2 = np.array([int(midi - MIN_MIDI) for midi in p_ref])
        i_ref_2 = np.array([(int(round(on * SAMPLE_RATE / HOP_LENGTH)), int(round(off * SAMPLE_RATE / HOP_LENGTH))) for on, off in i_ref])
        p_est_2 = np.array([int(midi - MIN_MIDI) for midi in p_est])
        i_est_2 = np.array([(int(round(on * SAMPLE_RATE / HOP_LENGTH)), int(round(off * SAMPLE_RATE / HOP_LENGTH))) for on, off in i_est])

        t_ref, f_ref = notes_to_frames(p_ref_2, i_ref_2, (n_steps, n_keys))
        t_est, f_est = notes_to_frames(p_est_2, i_est_2, (n_steps_transcriber, n_keys))

        scaling = HOP_LENGTH / SAMPLE_RATE

        p_ref = np.array([midi2hz(midi) for midi in p_ref])
        p_est = np.array([midi2hz(midi) for midi in p_est])


        t_ref = t_ref.astype(np.float64) * scaling
        f_ref = [np.array([midi2hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
        t_est = t_est.astype(np.float64) * scaling
        f_est = [np.array([midi2hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None, onset_tolerance=0.05)

        print('onset:', p, r, f, o)
        on_p, on_r, on_f = p, r, f


        metrics['metric/note/precision'].append(p)
        metrics['metric/note/recall'].append(r)
        metrics['metric/note/f1'].append(f)
        metrics['metric/note/overlap'].append(o)

        p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
        print('onset-offset:', p, r, f, o)

        metrics['metric/note-with-offsets/precision'].append(p)
        metrics['metric/note-with-offsets/recall'].append(r)
        metrics['metric/note-with-offsets/f1'].append(f)
        metrics['metric/note-with-offsets/overlap'].append(o)

        p, r, f, o = evaluate_notes_with_instrument(i_ref, p_ref, ins_ref, i_est, p_est, ins_est,
                                                    offset_ratio=None)
        print('onset-instrument:', p, r, f, o)
        oni_p, oni_r, oni_f = p, r, f
        metrics['metric/note-with-instrument/precision'].append(p)
        metrics['metric/note-with-instrument/recall'].append(r)
        metrics['metric/note-with-instrument/f1'].append(f)
        metrics['metric/note-with-instrument/overlap'].append(o)

        # if instruments is not None:
        #     for inst in range(instruments):
        #         print('inst', inst)
        #         ins_ref_curr = ins_ref == inst
        #         if ins_ref_curr.sum() == 0:
        #             print('skipping', inst)
        #             continue
        #         ins_est_curr = ins_est == inst
        #         curr_args = []
        #         for arg in i_ref, p_ref:
        #             curr_args.append(arg[ins_ref_curr])
        #         for arg in i_est, p_est:
        #             curr_args.append(arg[ins_est_curr])
        #
        #         p, r, f, o = evaluate_notes(curr_args[0], curr_args[1], curr_args[2], curr_args[3],
        #                                     offset_ratio=None)
        #         metrics['metric/note-with-instrument' + str(inst) + '/precision'].append(p)
        #         metrics['metric/note-with-instrument' + str(inst) + '/recall'].append(r)
        #         metrics['metric/note-with-instrument' + str(inst) + '/f1'].append(f)
        #         metrics['metric/note-with-instrument' + str(inst) + '/overlap'].append(o)
        #         print('inst', inst, 'p, r, f, o', p, r, f, o)


        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                                  velocity_tolerance=0.1, offset_ratio=None)
        print('onset-velocity:', p, r, f, o)
        metrics['metric/note-with-velocity/precision'].append(p)
        metrics['metric/note-with-velocity/recall'].append(r)
        metrics['metric/note-with-velocity/f1'].append(f)
        metrics['metric/note-with-velocity/overlap'].append(o)

        p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
        print('onset-offset-velocity:', p, r, f, o)
        metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
        metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
        metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
        metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)

        frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
        fr_p, fr_r, fr_f = frame_metrics['Precision'], frame_metrics['Recall'], hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps
        metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)
        print('frame p/r/f:', frame_metrics['Precision'], frame_metrics['Recall'], hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

        print('overleaf:', '{}/{}/{}'.format(round(100 * on_f, 1), round(100 * oni_f, 1), round(100 * fr_f, 1)))
        for key, loss in frame_metrics.items():
            metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

        torch.cuda.empty_cache()


    return metrics


def evaluate_file(synthsized_midis, reference_midis, instruments=None):
    metrics = evaluate(synthsized_midis, reference_midis, instruments=instruments)
    # metrics = evaluate(tqdm(dataset), (model, parallel_model), onset_threshold, frame_threshold, save_path)


    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')



if __name__ == '__main__':
    transcribed_midis = os.listdir("evaluation/midis_no_pitch_shift_transcribed")
    reference_midis = os.listdir("evaluation/midis_no_pitch_shift_original")
    instruments = [0, 6, 10, 19, 24, 40, 41, 42, 43, 44, 45, 46, 47, 48, 52, 58, 60, 61, 64, 68, 70, 71, 73]
    instruments = [0, 68, 70, 71, 40, 73, 41, 42, 45, 6, 60]
    evaluate_file(transcribed_midis, reference_midis, instruments=instruments)

