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
from onsets_and_frames.decoding import notes_to_frames
from onsets_and_frames.midi_utils import extract_notes_np
import os
from onsets_and_frames.constants import MIN_MIDI, MAX_MIDI
import json 


eps = sys.float_info.epsilon


def midi2hz(midi):
    res = 440.0 * (2.0 ** ((midi - 69.0)/12.0))
    return res


def evaluate(synthsized_midis, reference_midis, instruments=None, conversion_map=None, tolerance=0.05):
    metrics = defaultdict(list)

    counter = 0

    for transcribed, reference in tqdm(zip(synthsized_midis, reference_midis)):
        print('eval for', transcribed.split('/')[-1], reference.split('/')[-1])
        counter += 1
        if reference.endswith('.tsv'):
            reference_events = np.loadtxt(reference, delimiter='\t', skiprows=1)
        else:
            reference_events = parse_midi_multi(reference)   
        transcribed_events = parse_midi_multi(transcribed)
        
        # transcribed_events = extra
        if conversion_map is not None:
            convert_func = np.vectorize(lambda x: conversion_map.get(x, x))
            reference_events[:, 4] = convert_func(reference_events[:, 4])
            transcribed_events[:, 4] = convert_func(transcribed_events[:, 4])
            

        max_time = int(reference_events[:, 1].max() + 5)
        audio_length = max_time * SAMPLE_RATE
        n_keys = MAX_MIDI - MIN_MIDI + 1

        n_steps = (audio_length - 1) // HOP_LENGTH + 1
        n_steps_transcriber = (audio_length - 1) // HOP_LENGTH + 1
        
        p_est, i_est, v_est, ins_est = transcribed_events[:, 2], transcribed_events[:, 0: 2], transcribed_events[:, 3], transcribed_events[:, 4]
        
        # onset_pred = np.load(transcribed.replace('.mid', '_onset_pred.npy'))
        # frame_pred = np.load(transcribed.replace('.mid', '_frame_pred.npy'))
        # p_est, i_est, v_est, ins_est = extract_notes_np_rescaled(onset_pred, frame_pred, onset_pred * 64)
        
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
        # print("negative interval self check", (i_ref[:,1] <= i_ref[:,0]).any())
        # print("negative interval self check", (i_est[:,1] <= i_est[:,0]).any())
        try:
            p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None, onset_tolerance=tolerance)
        except:
            print("error in file", reference)
            continue
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

def get_bag_of_notes_for_note(synthsized_midis, reference_midis, threshold, note, tolerance=0.05):
    dist_list = []
    
    for transcribed, reference in zip(synthsized_midis, reference_midis):
        onset_pred = np.load(transcribed.replace('.mid', '_onset_pred.npy'))[:, note - MIN_MIDI]
        data = torch.load(reference)
        unaligned_onsets = (data['unaligned_label'] == 3).float().numpy()
        reference_pred = unaligned_onsets[:, note - MIN_MIDI]
        if reference_pred.sum() == 0 or onset_pred.sum() == 0:
            continue
        dist_list.append(abs(reference_pred.sum() - onset_pred.sum()))
        
    if len(dist_list) == 0:
        print(f"there were no evaluations for note {note} and threshold {threshold}")
        return None
    return np.mean(dist_list)

def get_p_r_f(synthsized_midis, reference_midis, threshold, tolerance=0.05, filter_note=None):
    
    precision_list = []
    recall_list = []
    f1_list = []
    
    for transcribed, reference in zip(synthsized_midis, reference_midis):
        if reference.endswith('.tsv'):
            reference_events = np.loadtxt(reference, delimiter='\t', skiprows=1)
        else:
            reference_events = parse_midi_multi(reference)
        onset_pred = np.load(transcribed.replace('.mid', '_onset_pred.npy'))
        frame_pred = np.load(transcribed.replace('.mid', '_frame_pred.npy'))
        p_est, i_est, v_est, ins_est = extract_notes_np_rescaled(onset_pred[:, -88:], frame_pred[:, -88:],
                64. * onset_pred[:, -88:], threshold, 0.5)
        transcribed_events = np.vstack([i_est.transpose(), p_est, v_est, ins_est]).transpose()
        # transcribed_events = extract_notes_np(onset_pred[:, -88:], frame_pred[:, -88:],
        #         64. * onset_pred[:, -88:], threshold, 0.5)
        
        if filter_note is not None:
            transcribed_events = transcribed_events[transcribed_events[:, 2] == filter_note]
            reference_events = reference_events[reference_events[:, 2] == filter_note]
            # the note did not appear
            if reference_events.shape[0] == 0 or transcribed_events.shape[0] == 0:
                continue
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
        # print("negative interval self check", (i_ref[:,1] <= i_ref[:,0]).any())
        # print("negative interval self check", (i_est[:,1] <= i_est[:,0]).any())
        try:
            p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None, onset_tolerance=tolerance)
        except:
            print("error in file", reference)
            continue

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f)
    if len(precision_list) == 0:
        print(f"there were no evaluations for note {filter_note} and threshold {threshold}")
        return None, None, None
    return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

def find_threshold_bag_of_note(transcribed_path, reference_path, note):
    th = 0.5 # start with the trivial threshold, and try improving it by looking left and right
    search_width = 0.25 # we decrease search width by factor of 2 each iteration
    # sanity_chelk_f1 = get_p_r_f(transcribed_path, reference_path, 0.5)
    # print("sanity check p/r/f", sanity_chelk_f1)
    best_dist = get_bag_of_notes_for_note(transcribed_path, reference_path, th, note)
    if best_dist == None:
        print(f"no evaluation for note {note} the threshold is set to default")
        return th
    print("best dist type", type(best_dist))
    # 10 iterations should be enough (2 ** (-10) < 0.001)
    for i in range(3):
        lower_th = th - search_width
        higher_th = th + search_width
        left = get_bag_of_notes_for_note(transcribed_path, reference_path, lower_th, note)
        right = get_bag_of_notes_for_note(transcribed_path, reference_path, higher_th, note)
        print(f"iteration {i}, lower threshold {lower_th} dist {left}, higher threshold {higher_th} dist {right}, best dist {best_dist}")
        if right is not None and right < best_dist: # increasing the threshold brings us closer to the target
            best_f1 = right
            th = th + search_width
        elif left is not None and left < best_dist: # decreasing the thresholds brings us closer to the target, or doesn't exceed the defined limit
            best_f1 = left
            th = th - search_width

        search_width *= 0.5 # didn't reach desired range yet, reduce search width for next iteration
    print('pitch number {} chosen threshold {}'.format(note, th, best_dist))
    return th

    
def find_threshold(transcribed_path, reference_path, note):
    th = 0.5 # start with the trivial threshold, and try improving it by looking left and right
    search_width = 0.25 # we decrease search width by factor of 2 each iteration
    # sanity_chelk_f1 = get_p_r_f(transcribed_path, reference_path, 0.5)
    # print("sanity check p/r/f", sanity_chelk_f1)
    print("start threshold")
    best_f1 = get_p_r_f(transcribed_path, reference_path, th, filter_note=note)[-1]
    if best_f1 == None:
        print(f"no evaluation for note {note} the threshold is set to default")
        return th
    print("best f1 type", type(best_f1))
    # 10 iterations should be enough (2 ** (-10) < 0.001)
    for i in range(3):
        lower_th = th - search_width
        higher_th = th + search_width
        left = get_p_r_f(transcribed_path, reference_path, threshold=lower_th, filter_note=note)[-1]
        right = get_p_r_f(transcribed_path, reference_path, threshold=higher_th, filter_note=note)[-1]
        print(f"iteration {i}, lower threshold {lower_th} f1 {left}, higher threshold {higher_th} f1 {right}, best f1 {best_f1}")
        if right is not None and right > best_f1: # increasing the threshold brings us closer to the target
            best_f1 = right
            th = th + search_width
        elif left is not None and left > best_f1: # decreasing the thresholds brings us closer to the target, or doesn't exceed the defined limit
            best_f1 = left
            th = th - search_width

        search_width *= 0.5 # didn't reach desired range yet, reduce search width for next iteration
    print('pitch number {} chosen threshold {}'.format(note, th, best_f1))
    return th



def fine_tune_notes_thresholds(transcribed_path, ref_path, log_dir=None):
    print("starting to fine tune thresholds")
    thresholds = []
    threshold_dict = {}
    # for note in tqdm(range(MIN_MIDI, MAX_MIDI + 1)):
    
    for note in tqdm(range(MIN_MIDI + 77, MAX_MIDI + 1)):
        threshold = find_threshold(transcribed_path, ref_path, note)
        thresholds.append(threshold)
        threshold_dict[note] = threshold
        if log_dir is not None:
            with open(os.path.join(log_dir, 'thresholds.json'), 'w') as json_file:
                json.dump(threshold_dict, json_file)
    print(thresholds)


def fine_tune_thresholds_with_files(yaml_config, log_dir=None):
    config = yaml_config['threshold']
    tsv_path = config['tsv_path']
    transcribed_path = config['transcribed_path']
    tsv_list = sorted(os.listdir(tsv_path))
    transcribed_list = sorted(os.listdir(transcribed_path))
    transcribed_list = [m for m in transcribed_list if m.endswith('.mid')]
    
    tsv_list = [os.path.join(tsv_path, t) for t in tsv_list]
    transcribed_list = [os.path.join(transcribed_path, t) for t in transcribed_list]
    fine_tune_notes_thresholds(transcribed_list, tsv_list, log_dir=log_dir)
    
    

def evaluate_file(synthsized_midis, reference_midis, instruments=None, conversion_map=None, tolerance=0.05):
    metrics = evaluate(synthsized_midis, reference_midis, instruments=instruments, conversion_map=conversion_map, tolerance=tolerance)
    # metrics = evaluate(tqdm(dataset), (model, parallel_model), onset_threshold, frame_threshold, save_path)
    

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')


if __name__ == '__main__':
    # transcribed_path = "evaluation/midis_no_pitch_shift_transcribed"
    # reference_path = "evaluation/new_inference_dir/results"
    # transcribed_midis = os.listdir(transcribed_path)
    # reference_midis = os.listdir(reference_path)
    # transcribed_midis = [transcribed_path + '/' + s for s in transcribed_midis]
    # reference_midis = [reference_path + '/' + s for s in reference_midis]
    # instruments = [0, 6, 10, 19, 24, 40, 41, 42, 43, 44, 45, 46, 47, 48, 52, 58, 60, 61, 64, 68, 70, 71, 73]
    instruments = [0, 68, 70, 71, 40, 73, 41, 42, 45, 6, 60]
    instruments = [0]
    transcribed_path = "evaluation/midis_no_pitch_shift_transcribed"
    transcribed_midis = os.listdir(transcribed_path)
    reference_path = "evaluation/new_inference_dir/results"
    reference_midis = os.listdir(reference_path)
    evaluate_file(transcribed_midis, reference_midis, instruments=instruments)

