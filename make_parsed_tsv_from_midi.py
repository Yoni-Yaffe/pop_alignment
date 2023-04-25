'''
Convert midi to a note-list representation in a tsv file. Each line in the tsv file will contain
information of a single note: onset time, offset time, note number, velocity, and instrument.
'''


import numpy as np
import os
import warnings
from onsets_and_frames.midi_utils import parse_midi_multi
warnings.filterwarnings("ignore")


def midi2tsv_process(midi_path, target_path, shift=0, force_instrument=None):
    midi = parse_midi_multi(midi_path, force_instrument=force_instrument)
    print(target_path)
    if shift != 0:
        midi[:, 2] += shift
    np.savetxt(target_path, midi,
               fmt='%1.6f', delimiter='\t', header='onset,offset,note,velocity,instrument')



def create_tsv_for_single_group(midi_src_pth, target):
    FORCE_INSTRUMENT = None
    piece = midi_src_pth.split(os.sep)[-1]
    os.makedirs(target + os.sep + piece, exist_ok=True)
    for f in os.listdir(midi_src_pth):
        if f.endswith('.mid') or f.endswith('.MID'):
            print(f)
            midi2tsv_process(midi_src_pth + os.sep + f,
                            target + os.sep + piece + os.sep + f.replace('.mid', '.tsv').replace('.MID', '.tsv'),
                            force_instrument=FORCE_INSTRUMENT)


def create_tsv_for_multiple_groups(midi_src_pth_list, target):
    for midi_src in midi_src_pth_list:
        print(f"Creating tsv for group {midi_src}")
        create_tsv_for_single_group(midi_src, target)


if __name__ == "__main__":
        # midi_src_pth = '/path/to/midi/perfromance'
    midi_src_pth = '/vol/scratch/jonathany/datasets/musicnet_groups_of_20/midis'
    midi_src_path_list = [os.path.join(midi_src_pth, f"group{i}") for i in range(1, 10)]
    # target = '/disk4/ben/UnalignedSupervision/NoteEM_tsv'
    target = 'NoteEM_tsv'
    create_tsv_for_multiple_groups(midi_src_path_list, target)

