import os
import shutil
from tqdm import tqdm
import time


def merge_audio_dir(src_dir, res_dir, group_name):
    for i in range(-5, 6):
        os.makedirs(os.path.join(res_dir, f"{group_name}#{i}"), exist_ok=True)
    counter = 0
    for root, dirs, files in tqdm(os.walk(src_dir, topdown=False)):
        for name in files:
            if not name.endswith('.flac'):
                continue
            shft = int(name.split('#')[-1].split('.flac')[0])
            src_path = os.path.join(root, name)
            dst_path = os.path.join(res_dir, f"{group_name}#{shft}", name)
            # shutil.copy(src=src_path, dst=dst_path)
            # createing a sym link is the more efficient method
            os.symlink(src=src_path, dst=dst_path)
            counter += 1
        
    print(f"Finished merging, overall found {counter} files.")

def merge_midi(src, dst):
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in tqdm(os.walk(src, topdown=False)):
        for name in files:
            src_path = os.path.join(root, name)
            dst_path = os.path.join(dst, name)
            os.symlink(src=src_path, dst=dst_path)

def add_virtual_paths(src_dir_path, merged_dir_path):
    os.makedirs(merged_dir_path, exist_ok=True)
    for f in os.listdir(src_dir_path):
        curr_dir_path = os.path.join(src_dir_path, f)
        if os.path.isdir(curr_dir_path) and f.count('#') == 1:
            src = curr_dir_path
            dst = os.path.join(merged_dir_path, f)
            os.symlink(src=src, dst=dst)
    
def copy_all_files_virtual(src_dir_path, dst_dir_path):
    os.makedirs(dst_dir_path, exist_ok=True)
    counter = 0
    for f in os.listdir(src_dir_path):
        curr_dir_path = os.path.join(src_dir_path, f)
        src = curr_dir_path
        dst = os.path.join(dst_dir_path, f)
        os.symlink(src=src, dst=dst)
        counter += 1
    print(f"copied {counter} files")
    
def merge_museopen_and_musicnet(musicnet_dir, museopen_dir):
    pass

if __name__ == "__main__":
    ################ merge audio    
    # # audio_path = '/vol/scratch/jonathany/datasets/Museopen16'
    # # # mid_path = '/vol/scratch/jonathany/datasets/AllTranscriptions'
    # # res_path = '/vol/scratch/jonathany/datasets/full_museopen/noteEM_audio'
    # # merge_audio_dir(audio_path, res_path, 'full_museopen')
    
    # audio_path = '/vol/scratch/jonathany/datasets/FULL_POP_AUDIO_SHIFT'
    # # # mid_path = '/vol/scratch/jonathany/datasets/AllTranscriptions'
    # res_path = '/vol/scratch/jonathany/datasets/full_pop_merged/noteEM_audio'
    # merge_audio_dir(audio_path, res_path, 'full_pop_merged')
        
    # audio_path = '/vol/scratch/jonathany/datasets/POP_AUDIO_SHIFT'
    # # # mid_path = '/vol/scratch/jonathany/datasets/AllTranscriptions'
    # res_path = '/vol/scratch/jonathany/datasets/full_pop_merged_first/noteEM_audio'
    # merge_audio_dir(audio_path, res_path, 'full_pop_merged_first')
    

    
    # midi_path = '/vol/scratch/jonathany/datasets/FULL_POP_MIDI'
    # # # mid_path = '/vol/scratch/jonathany/datasets/AllTranscriptions'
    # res_path = '/vol/scratch/jonathany/datasets/full_pop_merged_midi/full_pop_merged_first'
    # merge_midi(midi_path, res_path)
    
    
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_musicnet_with_piano_random_shift/noteEM_audio'
    # dst = '/vol/scratch/jonathany/datasets/pop_and_museopen_2nd/noteEM_audio'
    # add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_musicnet_with_piano_random_shift/noteEM_audio'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop/noteEM_audio'
    # add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_musicnet_with_piano_random_shift/noteEM_audio'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop/noteEM_audio'
    # add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_musicnet_with_piano_random_shift/noteEM_audio'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop/noteEM_audio'
    # add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_musicnet_with_piano_random_shift/noteEM_audio'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop_second_iter/noteEM_audio'
    # add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
    
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_pop_merged/noteEM_audio'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop/noteEM_audio'
    # add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_pop_merged_first/noteEM_audio'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop/noteEM_audio'
    # add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
    
    src_dir = '/vol/scratch/jonathany/datasets/full_musicnet_with_piano_random_shift/noteEM_audio'
    dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop_second_iter/noteEM_audio'
    add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
    
    
    src_dir = '/vol/scratch/jonathany/datasets/full_pop_merged/noteEM_audio'
    dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop_second_iter/noteEM_audio'
    add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
    
    src_dir = '/vol/scratch/jonathany/datasets/full_pop_merged_first/noteEM_audio'
    dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop_second_iter/noteEM_audio'
    add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_musicnet_with_piano_random_shift/noteEM_audio'
    # dst = '/vol/scratch/jonathany/datasets/full_pop_merged/noteEM_audio'
    # add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_musicnet_with_piano_random_shift/noteEM_audio'
    # dst = '/vol/scratch/jonathany/datasets/full_pop_merged_first/noteEM_audio'
    # add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
    
    # merge_audio_dir(src_dir=src_dir, res_dir='/vol/scratch/jonathany/datasets/museopen_merged', group_name='museopen')
            
    # src_dir = '/vol/scratch/jonathany/datasets/full_musicnet_with_piano_random_shift/NoteEm_labels'
    # dst_path = '/vol/scratch/jonathany/datasets/musicnet_and_museopen/NoteEm_labels'
    # copy_all_files_virtual(src_dir, dst_path)
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_museopen/NoteEm_labels_pitch'
    # dst = '/vol/scratch/jonathany/datasets/pop_and_museopen/NoteEm_labels_pitch'
    # copy_all_files_virtual(src_dir_path=src_dir, dst_dir_path=dst)
    
    ### add the labels to the merged dir

    # src_dir = '/vol/scratch/jonathany/runs/231217-143701_pop_merged_first_labels_second_iteration/NoteEm_labels'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop_second_iter/NoteEm_labels_pitch'
    # copy_all_files_virtual(src_dir_path=src_dir, dst_dir_path=dst)
    
        
    # src_dir = '/vol/scratch/jonathany/datasets/full_musicnet_with_piano_random_shift/NoteEm_labels_pitch'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop/NoteEm_labels_pitch'
    # copy_all_files_virtual(src_dir_path=src_dir, dst_dir_path=dst)
    
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_pop_merged/NoteEm_labels_pitch'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop/NoteEm_labels_pitch'
    # copy_all_files_virtual(src_dir_path=src_dir, dst_dir_path=dst)
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_pop_merged_first/NoteEm_labels_pitch'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop/NoteEm_labels_pitch'
    # copy_all_files_virtual(src_dir_path=src_dir, dst_dir_path=dst)
    
        
    # src_dir = '/vol/scratch/jonathany/datasets/full_musicnet_with_piano_random_shift/NoteEm_labels_pitch'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop_second_iter/NoteEm_labels_pitch'
    # copy_all_files_virtual(src_dir_path=src_dir, dst_dir_path=dst)
    
    
    # src_dir = '/vol/scratch/jonathany/runs/240103-135017_pop_merged_first_get_relabel_second_iter_pop/NoteEm_labels'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop_second_iter/NoteEm_labels_pitch'
    # copy_all_files_virtual(src_dir_path=src_dir, dst_dir_path=dst)
    
    # src_dir = '/vol/scratch/jonathany/runs/240103-135045_pop_merged_get_relabel_second_iter_pop/NoteEm_labels/'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop_second_iter/NoteEm_labels_pitch'
    # copy_all_files_virtual(src_dir_path=src_dir, dst_dir_path=dst)
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_pop_merged/NoteEm_labels_inst'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop/NoteEm_labels_inst'
    # copy_all_files_virtual(src_dir_path=src_dir, dst_dir_path=dst)
    
    # src_dir = '/vol/scratch/jonathany/datasets/full_pop_merged_first/NoteEm_labels_inst'
    # dst = '/vol/scratch/jonathany/datasets/musicnet_and_pop/NoteEm_labels_inst'
    # copy_all_files_virtual(src_dir_path=src_dir, dst_dir_path=dst)
    
    
    # src_dir = "/vol/scratch/jonathany/datasets/full_pop_merged/NoteEm_labels_inst"
    # dst = '/vol/scratch/jonathany/datasets/all_pop_songs/NoteEm_labels_inst'
    # copy_all_files_virtual(src_dir_path=src_dir, dst_dir_path=dst)
    # lst1 = os.listdir("/vol/scratch/jonathany/datasets/full_pop_merged/NoteEm_labels_inst")
    # lst2 = os.listdir("/vol/scratch/jonathany/datasets/full_pop_merged_first/NoteEm_labels_inst")
    # for i in lst1:
    #     if i in lst2:
    #         print(i)
    # print("finished!")