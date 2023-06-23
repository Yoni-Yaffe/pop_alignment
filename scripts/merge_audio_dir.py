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


def add_virtual_paths(src_dir_path, merged_dir_path):
    os.makedirs(merged_dir_path, exist_ok=True)
    for f in os.listdir(src_dir_path):
        curr_dir_path = os.path.join(src_dir_path, f)
        if os.path.isdir(curr_dir_path) and f.count('#') == 1:
            src = curr_dir_path
            dst = os.path.join(merged_dir_path, f)
            os.symlink(src=src, dst=dst)
    


if __name__ == "__main__":
    # audio_path = '/vol/scratch/jonathany/datasets/Museopen16'
    # # mid_path = '/vol/scratch/jonathany/datasets/AllTranscriptions'
    # res_path = '/vol/scratch/jonathany/datasets/full_museopen/noteEM_audio'
    # merge_audio_dir(audio_path, res_path, 'full_museopen')
    src_dir = '/vol/scratch/jonathany/datasets/Museopen16'
    dst = '/vol/scratch/jonathany/datasets/all_museopen_groups_group/noteEM_audio'
    add_virtual_paths(src_dir_path=src_dir, merged_dir_path=dst)
            
    