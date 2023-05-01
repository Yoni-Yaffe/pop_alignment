import os
import math
import shutil
import json

def list_all_songs_in_dir(path: str) -> list:
    song_set = set()
    for root, dirs, files in os.walk(path):
        for file in files:
            song_set.add(file.split('#')[0])
    return sorted(list(song_set))


def split_song_list_to_groups(songs_list: list, num_per_group: int):
    num_groups = math.ceil(len(songs_list) / num_per_group)
    # groups = [song_list[i * num_per_group: (i + 1) * num_per_group] for i in range(num_groups)]
    song_to_group_dict = {songs_list[i]: i // num_per_group + 1 for i in range(len(songs_list))}
    # print(song_to_group_dict)
    return song_to_group_dict

def create_new_noteEM_directory(old_noteEM_path: str, new_noteEM_path: str, num_per_group: int):
    song_list = list_all_songs_in_dir(old_noteEM_path)
    song_to_group_dict = split_song_list_to_groups(song_list, num_per_group)

    print(f"splitted to {len(set(song_to_group_dict.values()))} groups")
    with open(os.path.join(new_noteEM_path, "song_to_group_dict.json"), 'w') as fp:
        json.dump(song_to_group_dict, fp, indent=4)
    
    for i in range(-5, 6):
        for g in set(song_to_group_dict.values()):
            os.makedirs(os.path.join(new_noteEM_path, f"group{g}#{i}"), exist_ok=True)
    
    for root, dirs, files in os.walk(old_noteEM_path):
        for file in files:
            song_id, shift = file.split('#')
            shift = shift.split('.')[0]
            group = song_to_group_dict[song_id]
            destination = os.path.join(new_noteEM_path, f"group{group}#{shift}", file)
            source = os.path.join(root, file)
            shutil.copy(source, destination)
    print("created successfuly new noteEM dir")


def create_new_midi_dirs_per_group(old_midis_dir_path: str, new_midis_dir_path: str, groups_json_file_path: str):
    with open(groups_json_file_path, 'r') as fp:
        song_to_group_dict = json.load(fp)
    groups = list(set(song_to_group_dict.values()))

    for g in groups:
        os.makedirs(os.path.join(new_midis_dir_path, f"group{g}"), exist_ok=True)

    for root, dirs, files in os.walk(old_midis_dir_path):
        for file in files:
            if not file.endswith('.mid'):
                continue
            song_id = file.split('#')[0]
            song_id = song_id.split('.mid')[0]
            if song_id not in song_to_group_dict:
                continue
            group = song_to_group_dict[song_id]
            destination = os.path.join(new_midis_dir_path, f"group{group}", file)
            source = os.path.join(root, file)
            shutil.copy(source, destination)
    print("created new midis dir")
      
def create_sym_links_for_audio():
    src_dir = "/vol/scratch/jonathany/datasets/musicnet_groups_of_20/noteEM_audio"
    dst_dir = "/vol/scratch/jonathany/datasets/musicnet_em_grouped/noteEM_audio"
    for d in os.listdir(src_dir):
        os.symlink(os.path.join(src_dir, d), os.path.join(dst_dir, 'em_' + d))

if __name__ == "__main__":
    # song_list = list_all_songs_in_dir("/vol/scratch/jonathany/datasets/full_musicnet/noteEM_audio")
    # split_song_list_to_groups(song_list, 20)
    # create_new_noteEM_directory("/vol/scratch/jonathany/datasets/full_musicnet/noteEM_audio", "/vol/scratch/jonathany/datasets/musicnet_groups_of_20/noteEM_audio", 20)
    old_path = "/vol/scratch/jonathany/datasets/musicnet_em"
    new_path = "/vol/scratch/jonathany/datasets/musicnet_em_grouped"
    groups_json_path = "/vol/scratch/jonathany/datasets/musicnet_groups_of_20/noteEM_audio/song_to_group_dict.json"
    # create_new_midi_dirs_per_group(old_path, new_path, groups_json_path)