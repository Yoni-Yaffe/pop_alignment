import os
import shutil

def create_last_best_bon(best_bon_dir_path):
    file_list = os.listdir(best_bon_dir_path)
    new_bon_dir = os.path.join(best_bon_dir_path, "one_per_song")
    os.makedirs(new_bon_dir, exist_ok=True)
    # songs = list(set([s.split('#')[0] for s in file_list]))
    song_dict = {}
    for file_name in file_list:
        song_id = file_name.split('#')[0]
        if 'alignment' in file_name:
            if song_id not in song_dict or file_name > song_dict[song_id]:
                song_dict[song_id] = file_name

    for alignment in song_dict.values():
        source = os.path.join(best_bon_dir_path, alignment)
        destination = os.path.join(new_bon_dir, alignment)
        shutil.copy(source, destination)
    print(f"Created a new bon directory in {new_bon_dir}")


if __name__ == "__main__":
    best_bon_dir_path = "/vol/scratch/jonathany/runs/group1_transcriber-230425-152115/alignments/BEST_BON"
    groups = [i for i in os.listdir("/vol/scratch/jonathany/runs") if 'group' in i]
    for g in groups:
        best_bon_dir_path = f"/vol/scratch/jonathany/runs/{g}/alignments/BEST_BON/one_per_song"
        # src = best_bon_dir_path
        # dst = f"/vol/scratch/jonathany/datasets/full_musicnet_groups_of_20/{g.split('_')[0]}"
        # shutil.copytree(src, dst)
        # print(f"Copied group {g}")
        create_last_best_bon(best_bon_dir_path)





