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
    best_bon_dir_path = "/a/home/cc/students/cs/jonathany/tmp_runs/new_runs/no_solo_group3_transcriber-230420-212931/alignments/BEST_BON"
    create_last_best_bon(best_bon_dir_path)





