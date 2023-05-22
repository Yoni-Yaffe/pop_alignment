import os

def group_mid_files_from_audio(audio_path, mid_path, res_path):
    all_files = os.listdir(audio_path)
    all_groups = list(set([s.split('#')[0] for s in all_files]))
    reverse_dict = {}
    for g in all_groups:
        for file in os.listdir(os.path.join(audio_path, f"{g}#0")):
            reverse_dict[file.split('#')[0]] = g
    print(all_groups)
    
if __name__ == "__main__":
    audio_path = '/vol/scratch/jonathany/datasets/Museopen16'
    mid_path = '/vol/scratch/jonathany/datasets/AllTranscriptions'
    res_path = None
    group_mid_files_from_audio(audio_path, mid_path, res_path)