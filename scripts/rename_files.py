import os

def rename_audio_files(audio_path, group_name):
    for i in range(-5, 6):
        dir_name = f"{audio_path}/{group_name}#{i}"
        for file in os.listdir(os.path.join(audio_path, dir_name)):
            s = ""
            if file.endswith('.flac'):
                src_path = os.path.join(audio_path, dir_name, file)
                new_fname = file.split('#')[0] + '#' + file.split('#')[-1] 
                dst_path = os.path.join(audio_path, dir_name, new_fname)
                os.rename(src_path, dst_path)
            

def rename_list_of_groups(audio_path, group_list):
    for group in group_list:
        rename_audio_files(audio_path=audio_path, group_name=group)
        
        
if __name__ == '__main__':
    audio_path = '/vol/scratch/jonathany/datasets/Museopen16/'
    group = "Beethoven Violin Sonatas"
    rename_audio_files(audio_path=audio_path, group_name=group)
    