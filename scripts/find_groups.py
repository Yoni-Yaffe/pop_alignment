import os 
import shutil


def check_if_valid_groups(group_name, file_list):
    for i in range(-5, 5):
        if f"{group_name}#{i}" not in file_list:
            return False
    return True

def find_all_groups(path):
    all_files = os.listdir(path)
    all_groups = list(set([s.split('#')[0] for s in all_files]))
    valid_groups = [g for g in all_groups if check_if_valid_groups(g, all_files)]
    invalid_groups = [g for g in all_groups if g not in valid_groups]
    print(f"num groups: {len(all_groups)}")
    print(f"num valid: {len(valid_groups)}")
    print(f"invalid groups: {invalid_groups}")
    return valid_groups
                
def get_all_mid_files(path):
    l = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith(".mid"):
                l.append(name)
    print(len(l))
    # print(sorted(l)[:10])
    

def find_missmatches(audio_path, mid_path):
    all_files = os.listdir(audio_path)
    all_groups = list(set([s.split('#')[0] for s in all_files]))
    reverse_dict = {}
    for g in all_groups:
        for file in os.listdir(os.path.join(audio_path, f"{g}#0")):
            reverse_dict[file.split('#')[0]] = g
    print("reverse len", len(reverse_dict))
    mid_files_names = [f.split('.mid')[0] for f in os.listdir(mid_path)]
    flac_files = []
    for root, dirs, files in os.walk(audio_path, topdown=False):
        for name in files:
            if name.endswith("#0.flac"):
                flac_files.append(name.split('#0.flac')[0])
    errors = []
    errors_group_set = set()
    # for f in mid_files_names:
    #     if f"{f.split('#')[0]}" not in flac_files:
    #         errors.append(f)
    #         print(reverse_dict[f.split('#')[0]])
    for f in flac_files:
        exists = 0
        mid_fixed = [m.split('#')[0] for m in mid_files_names]
        for mid in mid_fixed:
            if f == mid:
                # if f != mid.split('#')[0]:
                #     print("mismatch: ")
                #     print(f, "$$$" ,mid.split('#')[0])
                exists = 1
                break
        if exists == 0:
            errors.append(f)
            errors_group_set.add(reverse_dict[f.split('#')[0]])
            print("reverse", reverse_dict[f.split('#')[0]], "$$", f.split('#')[0])
        

    print("groups", errors_group_set, len(errors_group_set))
    print(f"len midis = {len(mid_files_names)} len audio = {len(flac_files)}")
    print("errors", len(errors))
    # print("flac files")
    # print(mid_files_names[:10])
    # print("mid files")
    # print(flac_files[:10])
    print("II. Courante" in flac_files)

def copy_only_matching_midis(audio_path, mid_path, res_path):
    mid_files_names = [f for f in os.listdir(mid_path) if f.endswith('.mid')]
    flac_files = set()
    for root, dirs, files in os.walk(audio_path, topdown=False):
        for name in files:
            flac_files.add(name.split('#')[0])
    print(f"found {len(flac_files)} flac files")
    counter = 0 
    for mid in mid_files_names:
        name = mid.split('#')[0].split('.mid')[0]
        if name in flac_files:
            src_path = os.path.join(mid_path, mid)
            dst_path = os.path.join(res_path, mid)
            shutil.copy(src_path, dst_path)
            counter += 1
    print(f"copied {counter} midis")
        
    
    
if __name__ == "__main__":
    # find_all_groups('/vol/scratch/jonathany/ben_dataset/Museopen16')
    # get_all_mid_files('/vol/scratch/jonathany/ben_dataset/AllTranscriptions')
    audio_path = '/vol/scratch/jonathany/datasets/Museopen16'
    mid_path = '/vol/scratch/jonathany/datasets/AllTranscriptions'
    res_path = '/vol/scratch/jonathany/datasets/Museopen_merged/mid_files/full_museopen'
    # find_missmatches(audio_path=audio_path, mid_path=mid_path)
    copy_only_matching_midis(audio_path, mid_path, res_path)
    