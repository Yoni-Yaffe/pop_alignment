import os
import json

def get_reverse_dict(audio_dir_path):
    res_dict = {}
    for group in os.listdir(audio_dir_path):
        if not group.endswith('#0'):
            continue
        group_name = group.split('#')[0]
        for piece in os.listdir(os.path.join(audio_dir_path, group)):
            res_dict[piece.split('#')[0]] = group_name
    return res_dict

if __name__ == "__main__":
    d = get_reverse_dict('/vol/scratch/jonathany/datasets/Museopen16')
    # print(d)
    with open('test_json.json', 'w') as fp:
        json.dump(d, fp, indent=4)
    print("saved to test_json.json")