import os
import shutil

def make_runs_metadata_dir(run_dir_path, res_dir_path):
    os.makedirs(res_dir_path, exist_ok=True)
    runs = os.listdir(run_dir_path)
    for run in runs:
        new_run_dir = os.path.join(res_dir_path, run)
        os.makedirs(new_run_dir, exist_ok=True)
        desired_files = ['run_config.yaml', 'score_log.txt']
        for file in desired_files:
            src_file_path = os.path.join(run_dir_path, run, file)
            dst_file_path = os.path.join(res_dir_path, run, file)
            if os.path.exists(file):
                shutil.copy(src=src_file_path, dst=dst_file_path)
    print(f"created metadata runs dir at {res_dir_path}")
    
if __name__ == '__main__':
    run_dir_path = '/vol/scratch/jonathany/runs'
    new_run_dir_path = '/specific/a/home/cc/students/cs/jonathany/runs_metadata'
    make_runs_metadata_dir(run_dir_path, new_run_dir_path)