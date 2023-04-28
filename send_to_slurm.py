import os
import yaml


if __name__ == "__main__":
    with open("slurm_config.yaml", 'r') as fp:
        config = yaml.load(fp)
    sbatch_command = 'sbatch'
    for param in config:
        sbatch_command += f' --{param}={config[param]}'
    sbatch_command += ' /specific/a/home/cc/students/cs/jonathany/research/pop_alignment/run_train2'
    os.system(sbatch_command)
    print("submitted task")
    # print(sbatch_command)