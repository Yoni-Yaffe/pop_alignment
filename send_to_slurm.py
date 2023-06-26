import os
import yaml
from datetime import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_config', default='config.yaml')
    args = parser.parse_args()
    with open(args.yaml_config, 'r') as fp:
        config = yaml.safe_load(fp)
    slurm_config = config['slurm_params']
    sbatch_command = 'sbatch'
    logdir = f"/vol/scratch/jonathany/runs/{datetime.now().strftime('%y%m%d-%H%M%S')}_{config['run_name']}" # ckpts and midi will be saved here
    config['logdir'] = logdir
    slurm_config['output'] = os.path.join(logdir, slurm_config['output'])
    slurm_config['error'] = os.path.join(logdir, slurm_config['error'])
    os.makedirs(logdir, exist_ok=True)
    new_yaml_path = os.path.join(logdir, 'run_config.yaml')
    with open(new_yaml_path, 'w') as fp:
        yaml.dump(config, fp)
    
    for param in slurm_config:
        sbatch_command += f' --{param}={slurm_config[param]}'
    sbatch_command += f" {config['command']} --logdir {logdir} --yaml_config {new_yaml_path}"
    os.system(sbatch_command)
    