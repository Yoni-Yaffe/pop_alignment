from generate_labels import generate_labels_wrapper
from train import train_wrapper
import yaml
import os
import argparse
import sys
from evaluate_multi_inst import fine_tune_thresholds_with_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--yaml_config', default=None)
    args = parser.parse_args()
    
    logdir = args.logdir
    yaml_path = args.yaml_config
    if logdir is None:
        raise RuntimeError("no logdir provided")
    if yaml_path is None:
        yaml_path = os.path.join(logdir, 'run_config.yaml')
        if not os.path.exists(yaml_path):
            raise RuntimeError("no yaml file provided")
    print("yaml path:", yaml_path)
    with open(yaml_path, 'r') as fp:
        yaml_config = yaml.load(fp, Loader=yaml.FullLoader)
    if 'local' in yaml_config and yaml_config['local']:
        stdout_file = open(os.path.join(logdir, 'slurmlog.out'), 'w')
        stderr_file = open(os.path.join(logdir, 'slurmlog.err'), 'w')
        sys.stdout = stdout_file
        sys.stderr = stderr_file
    
    if 'run_type' not in yaml_config:
        raise RuntimeError("No run type in yaml file")
    
    if yaml_config['run_type'] == 'train':
        train_wrapper(yaml_config, logdir)
    elif yaml_config['run_type'] == 'inference':
        generate_labels_wrapper(yaml_config)
    elif yaml_config['run_type'] == 'threshold':
        fine_tune_thresholds_with_files(yaml_config, log_dir=logdir)

if __name__ == "__main__":
    main()