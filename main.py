from generate_labels import generate_labels_wrapper
from train import train_wrapper
import yaml
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', default=None)
    parser.add_argument('--yaml_path', default=None)
    args = parser.parse_args()
    
    logdir = args.logdir
    yaml_path = args.yaml_path
    if logdir is None:
        raise RuntimeError("no logdir provided")
    if yaml_path is None:
        yaml_path = os.path.join(logdir, 'run_config.yaml')
        if not os.path.exists(yaml_path):
            raise RuntimeError("no yaml file provided")
    print("yaml path:", yaml_path)
    with open(yaml_path, 'r') as fp:
        yaml_config = yaml.load(fp, Loader=yaml.FullLoader)
    
    if 'run_type' not in yaml_config:
        raise RuntimeError("No run type in yaml file")
    
    if yaml_config['run_type'] == 'train':
        train_wrapper(yaml_config, logdir)
    elif yaml_config['run_type'] == 'inference':
        generate_labels_wrapper(yaml_config)

if __name__ == "__main__":
    main()