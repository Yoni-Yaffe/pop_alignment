#!/bin/bash

#SBATCH --job-name=alignment_model
#SBATCH --output=/specific/a/home/cc/students/cs/jonathany/research/pop_alignment/slurmlog.out # redirect stdout
#SBATCH --error=/specific/a/home/cc/students/cs/jonathany/research/pop_alignment/slurmlog.err # redirect stderr
#SBATCH --partition=studentkillable # (see resources section)
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time�s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=8 # CPU cores per process
#SBATCH --gpus=1 # GPUs in tota
#SBATCH --constraint=titan_xp

./run_train
