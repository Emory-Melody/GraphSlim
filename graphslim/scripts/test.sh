#!/bin/bash --login

#SBATCH --exclude=lac-142
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=32G

source ~/.bashrc
source ~/anaconda3/bin/activate pygdgl

srun -n 1 python test.py

scontrol show job $SLURM_J0B_ID