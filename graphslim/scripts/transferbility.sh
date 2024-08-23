#!/bin/bash --login

#SBATCH --exclude=lac-142
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=32G
#SBATCH --job-name main-${1}-${2}-${3}

source ~/.bashrc
source ~/anaconda3/bin/activate pygdgl

python run_cross_arch.py -D ${1} -M ${2}

scontrol show job $SLURM_J0B_ID
