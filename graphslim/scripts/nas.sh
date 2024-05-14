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

srun -n 1 python run_nas.py --dataset ${1} --method ${2} --reduction_rate ${3}

scontrol show job $SLURM_J0B_ID