#!/bin/bash --login

#SBATCH --exclude=lac-142
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=32G

source ~/.bashrc
source ~/anaconda3/bin/activate torch1.13.1
cd /mnt/home/jinwei2/juntong/graphslim/graphslim


srun -n 1 python train_coarsen.py --dataset ${1} --method ${2} --reduction_rate ${3} --attack ${4} --ptb_r ${5}

#scontrol show job $SLURM_J0B_ID