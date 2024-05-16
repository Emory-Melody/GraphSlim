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

if [[ "${2}" == "gcondx" || "${2}" == "gcond" || "${2}" == "doscond" || "${2}" == "sgdd" ]]; then
    srun -n 1 python train_gcond.py --dataset ${1} --method ${2} --reduction_rate ${3}
elif [[ "${2}" == "random" || "${2}" == "herding" || "${2}" == "kcenter" || "${2}" == "cent_p" || "${2}" == "cent_d" ]]; then
    srun -n 1 python train_coreset.py --dataset ${1} --method ${2} --reduction_rate ${3}
else
    srun -n 1 python train_coarsen.py --dataset ${1} --method ${2} --reduction_rate ${3}
fi

#scontrol show job $SLURM_J0B_ID