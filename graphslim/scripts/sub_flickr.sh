#!/bin/bash
#SBATCH --account=a100v100
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=a100-8-gm320-c96-m1152
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate pygdgl
cd ~/GraphSlim/graphslim

echo '====start running===='
python train_all.py -M gcdm --dis_metric l1 --save_path /scratch/sgong36/checkpoints --load_path /scratch/sgong36/data -D flickr -R $1
