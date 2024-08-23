#!/bin/bash --login

##############SBATCH Lines for Resource Request ##########

#SBATCH --time=10:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                 # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=8           # number of CPUs (or cores) per task (same as -c)
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name lmlp      # you can give your job a name for easier identification (same as -J)
#SBATCH --account wang-compbio

########## Command Lines to Run ##########
conda init bash
export PATH="/mnt/home/yangji73/simon/miniconda3/bin:$PATH"
source ~/.bashrc
conda activate pygdgl
cd /mnt/home/yangji73/simon/GraphSlim/graphslim

srun -n 1 python train_all.py -D ${1} -M ${2} -R ${3} -A ${4} -P ${5}

#scontrol show job $SLURM_J0B_ID