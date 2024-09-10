#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=a100v100
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --partition=a100-8-gm320-c96-m1152
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate pygdgl

# Change directory to the project folder
cd ~/GraphSlim/graphslim

# Define hyperparameter values for grid search
lr_feat_values=(1e-4 1e-3 1e-2 1e-1)
lr_adj_values=(1e-4 1e-3 1e-2 1e-1)
pre_norm_values=(true)
dis_metric_values=("ours")
outer_loop_values=(5 10)
inner_loop_values=(0 1)
threshold_values=(0.0 0.05 0.1)
condense_model_values=("SGC" "GCN")
ntrans_values=(1)
epochs_values=(2000 3000)

# Paths
SAVE_PATH="/scratch/sgong36/checkpoints"
LOAD_PATH="/scratch/sgong36/data"

best_val_acc=0.0
best_test_acc=0.0
best_config=""

# Start grid search
echo '====start running grid search===='

for lr_feat in "${lr_feat_values[@]}"; do
  for lr_adj in "${lr_adj_values[@]}"; do
    for pre_norm in "${pre_norm_values[@]}"; do
      for dis_metric in "${dis_metric_values[@]}"; do
        for outer_loop in "${outer_loop_values[@]}"; do
          for inner_loop in "${inner_loop_values[@]}"; do
            for threshold in "${threshold_values[@]}"; do
              for condense_model in "${condense_model_values[@]}"; do
                for ntrans in "${ntrans_values[@]}"; do
                  for epochs in "${epochs_values[@]}"; do
                    # Construct the command with current hyperparameters
                    echo "Running with lr_feat=$lr_feat, lr_adj=$lr_adj, pre_norm=$pre_norm, dis_metric=$dis_metric, outer_loop=$outer_loop, inner_loop=$inner_loop, threshold=$threshold, condense_model=$condense_model, ntrans=$ntrans, epochs=$epochs"
                    
                    # Run the training script and capture the output
                    output=$(python train_all.py -M gcdm -D ogbn-arxiv\
                        --save_path "$SAVE_PATH" \
                        --load_path "$LOAD_PATH" \
                        --lr_feat "$lr_feat" \
                        --lr_adj "$lr_adj" \
                        --pre_norm "$pre_norm" \
                        --dis_metric "$dis_metric" \
                        --outer_loop "$outer_loop" \
                        --inner_loop "$inner_loop" \
                        --threshold "$threshold" \
                        --condense_model "$condense_model" \
                        --ntrans "$ntrans" \
                        --epochs "$epochs")
                    
                    # Extract the validation accuracy and test accuracy
                    val_acc=$(echo "$output" | grep "Val Accuracy and Std" | tail -1 | awk -F '[,\\[\\]]' '{print $2}')
                    test_acc=$(echo "$output" | grep "Test Mean Accuracy" | tail -1 | awk -F '[ ,]' '{print $5}')
                    
                    # Ensure val_acc is not empty and use bc for comparison
                    if [[ ! -z "$val_acc" ]]; then
                      # Compare val_acc with best_val_acc using bc
                      if (( $(echo "$val_acc > $best_val_acc" | bc -l) )); then
                        best_val_acc=$val_acc
                        best_test_acc=$test_acc
                        best_config="lr_feat=$lr_feat, lr_adj=$lr_adj, pre_norm=$pre_norm, dis_metric=$dis_metric, outer_loop=$outer_loop, inner_loop=$inner_loop, threshold=$threshold, condense_model=$condense_model, ntrans=$ntrans, epochs=$epochs"
                      fi
                    fi
                    
                    echo "Current Best Val Accuracy: $best_val_acc with Test Accuracy: $best_test_acc"
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

# Output the best configuration and its test accuracy
echo "=====Grid Search Finished====="
echo "Best Validation Accuracy: $best_val_acc"
echo "Corresponding Test Accuracy: $best_test_acc"
echo "Best Hyperparameter Configuration: $best_config"
