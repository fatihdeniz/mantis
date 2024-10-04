#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate fatih

# Array of epsilon values
epsilon_values=(0.02 0.05 0.08 0.11 0.14 0.17 0.20 0.30 0.40 0.50 0.60 0.70 0.80)

# Loop through each epsilon value and execute sequentially
for epsilon in "${epsilon_values[@]}"; do
    echo "Running training with epsilon = $epsilon"
    python -u train_adv.py --epsilon $epsilon
done

# # if you have multiple GPUs
# GPU_COUNT=4  # Modify this based on your system's GPU count

# # Wait for a free GPU
# wait_for_gpu() {
#     while [ $(jobs -p | wc -l) -ge $GPU_COUNT ]; do
#         sleep 100
#     done
# }

# # Loop through each epsilon value and execute in parallel based on GPU count
# for epsilon in "${epsilon_values[@]}"; do
#     wait_for_gpu  # Wait for a free GPU before starting a new process
    
#     echo "Running training with epsilon = $epsilon"
#     python -u train_adv.py --epsilon $epsilon &
    
#     sleep 100  # Short pause to allow process to start
# done

# # Wait for all background processes to finish
# wait