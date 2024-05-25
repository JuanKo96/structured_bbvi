#!/bin/bash

# Path to the config file
config_file="config.yaml"
# File to store PIDs
pid_file="pids.txt"

# Clear the pid_file
> $pid_file

# List of model types

# Outer loop for 8 sets of seeds
for outer in $(seq 0 0); do
    # Inner loop for 4 seeds in each set
    for inner in $(seq 0 2); do
        seed=$((outer * 3 + inner))
        
        CUDA_VISIBLE_DEVICES=1 python main.py --config $config_file --seed $seed --model_type "DiagonalVariational" &
        echo $! >> $pid_file
        CUDA_VISIBLE_DEVICES=0 python main.py --config $config_file --seed $seed --model_type "FullRankVariational" &
        echo $! >> $pid_file
        CUDA_VISIBLE_DEVICES=2 python main.py --config $config_file --seed $seed --model_type "StructuredVariational" &
        echo $! >> $pid_file
    done
    # Wait for the current batch of 4 processes to finish before starting the next batch
    wait
done