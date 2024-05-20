#!/bin/bash

N=1000
n_iterations=10000

# Path to the config file
config_file="config.yaml"
# File to store PIDs
pid_file="pids.txt"

# Clear the pid_file
> $pid_file
# Run experiments
for seed in 0; do
    CUDA_VISIBLE_DEVICES=1 python main.py --config $config_file --seed $seed --model_type "DiagonalVariational" --N $N --n_iterations $n_iterations&
    echo $! >> $pid_file
    CUDA_VISIBLE_DEVICES=2 python main.py --config $config_file --seed $seed --model_type "FullRankVariational" --N $N --n_iterations $n_iterations&
    echo $! >> $pid_file
    CUDA_VISIBLE_DEVICES=3 python main.py --config $config_file --seed $seed --model_type "StructuredVariational" --N $N --n_iterations $n_iterations&
    echo $! >> $pid_file
done
wait
