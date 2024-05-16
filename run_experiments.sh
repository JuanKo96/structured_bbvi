#!/bin/bash

# seeds=(0 1 2)
# model_types=("DiagonalVariational" "FullRankVariational" "StructuredVariational")

# Define the number of GPUs available
num_gpus=2
N=1000
n_iterations=1

# Path to the config file
config_file="config.yaml"

# Run experiments
for seed in 0 1 2; do
    for model_type in "DiagonalVariational" "FullRankVariational" "StructuredVariational"; do
        gpu_id=seed

        CUDA_VISIBLE_DEVICES=$gpu_id python main.py --config $config_file --seed $seed --model_type $model_type --N $N --n_iterations $n_iterations&
        # sleep 1
    done
done
wait
