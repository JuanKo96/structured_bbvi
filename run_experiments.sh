#!/bin/bash

seeds=(0 1 2)
model_types=("DiagonalVariational" "FullRankVariational" "StructuredVariational")

# Define the number of GPUs available
num_gpus=2

# Path to the config file
config_file="config.yaml"

# Run experiments
for seed in "${seeds[@]}"; do
    for model_type in "${model_types[@]}"; do
        gpu_id=$(( (i++ % num_gpus) ))

        CUDA_VISIBLE_DEVICES=$gpu_id python main.py --config $config_file --seed $seed --model_type $model_type &
        # sleep 1
    done
done
wait
