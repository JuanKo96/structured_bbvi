#!/bin/bash

N=1000
n_iterations=10000

# Path to the config file
config_file="config.yaml"

# Run experiments
for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=1 python main.py --config $config_file --seed $seed --model_type "DiagonalVariational" --N $N --n_iterations $n_iterations&
    CUDA_VISIBLE_DEVICES=2 python main.py --config $config_file --seed $seed --model_type "FullRankVariational" --N $N --n_iterations $n_iterations&
    CUDA_VISIBLE_DEVICES=3 python main.py --config $config_file --seed $seed --model_type "StructuredVariational" --N $N --n_iterations $n_iterations&
done
wait
