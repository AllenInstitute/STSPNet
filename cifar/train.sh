#!/bin/bash

source activate pytorch-latest

seed_arr=($(seq 1 1 10))

# train
for seed in "${seed_arr[@]}"
do
    python main.py --seed=$seed --save-model
done
