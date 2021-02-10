#!/bin/bash

seed_arr=($(seq 1 1 10))

# test
for seed in "${seed_arr[@]}"
do
    python get_image_features.py --seed=$seed
done
