#!/bin/bash

source ~/movo/movo-env/bin/activate
python train.py \
    --save_dir $1 \
    --seed $2 \
    --use_tb