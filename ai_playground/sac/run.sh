#!/bin/bash

source ~/movo/movo-env/bin/activate
python train.py \
    --eval_freq 10000 \
    --start_steps 10000 \
    --max_steps 10000000 \
    --num_eval_episodes 5 \
    --batch_size 1024  \
    --save_dir $1 \
    --seed $2 \
    --use_tb