#!/bin/bash

CDIR=./runs
mkdir -p ${CDIR}

for SEED in 1 2 3 4 5 6 7 8 9 10; do
  SUBDIR=seed_${SEED}
  SAVEDIR=${CDIR}/${SUBDIR}
  mkdir -p ${SAVEDIR}
  screen -dmS ${SUBDIR} source ~/movo/movo-env/bin/activate && python train.py \
    --eval_freq 10000 \
    --start_steps 10000 \
    --max_steps 10000000 \
    --num_eval_episodes 5 \
    --batch_size 1024  \
    --save_dir ${SAVEDIR} \
    --seed ${SEED} \
    --use_tb
done