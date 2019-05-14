#!/bin/bash

CDIR=./runs
mkdir -p ${CDIR}

for SEED in 1 3 5 7 9; do
  SUBDIR=seed_${SEED}
  SAVEDIR=${CDIR}/${SUBDIR}
  mkdir -p ${SAVEDIR}
  screen -dmS ${SUBDIR} bash -c 'source ~/movo/movo-env/bin/activate; python train.py \
    --eval_freq 10000 \
    --start_steps 10000 \
    --max_steps 10000000 \
    --num_eval_episodes 5 \
    --batch_size 1024  \
    --save_dir ${SAVEDIR} \
    --seed ${SEED} \
    --use_tb'
done