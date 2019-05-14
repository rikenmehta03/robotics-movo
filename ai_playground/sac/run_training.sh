#!/bin/bash

CDIR=./runs

for SEED in 1; do
  SUBDIR=seed_${SEED}
  SAVEDIR=${CDIR}/${SUBDIR}
  screen -dmS ${SUBDIR} ./run.sh ${SAVEDIR} ${SEED}
done