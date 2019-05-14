#!/bin/bash

CDIR=./runs

for SEED in 1 2 3 4 5 6 7 8 9 10; do
  SUBDIR=seed_${SEED}
  SAVEDIR=${CDIR}/${SUBDIR}
  screen -dmS ${SUBDIR} ./run.sh ${SAVEDIR} ${SEED}
done