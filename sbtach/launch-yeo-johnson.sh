#!/bin/bash

HIST_SIZE=20
EXCLUDE_CORES=False
P="0.8"
for METAL in "magnesium" "iron" "copper" "zinc"; do
  sbatch \
    --export=HIST_SIZE=${HIST_SIZE},EXCLUDE_CORES=${EXCLUDE_CORES},METAL=${METAL},P=${P} \
    --job-name=yeo-johnson_hist_size_${HIST_SIZE}_${EXCLUDE_CORES}_${METAL}_${P} \
    --output=yeo-johnson_hist_size_${HIST_SIZE}_${EXCLUDE_CORES}_${METAL}_${P}.out \
    yeo-johnson.batch
done
