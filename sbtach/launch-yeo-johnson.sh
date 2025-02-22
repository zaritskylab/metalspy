#!/bin/bash

HIST_SIZE=20
EXCLUDE_CORES=False
P="0.8"
for METAL in "magnesium" "iron" "copper" "zinc"; do
  sbatch \
    --export=HIST_SIZE=${HIST_SIZE},EXCLUDE_CORES=${EXCLUDE_CORES},METAL=${METAL},P=${P} \
    --job-name=yeo_johnson_hist_size_${HIST_SIZE}_${EXCLUDE_CORES}_${METAL}_${P} \
    --output=yeo_johnson_hist_size_${HIST_SIZE}_${EXCLUDE_CORES}_${METAL}_${P}.out \
    job-yeo-johnson.batch
done
