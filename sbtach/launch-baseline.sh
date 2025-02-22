#!/bin/bash

HIST_SIZE=20
EXCLUDE_CORES=False
P=0.8
for METAL in "magnesium" "iron" "copper" "zinc"; do
  sbatch \
    --export=HIST_SIZE=${HIST_SIZE},EXCLUDE_CORES=${EXCLUDE_CORES},METAL=${METAL},P=${P} \
    --job-name=baseline_hist_size_${HIST_SIZE}_${EXCLUDE_CORES}_${METAL}_${P} \
    --output=baseline_hist_size_${HIST_SIZE}_${EXCLUDE_CORES}_${METAL}_${P}.out \
    baseline.batch
done
