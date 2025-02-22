#!/bin/bash

HIST_SIZE=20
EXCLUDE_CORES="False"
for P in 0.8 0.9 0.95 0.99; do
  for METAL in "magnesium" "iron" "copper" "zinc"; do
    sbatch \
      --export=HIST_SIZE=${HIST_SIZE},EXCLUDE_CORES=${EXCLUDE_CORES},METAL=${METAL},P=${P} \
      --job-name=hotspots_excluded_hist_size_${HIST_SIZE}_${EXCLUDE_CORES}_${METAL}_${P} \
      --output=hotspots_excluded_hist_size_${HIST_SIZE}_${EXCLUDE_CORES}_${METAL}_${P}.out \
      job-hotspots-excluded.batch
  done
done
