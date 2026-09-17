#!/bin/bash
# Sweep the late checkpoints on the full validation set.
# One (run, step) per process: a single CUDA context loading several models
# trips an allocator internal assert on this host (c10 CUDACachingAllocator
# "!handles_.at(i)"), so never batch runs. expandable_segments is likewise not
# set here - the eval footprint is tiny and the flag is implicated in that
# assert; it is only needed for the long trainings.
#
# Usage: bash sweep_valid.sh [run_dir ...]
# With no arguments it sweeps the six original (decay_steps 5000) arms. The
# output tsv name follows the runs' learning-rate schedule (eval_full.py picks
# it), so a second schedule sweeps into its own file.
set -u
cd "$(dirname "$0")"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py39

if [ "$#" -gt 0 ]; then
  RUNS="$*"
else
  RUNS="run_ordinary_sA run_hybrid_q_sA run_hybrid_fixed_sA
        run_ordinary_sB run_hybrid_q_sB run_hybrid_fixed_sB"
fi

first=1
for step in 6000 7000 8000 9000 10000; do
  echo "=== step $step $(date) ==="
  for r in $RUNS; do
    reset=""
    if [ $first -eq 1 ]; then reset="--reset"; first=0; fi
    python eval_full.py valid "$step" "$r" $reset 2>&1 \
      | grep -E "rmse_e/Natoms|INTERNAL|Error" \
      || echo "!!! FAILED $r step=$step"
  done
done
echo "SWEEP DONE $(date)"
