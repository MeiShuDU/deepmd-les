#!/bin/bash
# Extended matched comparison: 10k steps, 2 replicates, 3 models.
# Single 8GB GPU -> strictly sequential. Each run owns its subdir so
# checkpoints / lcurve / les.log never clobber each other.
#
# Usage:  bash run_extended.sh [run_dir ...]
# With no arguments it runs the six original (decay_steps 5000) arms, which is
# what the reproduce block in LES_PERFORMANCE.md documents. Pass explicit run
# directory names to run a different arm set, e.g.
#   LOGTAG=_d7500 bash run_extended.sh run_*_d7500 > run_extended_d7500.log 2>&1
# LOGTAG suffixes the gpu_mem log so arm sets never overwrite each other's
# record; the caller redirects stdout to a matching log file.
set -u
cd "$(dirname "$0")/extended"
export OMP_NUM_THREADS=4 DP_INTRA_OP_PARALLELISM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=2
# expandable_segments: the CUDA caching allocator otherwise hoards its high-water
# mark (nlist grows over training), climbing to ~7.9GB/8.19GB on this WSL2 GPU and
# deadlocking in dxgkrnl (dxgvmb_send_wait_sync_object_gpu). With segments returned
# to the driver the footprint stays flat at ~5.4GB. Verified over a 3000-step probe.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGTAG="${LOGTAG:-}"

# Sample GPU memory for the whole batch so a future stall is visible in the trend.
( while true; do
    echo "$(date +%F' '%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
    sleep 30
  done > "gpu_mem${LOGTAG}.log" 2>&1 ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

if [ "$#" -gt 0 ]; then
  RUNS="$*"
else
  RUNS="run_ordinary_sA run_hybrid_q_sA run_hybrid_fixed_sA
        run_ordinary_sB run_hybrid_q_sB run_hybrid_fixed_sB"
fi

failed=""
for rundir in $RUNS; do
  echo "=== START $rundir $(date) ==="
  (
    cd "$rundir"
    rm -f model.ckpt* checkpoint les.log
    dp --pt train input.yaml > train.log 2>&1
  )
  rc=$?
  if [ $rc -ne 0 ]; then
    failed="$failed $rundir"
    echo "!!! FAILED $rundir rc=$rc $(date) (see $rundir/train.log)"
  fi
  echo "=== DONE  $rundir rc=$rc $(date) ==="
done

if [ -n "$failed" ]; then
  echo "ALL DONE WITH FAILURES:$failed $(date)"
  exit 1
fi
echo "ALL DONE $(date)"
