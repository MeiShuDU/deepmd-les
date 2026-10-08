#!/bin/bash
# Matched rerun: ordinary vs hybrid_ener (learnable q) vs hybrid_ener (fixed q).
# Same data, same SR net, same LR schedule, equal steps (2000).
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 DP_INTRA_OP_PARALLELISM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=2

run () {  # $1=input  $2=tag
  echo "=== START $2 $(date) ==="
  [ -f les.log ] && mv les.log "les_${2}.log"
  rm -f model.ckpt* checkpoint
  dp --pt train "$1" > "train_${2}.log" 2>&1
  mv les.log "les_${2}.log" 2>/dev/null || true
  echo "=== DONE  $2 $(date) ==="
}

run input_ordinary.yaml   ordinary
run input_hybrid_q.yaml   hybrid_q
run input_hybrid_fixed.yaml hybrid_fixed
echo "ALL DONE $(date)"
