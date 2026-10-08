#!/bin/bash
# Local CPU smoke test for deepmd-cace-eq_sA.
#
# Runs against the copy of input.json in this directory (byte-identical to
# ../../runs/deepmd-cace-eq_sA/input.json - same relative data paths, same
# schedule), so every artifact lands here and the production run directory stays
# free of one-epoch checkpoints that could later be mistaken for a trained model.
#
#   ./smoke.sh            # --check-only, then one epoch per fit block
#   ./smoke.sh --check    # --check-only only (fast)
#
# --check-only is the real gate: it builds the se_a descriptor, computes its
# statistics over the 320 training frames, runs a full forward (energy, force
# derivative, ChargeEqLatent solve) and asserts the gradient reaches se_a.
# --smoke then runs all 8 fits at one epoch each (8 x 160 = 1,280 steps) to
# exercise the block/fresh-task/loss-swap plumbing.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(cd ../../../../.. && pwd)"
export PYTHONPATH="$REPO/desc_bridging${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DP_INTERFACE_PREC="${DP_INTERFACE_PREC:-high}"
export CUDA_VISIBLE_DEVICES=""
export DEEPMD_CACE_DEVICE=cpu

echo "== check-only =="
python -m deepmd_cace input.json --check-only

if [[ "${1:-}" == "--check" ]]; then
    echo "check-only done; --smoke not requested"
    exit 0
fi

echo
echo "== smoke (1 epoch per fit block, CPU) =="
python -m deepmd_cace input.json --smoke
