#!/bin/bash
# Run one sea-lr-eq replicate directly, without the timing wrapper.
# Prefer the campaign runner, which also records timing:
#   python run_deepmd_cace.py --arm sea-lr-eq --rep A
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(cd ../../../../.. && pwd)"
export PYTHONPATH="$REPO/desc_bridging${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -m deepmd_cace input.json "$@"
