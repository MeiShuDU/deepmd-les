#!/bin/bash
# Run deepmd-cace-eq_sA (the full 80,000-step schedule) in this directory.
#
#   ./run.sh                 # train on the device in input.json (cuda)
#   ./run.sh --check-only    # forward + backward + descriptor-grad check, no training
#
# DP_INTERFACE_PREC=high is required: input.json sets the se_a descriptor to
# float64, matching cace-sea-lr's SE_A, and DeepmdSeAInput casts coordinates to
# deepmd's global precision. Setting it "low" would feed float32 coordinates to a
# float64 descriptor.
#
# For the smoke test use ../../runs_smoke/deepmd-cace-eq_sA/smoke.sh instead,
# which trains one epoch per block on CPU and keeps this directory free of
# half-trained checkpoints. For timed/on-pod runs prefer
#   python run_deepmd_cace.py --root <this campaign>/deepmd/runs \
#       --arm deepmd-cace-eq --rep A --precision float64
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(cd ../../../../.. && pwd)"
export PYTHONPATH="$REPO/desc_bridging${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DP_INTERFACE_PREC="${DP_INTERFACE_PREC:-high}"
python -m deepmd_cace input.json "$@"
