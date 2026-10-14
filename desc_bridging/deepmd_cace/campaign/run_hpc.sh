#!/usr/bin/env bash
# Run the formal bridge campaign on the already-provisioned HPC GPU host.
# This script never provisions or starts a remote instance; run it after the
# local smoke test and after the environment has passed hpc_setup.sh checks.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
DATASET=${1:?usage: run_hpc.sh DATASET_ROOT [OUTPUT_ROOT]}
OUTPUT=${2:-"$PROJECT_ROOT/campaign_runs/formal"}
DEVICE=${DEVICE:-cuda}
PYTHON=${PYTHON:-python}

if [[ "$DEVICE" == cuda ]] && ! "$PYTHON" - <<'PY' 2>/dev/null
import torch
assert torch.cuda.is_available()
PY
then
    echo "CUDA is unavailable; set DEVICE=cpu only for an intentional non-formal test" >&2
    exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/desc_bridging${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m deepmd_cace.campaign.run \
    --mode formal \
    --dataset "$DATASET" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --arm both
