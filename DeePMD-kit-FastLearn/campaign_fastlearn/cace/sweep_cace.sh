#!/usr/bin/env bash
# Score every kept cace checkpoint of both arms on the campaign's validation split.
#
# Five checkpoints per arm: the four phase boundaries (`fit_cace.py` writes
# `model.pth`, `model-2/3/4.pth` at epochs 200/300/400/500) plus `best_model.pth`,
# which is selected on this same 80-frame split and is therefore reported and
# flagged by `collect_sweep.py` but excluded from the headline mean.
#
# Writes per-checkpoint JSON + NPZ under `runs/cace-<arm>/`, and the training
# trace once per arm (`--epochs-tsv`, on the first stem only - it is arm-level).
# Then run `python collect_sweep.py` to fold them into the two TSVs the notebook reads.
#
#   PYTHON=/root/venv310/bin/python ./sweep_cace.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
cd "$HERE" || exit 1

for arm in sr lr; do
    mkdir -p "runs/cace-$arm/npz"
    first=1
    for stem in model model-2 model-3 model-4 best_model; do
        model="runs/cace-$arm/$stem.pth"
        if [ ! -f "$model" ]; then
            echo "MISSING $model"
            continue
        fi
        trace=""
        if [ "$first" -eq 1 ]; then
            trace="--epochs-tsv runs/cace-$arm/epoch_metrics.tsv"
        fi
        echo "--- cace $arm $stem $(date -u) ---"
        "$PYTHON" score_cace.py --arm "$arm" --model "$model" \
            --tag "cace-${arm}_${stem}" \
            --out-json "runs/cace-$arm/$stem.json" \
            --npz "runs/cace-$arm/npz/$stem.npz" $trace
        echo "    exit=$?"
        first=0
    done
done

echo "--- cace sweep done $(date -u) ---"
