#!/usr/bin/env bash
# Score every kept checkpoint of the two se_a-descriptor arms (see fit_cace_sea.py).
#
# Same five checkpoints per arm as `sweep_cace.sh` - the four phase boundaries
# (`model.pth`, `model-2/3/4.pth` at epochs 200/300/400/500) plus `best_model.pth` -
# but through `score_cace.py --arm sea-<arm>`, which loads them with
# `sea_seam.load_sea_checkpoint` (a state dict + rebuild recipe) instead of a
# module pickle, and records the se_a rebuild spec in the score JSON.
#
# Two differences from the cace driver, both forced by the artifacts:
#
#   * The training trace is written FIRST and unconditionally, via
#     `--epochs-only`. For the cace arms the trace rode along with the first
#     scored checkpoint; here it has to stand alone, because the phase checkpoints
#     do not exist until epoch 200 while the trace is a pure function of the log.
#     That is what lets `collect_sweep.py` and the notebook show a training curve
#     while the run is still going.
#   * `best_model.pth` exists long before the others, so a mid-run invocation
#     scores it and reports the rest as MISSING. That is the supported use: the
#     numbers are superseded, the wiring is what is being checked.
#
# Writes per-checkpoint JSON + NPZ under `runs/cace-sea-<arm>/`, the trace as
# `epoch_metrics.tsv` there, then run `python collect_sweep.py` to fold everything
# into the `analysis/data/cace_sea_*.tsv` tables the notebook reads.
#
#   PYTHON=/root/miniconda3/envs/py39/bin/python ./sweep_sea.sh
#   PYTHON=... CUDA_VISIBLE_DEVICES="" ./sweep_sea.sh            # cpu, for a check
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
cd "$HERE" || exit 1

for arm in sr lr; do
    holder="runs/cace-sea-$arm"
    mkdir -p "$holder/npz"

    # the trace is exactly the epochs we have finished, so this also works on a
    # run that has not written a single checkpoint yet
    if [ -f "$holder/train.log" ]; then
        echo "--- sea $arm trace $(date -u) ---"
        "$PYTHON" score_cace.py --arm "sea-$arm" --epochs-only \
            --log "$holder/train.log" --epochs-tsv "$holder/epoch_metrics.tsv"
        echo "    exit=$?"
    else
        echo "MISSING $holder/train.log (no trace written)"
    fi

    for stem in model model-2 model-3 model-4 best_model; do
        model="$holder/$stem.pth"
        if [ ! -f "$model" ]; then
            echo "MISSING $model"
            continue
        fi
        echo "--- sea $arm $stem $(date -u) ---"
        "$PYTHON" score_cace.py --arm "sea-$arm" --model "$model" \
            --tag "cace-sea-${arm}_${stem}" \
            --out-json "$holder/$stem.json" \
            --npz "$holder/npz/$stem.npz"
        echo "    exit=$?"
    done
done

echo "--- sea sweep done $(date -u) ---"
