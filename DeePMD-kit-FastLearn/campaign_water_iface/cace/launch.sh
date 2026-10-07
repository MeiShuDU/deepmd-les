#!/usr/bin/env bash
# Run the four CACE water/slab-interface arms sequentially.
#
#   ./launch.sh [PYTHON] [extra run_cace_nnp.py flags...]
#
# Each run gets its own directory under runs/, and its stdout/stderr goes to
# runs/<arm>_s<X>/run.log. The train/valid split (np.random.default_rng(1)) is
# identical across replicates; only the torch seed differs (10 for sA, 20 for sB).
#
#   PYTHON defaults to `python`. On the pod use /root/venv310/bin/python.
#   Add --smoke to shrink every block to 2 epochs (timing calibration only).
set -euo pipefail

PY="${1:-python}"
shift || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS="$HERE/runs"
mkdir -p "$RUNS"

for arm in lr sr; do
  for rep in sA sB; do
    out="$RUNS/${arm}_${rep}"
    mkdir -p "$out"
    echo "[launch] arm=$arm replicate=$rep -> $out"
    "$PY" "$HERE/run_cace_nnp.py" --arm "$arm" --replicate "$rep" --out "$out" "$@" \
      > "$out/run.log" 2>&1
    echo "[launch] done arm=$arm replicate=$rep"
  done
done

echo "[launch] all four cace runs finished"
