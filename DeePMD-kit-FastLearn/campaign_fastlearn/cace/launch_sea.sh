#!/usr/bin/env bash
# Launch the two se_a-descriptor cace arms (see fit_cace_sea.py) on a cuda host.
#
#   cace-sea-sr   deepmd se_a -> Atomwise -> Forces              the control
#   cace-sea-lr   the same, plus cace-lr's Atomwise(q) + EwaldPotential block
#
# Both are the campaign's cace-<arm> recipe with ONLY the representation swapped
# for deepmd's se_a (and the SR head's width, which follows from it): 80,000
# optimizer steps on the same 320 training frames, same schedule, losses, phases.
# run_campaign.sh deliberately does not cover them, for the same reason
# run_extend.sh is separate: it skips any holder that already holds DONE, and the
# cace holders do.
#
# The two run CONCURRENTLY by default (they are small - 119k params, 2-frame
# batches, ~600 kB checkpoints - and the main campaign reaches 3 concurrent jobs
# on this pod, so the wall clock halves). The descriptor work is cpu-bound and the
# pod has 14 cores, so 2x4 threads fit.
#
# Usage:
#   ./launch_sea.sh                     # both arms, 2 at a time
#   ./launch_sea.sh --only sr
#   ./launch_sea.sh --dry-run
#
# Environment overrides: PY, CACE_ROOT, CUDA_VISIBLE_DEVICES,
# PYTORCH_CUDA_ALLOC_CONF, and the three thread counts.
#
# A job is skipped when its holder already holds DONE, so an interrupted launch
# resumes without redoing a finished arm. A non-zero exit leaves FAILED behind and
# is listed at the end - grep for `!!! FAILED`, because a partial batch that looks
# complete is the expensive failure mode.
#
# The checkpoints exist only on this host. They are small and rewritten often, so
# rsync the two holders home while these run rather than after.
set -u
cd "$(dirname "$0")"
CAMPAIGN=$(pwd)

JOBS=2
ONLY='.'
DRY=0
# The pod puts no python on PATH at all, not even for a login shell: every job
# died with "python: command not found" (observed 2026-09-23). The venv holding dp
# is the one with the modified deepmd and this host's cace, so resolve it
# explicitly rather than inheriting whatever the caller had.
PY=${PY:-/root/venv310/bin/python}
CACE_ROOT=${CACE_ROOT:-/root/app/cace-ts}
ALLOC=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export DP_INTRA_OP_PARALLELISM_THREADS=${DP_INTRA_OP_PARALLELISM_THREADS:-4}
export DP_INTER_OP_PARALLELISM_THREADS=${DP_INTER_OP_PARALLELISM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF="$ALLOC"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) awk 'NR>1 && /^set -u/ {exit} NR>1 {print}' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "campaign   $CAMPAIGN"
echo "only /$ONLY/   jobs $JOBS"
echo "python     $PY   (cace root $CACE_ROOT)"
echo "gpu        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  alloc=$ALLOC"
echo

# Preflight: an unrunnable python, an unimportable cace or an invisible gpu fails
# both arms in seconds, and because the summary at the end reports those as
# training failures, a pure setup error reads as a broken pair of arms.
if [ "$DRY" = 0 ]; then
  if ! CACE_ROOT="$CACE_ROOT" "$PY" - <<'PYEOF'
import os
import sys

import torch

sys.path.insert(0, os.environ["CACE_ROOT"])
import cace

assert os.path.abspath(cace.__file__).startswith(
    os.path.abspath(os.environ["CACE_ROOT"]) + os.sep), cace.__file__
assert torch.cuda.is_available(), "no cuda device visible"
print(f"  ok  cace {cace.__file__}  |  torch {torch.__version__}")
PYEOF
  then
    echo "!!! preflight failed: $PY cannot import cace from $CACE_ROOT with cuda visible" >&2
    exit 2
  fi
fi

selected=0
queued=0
POOL=()
failed=""

# `jobs -r` inside a command substitution reports nothing, because the
# substitution is a subshell with its own empty job table - so the pool is tracked
# by PID instead, and a reaped child stops answering kill -0. (Same as
# run_campaign.sh, which learned this the hard way.)
pool_count() {
  local alive=0 pid
  for pid in ${POOL[@]+"${POOL[@]}"}; do
    if kill -0 "$pid" 2>/dev/null; then alive=$((alive + 1)); fi
  done
  echo "$alive"
}

launch() { # arm
  local arm="$1" holder="$CAMPAIGN/runs/cace-sea-$1"
  if [ -f "$holder/DONE" ]; then
    echo "skip   cace-sea-$arm (DONE)"
    return
  fi
  selected=$((selected + 1))
  if [ "$DRY" = 1 ]; then
    echo "would  cace-sea-$arm -> $holder"
    return
  fi
  mkdir -p "$holder"
  rm -f "$holder/DONE" "$holder/FAILED"
  (
    rc=0
    # the arm chdirs into the holder itself, so best_model.pth and sea_terms.tsv
    # land there with train.log
    "$PY" "$CAMPAIGN/fit_cace_sea.py" --arm "$arm" --out-dir "$holder" \
      --cace-root "$CACE_ROOT" --device cuda > "$holder/train.log" 2>&1 || rc=$?
    if [ "$rc" -eq 0 ]; then
      touch "$holder/DONE"
    else
      echo "exit=$rc" > "$holder/FAILED"
    fi
  ) &
  POOL+=("$!")
  queued=$((queued + 1))
  echo "start  cace-sea-$arm -> $holder"
}

for arm in sr lr; do
  [ "$arm" = "${ONLY}" ] || [ "$ONLY" = "." ] || continue
  while [ "$(pool_count)" -ge "$JOBS" ]; do sleep 3; done
  launch "$arm"
done

if [ "$DRY" = 1 ]; then
  echo
  echo "$selected job(s) selected"
  exit 0
fi

echo
echo "waiting for $queued job(s) ..."
wait

echo
echo "=== summary $(date) ==="
for arm in sr lr; do
  holder="$CAMPAIGN/runs/cace-sea-$arm"
  [ -d "$holder" ] || continue
  if [ -f "$holder/FAILED" ]; then
    failed="$failed cace-sea-$arm"
  elif [ -f "$holder/DONE" ]; then
    printf '  ok    %-16s %s (%s term rows)\n' "cace-sea-$arm" \
      "$(ls "$holder" 2>/dev/null | grep -c '^model.*\.pth$') checkpoints" \
      "$(( $(wc -l < "$holder/sea_terms.tsv" 2>/dev/null || echo 1) - 1 ))"
  fi
done
if [ -n "$failed" ]; then
  echo "!!! FAILED:$failed"
  for arm in $failed; do
    echo "--- last lines of $arm/train.log"
    tail -n 15 "$CAMPAIGN/runs/$arm/train.log" 2>/dev/null
  done
  exit 1
fi
echo "ALL DONE $(date)"
