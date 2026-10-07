#!/usr/bin/env bash
# Run the FastLearn cace-vs-deepmd campaign: 8 deepmd arms + 2 cace arms.
#
#   deepmd   standard                        SR only, the control
#   deepmd-les / -claim-neutral / -freeze-charge    x replicates A and B
#   cace-sr  Cace + Atomwise only            the author's fitted control
#   cace-lr  Cace + Atomwise + Ewald         the author's long-range recipe
#
# Both families run 80,000 optimizer steps on the same 320 training frames;
# see deepmd/gen_inputs.py and cace/fit_cace.py for the settings and for how
# each side's recipe was transcribed.
#
# One script rather than two so the two families cannot drift apart: the only
# per-family difference is the command each job runs.
#
# Usage:
#   ./run_campaign.sh                       # everything, 3 jobs at a time
#   ./run_campaign.sh --family deepmd --jobs 4
#   ./run_campaign.sh --only 'les-.*_sA'    # ERE match on the run name
#   ./run_campaign.sh --dry-run
#
# Environment overrides: DP (default `dp`), PY, CACE_ROOT, CUDA_VISIBLE_DEVICES,
# PYTORCH_CUDA_ALLOC_CONF, and the three thread counts.
#
# A job is skipped when its directory already holds DONE, so an interrupted
# batch resumes without redoing finished arms. Anything that exits non-zero
# leaves FAILED behind and is listed again at the end - grep for `!!! FAILED`,
# because a partial batch that looks complete is the expensive failure mode.
set -u
cd "$(dirname "$0")"
CAMPAIGN=$(pwd)

JOBS=3
FAMILY=all
ONLY='.'
DRY=0
# The venv holding dp is not on PATH, not even for a login shell, so resolve it
# explicitly rather than inheriting whatever the caller had. Getting this wrong
# fails every deepmd arm in ~20 s with "dp: command not found" (observed
# 2026-09-21, and again on 2026-09-21 in run_extend.sh), which is why the
# preflight below now asserts dp is runnable before anything is launched.
if [ -z "${DP:-}" ]; then
  if [ -x /root/venv310/bin/dp ]; then DP=/root/venv310/bin/dp; else DP=dp; fi
fi
PY=${PY:-python}
# the author's fit_cace_new.py appends ../cace/ #torchscript branch, so the
# default mirrors that branch rather than whichever `cace` happens to be
# importable. Override on a host that lays it out differently.
CACE_ROOT=${CACE_ROOT:-/root/app/cace-ts}
ALLOC=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# A job is latency-bound on the data pipeline and small kernels, so extra
# threads do not help but extra processes do.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export DP_INTRA_OP_PARALLELISM_THREADS=${DP_INTRA_OP_PARALLELISM_THREADS:-4}
export DP_INTER_OP_PARALLELISM_THREADS=${DP_INTER_OP_PARALLELISM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF="$ALLOC"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --family) FAMILY="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) awk 'NR>1 && /^set -u/ {exit} NR>1 {print}' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# name|kind|target
#   deepmd  target is the run directory, which is also the holder
#   cace    target is the arm (sr|lr); the holder is cace/runs/cace-<arm>
#
# The cace arms are listed first on purpose: they are two orders of magnitude
# cheaper than a deepmd arm, so starting them first fills the spare slots while
# the first deepmd arms are still warming up, and it surfaces any cace-specific
# problem in minutes instead of after hours of deepmd training.
jobs_list() {
  if [ "$FAMILY" = all ] || [ "$FAMILY" = cace ]; then
    for arm in sr lr; do
      printf 'cace-%s|cace|%s\n' "$arm" "$arm"
    done
  fi
  if [ "$FAMILY" = all ] || [ "$FAMILY" = deepmd ]; then
    for rundir in "$CAMPAIGN"/deepmd/runs/*/; do
      [ -d "$rundir" ] || continue
      # Only a native `dp --pt train` arm owns an input.yaml in its run dir. The
      # chained arms (deepmd-*-cace-sched_s*) keep theirs in per-block s1..s8
      # subdirs and the deepmd_cace arm (deepmd-cace-eq_sA) uses input.json; both
      # have their own runners. Claiming them here would only cd into their
      # directory, fail on the missing file and leave a FAILED marker plus a
      # train.log behind, which reads as a broken arm in the summary below.
      if [ ! -f "$rundir/input.yaml" ]; then
        echo "skip   $(basename "${rundir%/}") (no input.yaml; not a native deepmd arm)" >&2
        continue
      fi
      printf '%s|deepmd|%s\n' "$(basename "${rundir%/}")" "${rundir%/}"
    done
  fi
}

holder_for() { # kind target
  case "$1" in
    deepmd) echo "$2" ;;
    cace) echo "$CAMPAIGN/cace/runs/cace-$2" ;;
  esac
}

echo "campaign   $CAMPAIGN"
echo "family     $FAMILY   only /$ONLY/   jobs $JOBS"
echo "dp         $DP"
echo "python     $PY   (cace root $CACE_ROOT)"
echo "gpu        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  alloc=$ALLOC"
echo

# Preflight: nothing is launched until dp is actually runnable. An unrunnable DP
# fails every deepmd arm within ~20 s, and because the summary at the end lists
# them as training failures, a pure setup error reads as a broken campaign. The
# cace arms do not need dp, so a cace-only run is not blocked by it.
if { [ "$FAMILY" = all ] || [ "$FAMILY" = deepmd ]; } \
   && ! command -v "$DP" >/dev/null 2>&1; then
  echo "!!! dp not runnable: $DP   (set DP=/root/venv310/bin/dp)" >&2
  exit 2
fi

selected=0
queued=0
POOL=()

# `jobs -r` inside a command substitution reports nothing, because the
# substitution is a subshell with its own empty job table - so the pool is
# tracked by PID instead, and a reaped child stops answering kill -0.
pool_count() {
  local alive=0 pid
  for pid in ${POOL[@]+"${POOL[@]}"}; do
    if kill -0 "$pid" 2>/dev/null; then alive=$((alive + 1)); fi
  done
  echo "$alive"
}

launch() { # name kind target holder
  local name="$1" kind="$2" target="$3" holder="$4"
  if [ -f "$holder/DONE" ]; then
    echo "skip   $name (DONE)"
    return
  fi
  selected=$((selected + 1))
  if [ "$DRY" = 1 ]; then
    echo "would  $name -> $holder [$kind]"
    return
  fi
  rm -f "$holder/DONE" "$holder/FAILED"
  (
    rc=0
    if [ "$kind" = deepmd ]; then
      # CWD must be the run dir: lcurve.out goes to the CWD and the les package
      # hardcodes les.log there, which is what keeps arms from colliding.
      cd "$target" || rc=1
      if [ "$rc" -eq 0 ]; then
        "$DP" --pt train input.yaml > "$holder/train.log" 2>&1
        rc=$?
      fi
    else
      mkdir -p "$holder"
      "$PY" "$CAMPAIGN/cace/fit_cace.py" --arm "$target" --out-dir "$holder" \
        --cace-root "$CACE_ROOT" --device cuda > "$holder/train.log" 2>&1
      rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
      touch "$holder/DONE"
    else
      echo "exit=$rc" > "$holder/FAILED"
    fi
  ) &
  POOL+=("$!")
  queued=$((queued + 1))
  echo "start  $name -> $holder"
}

while IFS='|' read -r name kind target; do
  [ -n "$name" ] || continue
  if ! [[ "$name" =~ $ONLY ]]; then continue; fi
  holder=$(holder_for "$kind" "$target")
  while [ "$(pool_count)" -ge "$JOBS" ]; do sleep 3; done
  launch "$name" "$kind" "$target" "$holder"
done < <(jobs_list)

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
done_n=0
failed=""
for holder in "$CAMPAIGN"/deepmd/runs/*/ "$CAMPAIGN"/cace/runs/*/; do
  [ -d "$holder" ] || continue
  name=$(basename "${holder%/}")
  if [ -f "$holder/FAILED" ]; then
    failed="$failed $name"
  elif [ -f "$holder/DONE" ]; then
    done_n=$((done_n + 1))
    last=$(ls "$holder" 2>/dev/null | grep -oE 'model\.ckpt-[0-9]+\.pt' \
           | sort -t- -k2 -n | tail -1)
    printf '  ok    %-36s %s\n' "$name" "${last:-saved}"
  fi
done
echo "done:   $done_n"
if [ -n "$failed" ]; then
  echo "!!! FAILED:$failed"
  for name in $failed; do
    echo "--- last lines of $name/train.log"
    tail -n 15 "$(find "$CAMPAIGN"/deepmd/runs "$CAMPAIGN"/cace/runs -maxdepth 1 -name "$name")/train.log" 2>/dev/null
  done
  exit 1
fi
echo "ALL DONE $(date)"
