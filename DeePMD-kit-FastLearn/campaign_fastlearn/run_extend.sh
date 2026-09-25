#!/usr/bin/env bash
# Extend the 8 deepmd campaign arms by more annealing steps, restarting from a
# checkpoint the 80k campaign already wrote.
#
# Why: the 80k campaign's full-split sweep (all 80 validation frames, replicate-
# averaged, not the 3-frame lcurve column) puts the force error converged - the
# 40k to 80k drift is under 0.7% for every arm - while the ENERGY error is not.
# The learned arms still drop 19-21% from 70k to 80k alone, ~3x the 3-6% replicate
# spread, and the series is non-monotone (60-70k is a high outlier), so no 80k
# energy number describes a converged checkpoint. The energy question needs a
# longer anneal, not a lower LR.
#
# The lcurve column cannot support (or refute) this. Its energy value oscillates in
# a ~5e-4 band with no trend past ~20k, and the successive-10k-block ratio scatters
# 0.57-1.46 across the 8 arms (std 0.31) - so a single "window ratio" there is the
# mean of a noisy estimator, not a resolvable decline. Quote the full split, not it.
#
# NOTE the non-obvious consequence of the deepmd LR schedule (see
# deepmd/gen_extend_inputs.py): because `stop_steps` comes from the config's
# numb_steps, a SHORTER extension reheats LESS. To 120k the resume LR is
# 1.07e-6 (30x); to 160k it is 5.92e-6 (169x). Both land on the 3.51e-8 floor
# at their endpoint, so both are comparable to the 80k result.
#
# This is a SEPARATE script from run_campaign.sh on purpose, for three reasons:
#   * run_campaign.sh skips any holder that already holds DONE - every arm
#     holds DONE from the 80k run, so it would launch nothing;
#   * it runs `input.yaml` with no `--restart`;
#   * the arm is not done at 80k, so the markers must differ (DONE_<TO>K).
# Reusing it would have needed a flag that silently changes what "done" means.
#
# Usage:
#   ./run_extend.sh                                  # 80k -> 120k, all 8 arms
#   ./run_extend.sh --from 120000 --to 160000         # stage 2, if 120k warrants it
#   ./run_extend.sh --dry-run
#   ./run_extend.sh --only 'les-.*_sA'
#
# Requires the matching config (input_<TO/1000>k.yaml) in every arm dir - make
# it with `python deepmd/gen_extend_inputs.py --from F --to T`. A job is skipped
# when its directory already holds DONE_<TO>K, so an interrupted batch resumes
# without redoing finished arms. Non-zero exits leave FAILED_<TO>K behind and are
# listed at the end - grep for `!!! FAILED`.
set -u
cd "$(dirname "$0")"
CAMPAIGN=$(pwd)
RUNS="$CAMPAIGN/deepmd/runs"

FROM=80000
TO=120000
JOBS=3
ONLY='.'
DRY=0
# The venv holding dp is not on PATH, not even for a login shell, so resolve it
# explicitly rather than inheriting whatever the caller had. Getting this wrong
# fails all 8 arms in ~20 s with "dp: command not found" (observed 2026-09-21),
# which is why the preflight below now asserts dp is runnable.
if [ -z "${DP:-}" ]; then
  if [ -x /root/venv310/bin/dp ]; then DP=/root/venv310/bin/dp; else DP=dp; fi
fi
ALLOC=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Same reasoning as run_campaign.sh: latency-bound, so threads do not help but
# processes do.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export DP_INTRA_OP_PARALLELISM_THREADS=${DP_INTRA_OP_PARALLELISM_THREADS:-4}
export DP_INTER_OP_PARALLELISM_THREADS=${DP_INTER_OP_PARALLELISM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF="$ALLOC"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) awk 'NR>1 && /^set -u/ {exit} NR>1 {print}' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

K=$((TO / 1000))                 # 120 -> DONE_120K, train_120k.log
CONFIG="input_${K}k.yaml"
MARK="DONE_${K}K"
FAILMARK="FAILED_${K}K"
LOG="train_${K}k.log"
echo "campaign   $CAMPAIGN"
echo "resume     step $FROM   config $CONFIG   only /$ONLY/   jobs $JOBS"
echo "dp         $DP"
echo "gpu        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  alloc=$ALLOC"
echo

# Preflight: nothing is launched until every selected arm has what it needs.
# A missing checkpoint or config is a setup error, not a training failure, and
# discovering it 3 h into a batch wastes the whole batch.
selected=()
bad=0
if ! command -v "$DP" >/dev/null 2>&1; then
  echo "!!! dp not runnable: $DP   (set DP=/root/venv310/bin/dp)" >&2
  bad=1
fi
for rundir in "$RUNS"/*/; do
  [ -d "$rundir" ] || continue
  name=$(basename "${rundir%/}")
  [[ "$name" =~ $ONLY ]] || continue
  ckpt="$rundir/model.ckpt-$FROM.pt"
  if [ ! -f "$ckpt" ]; then
    echo "!!! $name: missing $ckpt" >&2; bad=1; continue
  fi
  if [ ! -f "$rundir/$CONFIG" ]; then
    echo "!!! $name: missing $CONFIG" >&2; bad=1; continue
  fi
  # Clear every selected arm's FAILED marker NOW, not when its turn comes.
  # Otherwise a marker left by a previous attempt survives until that arm starts,
  # and any inspector (a monitor, a human) reads it as a LIVE failure - observed
  # 2026-09-21, when 5 arms still carried exit=127 from a launch that `dp` had
  # already fixed. DONE is deliberately NOT cleared here: it is the resume
  # mechanism, so clearing it would make an interrupted batch redo finished arms.
  # Skipped under --dry-run, which must not mutate state.
  [ "$DRY" = 1 ] || rm -f "$rundir/$FAILMARK"
  selected+=("$name")
done
if [ "$bad" = 1 ]; then
  echo "preflight failed - nothing launched" >&2
  exit 2
fi
if [ "${#selected[@]}" -eq 0 ]; then
  echo "nothing selected" >&2
  exit 2
fi
echo "preflight ok: ${#selected[@]} arm(s)"
echo

pool_count() {
  local alive=0 pid
  for pid in ${POOL[@]+"${POOL[@]}"}; do
    if kill -0 "$pid" 2>/dev/null; then alive=$((alive + 1)); fi
  done
  echo "$alive"
}

POOL=()
queued=0
for name in "${selected[@]}"; do
  rundir="$RUNS/$name"
  if [ -f "$rundir/$MARK" ]; then
    echo "skip   $name ($MARK)"
    continue
  fi
  if [ "$DRY" = 1 ]; then
    echo "would  $name -> $rundir/$CONFIG --restart model.ckpt-$FROM.pt"
    continue
  fi
  rm -f "$rundir/$MARK" "$rundir/$FAILMARK"
  (
    rc=0
    cd "$rundir" || rc=1
    if [ "$rc" -eq 0 ]; then
      # CWD must be the run dir: lcurve.out and les.log go to the CWD, and the
      # relative `systems` paths in the config resolve against it. The restart
      # path is absolute so CWD cannot change which checkpoint is loaded.
      "$DP" --pt train "$CONFIG" --restart "$rundir/model.ckpt-$FROM.pt" \
        --skip-neighbor-stat > "$LOG" 2>&1
      rc=$?
    fi
    if [ "$rc" -eq 0 ]; then
      touch "$rundir/$MARK"
    else
      echo "exit=$rc" > "$rundir/$FAILMARK"
    fi
  ) &
  POOL+=("$!")
  queued=$((queued + 1))
  echo "start  $name"
  while [ "$(pool_count)" -ge "$JOBS" ]; do sleep 3; done
done

if [ "$DRY" = 1 ]; then
  echo
  echo "${#selected[@]} job(s) selected"
  exit 0
fi

echo
echo "waiting for $queued job(s) ..."
wait

echo
echo "=== summary $(date) ==="
done_n=0
failed=""
for rundir in "$RUNS"/*/; do
  [ -d "$rundir" ] || continue
  name=$(basename "${rundir%/}")
  if [ -f "$rundir/$FAILMARK" ]; then
    failed="$failed $name"
  elif [ -f "$rundir/$MARK" ]; then
    done_n=$((done_n + 1))
    last=$(grep -oE 'model\.ckpt-[0-9]+\.pt' "$rundir/checkpoint" 2>/dev/null | tail -1)
    printf '  ok    %-36s %s\n' "$name" "${last:-saved}"
  fi
done
echo "done:   $done_n"
if [ -n "$failed" ]; then
  echo "!!! FAILED:$failed"
  for name in $failed; do
    echo "--- last lines of $name/$LOG"
    tail -n 15 "$RUNS/$name/$LOG" 2>/dev/null
  done
  exit 1
fi
echo "ALL DONE $(date)"
