#!/bin/bash
# The water-interface 8-arm cost campaign, arranged for one RTX 4090 (24 GiB).
#
# The arms
# --------
#   family  arm         replicate  training steps  precision
#   cace    cace-lr     A B          225,000       float32
#   cace    cace-sr     A B          180,000       float32
#   deepmd  deepmd-les  A B          225,000       float32
#   deepmd  deepmd      A B          180,000       float32
#   -> 4 arms x 2 replicates = 8 runs.  deepmd-les is the cace-lr counterpart and
#      deepmd is the cace-sr counterpart: the same step budget, so a lane pair is
#      always two arms that are meant to be compared with each other.
#
# Both families run float32. That is a deliberate departure from deepmd's
# as-shipped float64 and it is the point of `--precision` in run_chain.py: cace is
# fp32 throughout, and on Ada fp64 runs at 1/64 of the fp32 rate, so an fp64
# deepmd arm against an fp32 cace arm would measure a precision default rather
# than the algorithm. The switch is DP_INTERFACE_PREC on the child process; it is
# not an input.yaml key and a bare `dp --pt train` would silently run float64.
#
# The fairness rule
# -----------------
# Contention is only shared evenly while every concurrent lane wants the GPU for
# the same length of time. A lane that finishes early hands its share to the
# others and speeds them up mid-run, which makes their cost depend on their
# neighbours rather than on themselves. So:
#
#   1. a cohort is a set of lanes with an IDENTICAL step budget, so no lane ever
#      finishes early and the contention is constant for the whole cohort;
#   2. within a cohort the two cace replicates run together and every deepmd lane
#      runs alone. The cace replicates are the same arm at the same budget, so
#      their contention is symmetric and neither can finish early; a wave that
#      mixed the families could not promise that. cace-lr + deepmd-les would fit
#      on paper (4694 + 15734 = 20428 MiB against the 20879 MiB budget) and is
#      exactly what this rule forbids: the deepmd lane's memory is bimodal, so
#      its own peak would decide whether its cace neighbour was measured
#      contended or not - the confound this rule exists to prevent;
#   3. stage 0 runs each arm ALONE for a short bounded probe, giving the
#      uncontended ms/step. Dividing a wave's ms/step by that figure is the
#      contention factor, so contended and uncontended costs are both on record;
#   4. stage 0 also reports torch's peak memory per arm, which is what sets the
#      wave size. It is not a guess: the script refuses to start a wave whose
#      arms' measured peaks do not fit, rather than discovering it as an OOM
#      three hours in.
#
# Wave sizes, and the memory that sets them. Stage 0 on this pod (RTX 4090,
# 24564 MiB), each arm alone:
#
#   arm         peak MiB                       arm          peak MiB
#   cace-lr      4694  (both replicates)       deepmd-les   24090 / 23690
#   cace-sr      2392  (both replicates)       deepmd       23650 / 23730
#
# The cace figures are exact and reproducible: both replicates land on the same
# megabyte every time, so the two cace replicates pair (9388 and 4784 MiB) well
# inside the 20879 MiB budget. The deepmd figures are not reproducible. A clean
# isolated run plateaus at ~15.7 GB and holds it flat, and the LES arm sits only
# 114 MiB above the plain one - the long-range term is nearly free in memory and
# deepmd's own machinery is the whole cost. But the same arm inside a scripted
# sweep measures 23110-24090 MiB, and which of the two it is does not follow from
# the config: replicate B measured 15734 once and 23890 another time in the same
# position in the same sweep. At either end 2 x 15.7 = 31.4 GB does not fit the
# card, so the deepmd lanes are solo unconditionally - the bimodality decides
# nothing about the arrangement, only how much risk a deepmd block carries.
#
# The bimodality is the neighbour-statistics auto batch-size tuner
# (deepmd/utils/batch_size.py:104). It grows the batch until an attempt OOMs and then
# backs off, so its largest allocation is a deliberate OOM probe whose size depends
# on how much of the card happened to be free at that moment; under
# expandable_segments the reservation it leaves behind is what nvidia-smi reports.
# The tuner's log trajectory was identical (1024 -> 32768 -> 16384) in both modes, so
# the difference is in what the probe was allowed to reach, not in the schedule.
#
# Honesty about the cace pairing. Running an arm's two cace replicates together
# measures them contended while every deepmd lane is measured solo, so the
# cross-family cost figure carries a contention term on the cace side only. That
# is a deliberate trade of comparison cleanliness for wall time - it is worth
# about 20 of the ~73 hours - and it is defensible only because the uncontended
# cace numbers are already on record: stage 0 probes all 8 lanes alone, and
# cace-lr_sA completed a full 8-block schedule alone at 319-338 ms/batch before
# this arrangement existed. So the contention factor is a measurement to be read
# off those two numbers, not an assumption. What it is NOT is a claim that the
# cace lanes ran uncontended: quote the cross-family figure with the pairing
# stated, or use the stage 0 solo probes, which contend with nothing.
#
# `--calibrate` is idempotent - it probes only the lanes with no peak on record - so
# it can be re-run to fill in a lane whose smoke directory already held a checkpoint
# (the runner calls that "already complete" and it never reaches the GPU) without
# paying for the other seven.
#
# Why the run is ordered LR-then-SR rather than arm-by-arm: the LR cohort's
# 225,000-step budget and the SR cohort's 180,000 are both much longer than the
# gap between them, so the two cohorts cannot overlap anyway, and sequencing them
# keeps the two cohorts' conditions identical instead of drifting.
#
# Why stage 0 matters more for cace than for deepmd: cace cannot resume.
# `TrainingTask.fit` only saves, nothing loads a checkpoint back
# (cace/tasks/train.py:261, :284), so an arm killed at epoch 480 of 500 restarts
# from epoch 0. A conservative N that wastes some GPU time is cheaper than one
# crash that costs a whole arm.
#
# Usage
# -----
#   ./run_campaign.sh --calibrate          # stage 0 only; prints the peaks
#   ./run_campaign.sh                      # both cohorts, waves as declared below
#   ./run_campaign.sh --stage lr           # one cohort
#   ./run_campaign.sh --dry-run
#
# Re-running IS the resume command. A cace lane whose blocks.json says `finished`
# is skipped, and a deepmd lane resumes per block from its last checkpoint, so the
# same invocation after an interruption continues the campaign instead of
# restarting it.
#
# Environment assumed on the pod: the LES deepmd-kit editable install is importable
# by `dp`, cace is importable by the same interpreter, both campaign runners are in
# this tree, and the data/ trees have been rsynced across.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN="$(dirname "$HERE")"
DEEPMD="$CAMPAIGN/deepmd"
CACE="$CAMPAIGN/cace"

# On the pod the venv is off a non-interactive PATH, so default to it when it is
# there and fall back to whatever the PATH has (which is what the local smoke uses).
if [[ -x /root/venv310/bin/dp ]]; then
  DP_BIN="${DP_BIN:-/root/venv310/bin/dp}"
  PY_BIN="${PY_BIN:-/root/venv310/bin/python}"
else
  DP_BIN="${DP_BIN:-dp}"
  PY_BIN="${PY_BIN:-python}"
fi
PEAKS="${PEAKS:-$HERE/peaks}"          # stage 0's measured peaks, one file per lane
STAGE=all
CAL_ONLY=0
DRY=0
PROBE_EPOCHS="${PROBE_EPOCHS:-1}"      # stage 0's bounded probe length
VCPUS="${VCPUS:-14}"                   # pod: 14 vCPU

# A cohort is a list of WAVES. Every lane in a wave runs at the same time; waves
# run one after another, and a wave starts only when the previous one has fully
# finished. Cohort member: "family:arm:rep". Membership is declared rather than
# derived from a concurrency number, because which lanes may share the card is a
# per-lane fact about measured peaks (rule 2), not a policy knob.
WAVES_LR=(
  "cace:cace-lr:A cace:cace-lr:B"
  "deepmd:deepmd-les:A"
  "deepmd:deepmd-les:B"
)
WAVES_SR=(
  "cace:cace-sr:A cace:cace-sr:B"
  "deepmd:deepmd:A"
  "deepmd:deepmd:B"
)
ALL_LANES=()
for _wave in "${WAVES_LR[@]}" "${WAVES_SR[@]}"; do
  for _lane in $_wave; do ALL_LANES+=("$_lane"); done
done

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)     STAGE="$2"; shift 2 ;;
    --calibrate) CAL_ONLY=1; shift ;;
    --dry-run)   DRY=1; shift ;;
    --dp)        DP_BIN="$2"; shift 2 ;;
    --python)    PY_BIN="$2"; shift 2 ;;
    -h|--help)   sed -n '2,116p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# One heavy process per lane, and as many threads per lane as the box can afford.
# deepmd's own thread pool reads its own two variables, not OMP_NUM_THREADS, so
# all three are set: leaving the intra-op pool at its default would have every
# lane ask for every core and the waves would thrash instead of share.
#
# THREADS is deliberately NOT divided by the wave size. Both cace lanes in a wave
# get the same 14 threads a solo deepmd lane gets, so the only difference between
# the two families' conditions is the GPU contention that rule 2 makes symmetric
# within a family. Halving the CPU for the cace pair would have stacked a second,
# one-sided penalty on top of it and made the cross-family figure unreadable.
THREADS="${THREADS:-$VCPUS}"
export OMP_NUM_THREADS="$THREADS"
export DP_INTRA_OP_PARALLELISM_THREADS="$THREADS"
export DP_INTER_OP_PARALLELISM_THREADS=1
export MKL_NUM_THREADS="$THREADS"
# WSL2 deadlocks under the default caching allocator; harmless and correct on a
# native pod, and set here so the two hosts run the same allocator.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say()  { printf '\n=== %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }

gpu_state() {
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
}

# Run "$@" while sampling GPU memory, then report the peak.
#
# The peak is the number that decides the cohort's lane count, and it is knowable
# only here: deepmd reports its wall clock but not its memory, and a lane that is
# merely close to the ceiling is what turns a 3-hour cohort into an OOM at hour
# two. nvidia-smi rather than torch.cuda.max_memory_allocated because the
# interesting figure includes the caching allocator's reserved-but-unfreed blocks,
# which is what actually competes for the 24 GiB.
peak_run() {
  local label="$1"; shift
  local log; log="$(mktemp)"
  ( while :; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null
      sleep 0.5
    done ) >"$log" &
  local sampler=$!
  "$@"
  local rc=$?
  kill "$sampler" 2>/dev/null
  wait "$sampler" 2>/dev/null
  local pk; pk="$(sort -n "$log" | tail -1)"
  rm -f "$log"
  if [[ -n "$pk" ]]; then
    mkdir -p "$PEAKS" && printf '%s\n' "$pk" >"$PEAKS/$label.txt"
    note "peak GPU memory: ${pk} MiB  (recorded in peaks/$label.txt)"
  else
    note "peak GPU memory: unknown"
  fi
  return $rc
}

# Refuse a wave the peaks say will not fit. Headroom is not decoration: the
# allocator's reserved-but-unfree blocks are not released between steps, and a
# wave that fits at step 1 of 225,000 can OOM at step 200,000. cace cannot resume,
# so that OOM costs the arm.
#
# Headroom is what a wave can be refused for, but it is not something a single lane
# can be refused for. deepmd-les measures 24090 MiB against a 20879 MiB headroom
# budget: there is no second lane to drop and no smaller configuration of the arm,
# so refusing it would refuse the campaign. A lone lane is therefore judged against
# the card itself, with the risk stated rather than hidden. All four deepmd lanes
# have now been measured solo and land in 23650-24090 MiB, i.e. the top of the range
# is 98% of the card, and an OOM late in a block is a real possibility. That is why
# the deepmd arms keep deepmd's own per-block checkpointing - a block that dies can
# be re-run from its predecessor, where a cace arm would be lost outright.
MAX_MEM_MIB="${MAX_MEM_MIB:-24564}"
HEADROOM="${HEADROOM:-0.15}"
fits() {
  local want=0 lane label pk
  for lane in "$@"; do
    IFS=: read -r _ arm rep <<<"$lane"
    # Braces are required: `$arm_s` is read as the variable named `arm_s`, and
    # `set -u` makes that a fatal unbound variable rather than an empty string.
    label="${arm}_s${rep}"
    pk="$(cat "$PEAKS/$label.txt" 2>/dev/null || echo '')"
    if [[ -z "$pk" ]]; then
      note "no measured peak for $label - run --calibrate first"; return 1
    fi
    want=$(( want + pk ))
  done
  local budget; budget="$(awk -v m="$MAX_MEM_MIB" -v h="$HEADROOM" 'BEGIN{printf "%d", m*(1-h)}')"
  note "wave needs ~${want} MiB against a ${budget} MiB budget (${MAX_MEM_MIB} MiB less ${HEADROOM} headroom)"
  (( want <= budget )) && return 0
  if (( $# == 1 )) && (( want <= MAX_MEM_MIB )); then
    note "  ^ one lane, and it fits the ${MAX_MEM_MIB} MiB card, so it runs: there is no"
    note "    second lane to drop. It peaks at $(( want * 100 / MAX_MEM_MIB ))% of the card."
    return 0
  fi
  return 1
}

# ---------------------------------------------------------------------------
# stage 0 - uncontended cost and peak memory, one arm at a time
# ---------------------------------------------------------------------------
# `--smoke` on both runners is the bounded probe: one short block per arm, with
# the schedule deliberately not the cace schedule. It is used here only to make
# the two families' per-step times comparable, which needs identical CONTENTION,
# not identical length. Prerequisite, once, before the first run:
#   (cd deepmd && python gen_inputs.py --smoke --out-root runs_smoke)
# The smoke configs carry the same descriptor, the same batch_size and the same
# data_stat_nbatch as production, so per-step cost transfers; only numb_steps and
# the frequencies differ.
#
# Read `reported_s_per_batch` for the cross-arm figure where the runner has one:
# that is deepmd's own "average training time", which excludes validation, so the
# comparison is not diluted by the smoke and production configs validating at
# different rates (numb_btch 2 against 50). The wall-clock s_per_batch includes
# validation and is the right figure for the cohort stages, where it is production
# throughout. cace's runner has only the wall-clock figure.
calibrate() {
  say "stage 0: uncontended probes, one lane at a time, before: $(gpu_state)"
  local lane family arm rep
  for lane in "${ALL_LANES[@]}"; do
    IFS=: read -r family arm rep <<<"$lane"
    # Idempotent: a lane that already has a peak on record is not re-probed, so
    # --calibrate can be re-run to fill in a lane that was skipped (a stale
    # checkpoint in its smoke directory makes the runner call it "already
    # complete" and it never reaches the GPU) without paying for the other seven.
    if [[ -s "$PEAKS/${arm}_s${rep}.txt" ]]; then
      note "--- $family $arm $rep: peak already measured ($(cat "$PEAKS/${arm}_s${rep}.txt") MiB), skipping"
      continue
    fi
    note "--- $family $arm $rep"
    if [[ $family == deepmd ]]; then
      if [[ $DRY == 1 ]]; then
        note "$DP_BIN (deepmd) --arm $arm --rep $rep --root runs_smoke"
      else
        peak_run "${arm}_s${rep}" bash -c "cd '$DEEPMD' && '$PY_BIN' run_chain.py --root runs_smoke \
            --arm '$arm' --rep '$rep' --blocks 1 --dp '$DP_BIN'"
      fi
    else
      if [[ $DRY == 1 ]]; then
        note "$PY_BIN (cace) --arm $arm --rep $rep --smoke --blocks 1"
      else
        peak_run "${arm}_s${rep}" bash -c "cd '$CACE' && CACE_PROBE_EPOCHS='$PROBE_EPOCHS' \
            '$PY_BIN' run_cace_chain.py --arm '$arm' --rep '$rep' --smoke --blocks 1"
      fi
    fi
  done
  say "stage 0 done, after: $(gpu_state)"
  note "each arm's timing.json holds the uncontended ms/batch the cohort stages are"
  note "divided by; peaks/ holds the memory the waves in WAVES_LR/WAVES_SR are"
  note "checked against. Both are inputs to the arrangement, so re-read them before"
  note "changing a wave."
}

# ---------------------------------------------------------------------------
# stages 1-2 - the two cohorts
# ---------------------------------------------------------------------------
one_lane() {
  local family="$1" arm="$2" rep="$3"
  if [[ $family == deepmd ]]; then
    ( cd "$DEEPMD" && "$PY_BIN" run_chain.py --arm "$arm" --rep "$rep" --dp "$DP_BIN" )
  else
    ( cd "$CACE" && "$PY_BIN" run_cace_chain.py --arm "$arm" --rep "$rep" )
  fi
}

# Run a cohort's waves in order. Every lane in a wave starts together, and the next
# wave starts only when the previous one is entirely finished - that is the point,
# since a half-finished wave is exactly the early exit that would make a lane's
# timing depend on its neighbours. Membership comes from WAVES_LR/WAVES_SR rather
# than from a concurrency number: the peaks say which lanes may share the card, and
# a wave is a whitespace-separated list of lanes, so it is split deliberately.
cohort() {
  local label="$1"; shift
  local waves=("$@")
  local total=0 w=0 wave_str lane family arm rep
  for wave_str in "${waves[@]}"; do
    for lane in $wave_str; do total=$(( total + 1 )); done
  done
  say "cohort $label: $total lanes in ${#waves[@]} waves, budget per lane is identical"
  note "threads per lane: $THREADS;  GPU before: $(gpu_state)"
  for wave_str in "${waves[@]}"; do
    w=$(( w + 1 ))
    local wave=($wave_str)
    local pids=()
    note "wave $w/${#waves[@]} (${#wave[@]} lane(s)): ${wave[*]}"
    # fits() is consulted even under --dry-run, and a dry refusal is reported
    # rather than acted on: this is the only code that reads the peak files, so a
    # dry-run that skipped it would print a plan it had not actually checked.
    if ! fits "${wave[@]}"; then
      if [[ $DRY == 1 ]]; then
        note "  ^ dry-run: this wave WOULD BE REFUSED (see above)"
      else
        say "REFUSING wave $w: the measured peaks do not fit."
        note "move a lane out of this wave in WAVES_LR/WAVES_SR, or raise"
        note "MAX_MEM_MIB if the reading was contaminated by another tenant."
        return 1
      fi
    fi
    for lane in "${wave[@]}"; do
      IFS=: read -r family arm rep <<<"$lane"
      if [[ $DRY == 1 ]]; then
        note "  (dry) $family $arm $rep"
      else
        one_lane "$family" "$arm" "$rep" &
        pids+=("$!")
      fi
    done
    for pid in "${pids[@]:-}"; do
      [[ -n "$pid" ]] && wait "$pid"
    done
    note "wave done, GPU: $(gpu_state)"
  done
}

# Pull the results home. cace cannot resume, so an arm's weights are the whole
# result and are worth moving off the pod the moment its cohort ends; the pod
# bills by the hour and can stop on its own.
collect() {
  say "results left in $DEEPMD/runs and $CACE/runs (timing.json per arm)"
  note "rsync both runs/ trees plus the lcurve.out/les.log/BLOCKS logs home next;"
  note "a cace arm cannot be restarted from a partial state, so do not let the"
  note "pod be reclaimed before they have been copied."
}

if [[ $CAL_ONLY == 1 ]]; then calibrate; collect; exit 0; fi
case "$STAGE" in
  all) calibrate; cohort LR "${WAVES_LR[@]}"; cohort SR "${WAVES_SR[@]}" ;;
  lr)  cohort LR "${WAVES_LR[@]}" ;;
  sr)  cohort SR "${WAVES_SR[@]}" ;;
  *)   echo "--stage must be lr, sr or all" >&2; exit 2 ;;
esac
collect