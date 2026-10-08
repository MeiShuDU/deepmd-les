#!/bin/bash
# Pull the deepmd-les-cace-sched arm home from the pod while it trains.
#
# The pod is Postpay and has stopped on its own before, so the arm's eight block
# directories are mirrored home every couple of minutes rather than once at the end.
# Checkpoints are small here (the arm's model is 169,692 parameters), so a full
# incremental rsync each pass costs almost nothing.
#
# Exits when the runner is gone (chain finished, or it died) - after one final sync.
# Also exits if the pod stops answering for POD_GONE passes in a row.
#
# SLEEP / MAX_PASSES / POD_GONE can be overridden from the environment for a test run.
set -uo pipefail

SLEEP=${SLEEP:-120}
MAX_PASSES=${MAX_PASSES:-400}   # ~13 h
POD_GONE=${POD_GONE:-10}        # 20 min with no contact

SSH_OPTS=(-o ControlPath=/tmp/hpc-%r-%h-%p -o ConnectTimeout=20)
REMOTE_ROOT=/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_water_interface/deepmd/runs
LOCAL_ROOT=/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_water_interface/deepmd/runs
RUNNER_LOG=/root/les-cace-sched-runner.log

pod() { ssh "${SSH_OPTS[@]}" hpc "$@"; }

sync_rep() {
    rsync -a --partial --timeout=120 -e "ssh ${SSH_OPTS[*]}" \
        "hpc:$REMOTE_ROOT/deepmd-les-cace-sched_s$1/" \
        "$LOCAL_ROOT/deepmd-les-cace-sched_s$1/"
}

sync_all() {
    # replicate B does not exist until A's eight blocks are done (the runner goes
    # replicate-major), so "absent remotely" is normal and must not read as a failure.
    # The final sync passes force=1 and tries both regardless, so a transient blip on
    # the existence probe cannot leave a finished replicate un-pulled.
    local force=${1:-0}
    for rep in A B; do
        if [ "$force" != "1" ] &&
           ! pod "test -d $REMOTE_ROOT/deepmd-les-cace-sched_s$rep" 2>/dev/null; then
            continue
        fi
        sync_rep "$rep" >/dev/null 2>&1 || echo "  rsync of s$rep failed this pass"
    done
}

gone=0        # consecutive passes with no contact at all
stopped=0     # consecutive passes confirming the runner is gone

for pass in $(seq 1 "$MAX_PASSES"); do
    if ! pod true 2>/dev/null; then
        gone=$((gone + 1))
        echo "pass $pass: pod unreachable ($gone/$POD_GONE)"
        if [ "$gone" -ge "$POD_GONE" ]; then
            echo "POD UNREACHABLE for $POD_GONE passes - stopping the watcher"
            exit 2
        fi
        sleep "$SLEEP"
        continue
    fi
    gone=0

    sync_all

    # anchored, so the pattern does not also match this very ssh command's shell
    # (an unanchored `pgrep -f run_chain.py` counts itself, and then the watcher
    # would never see the runner go away)
    alive=$(pod "pgrep -f '^/root/venv310/bin/python run_chain.py' | wc -l" 2>/dev/null)
    # "  s3: 1696.2 s for 18000 steps" is a FINISHED block; a bare "  s3: <cmd>" is a
    # started one. Counting the finished form is the only progress signal that moves
    # before the whole replicate is done (the " -> " summary line is per-replicate).
    blocks=$(pod "grep -cE '^  s[0-9]+: [0-9.]+ s for' $RUNNER_LOG" 2>/dev/null)
    last=$(pod "grep -E '^  s[0-9]+: |^ -> |^### ' $RUNNER_LOG | tail -1" 2>/dev/null)
    echo "pass $pass: runner=$alive blocks_done=$blocks | ${last:-?}"

    # An empty reading means the ssh call failed, NOT that the runner exited; treating
    # it as zero ended the watch early once already. Only an explicit 0 counts, and it
    # has to be confirmed twice, so one bad pass cannot end the watch.
    if ! [[ "$alive" =~ ^[0-9]+$ ]]; then
        echo "pass $pass: no runner reading (ssh failed) - treating as inconclusive"
        stopped=0
        sleep "$SLEEP"
        continue
    fi

    if [ "$alive" = "0" ]; then
        stopped=$((stopped + 1))
        if [ "$stopped" -ge 2 ]; then
            echo "runner is gone (confirmed twice) - doing the final sync"
            sync_all 1
            pod "tail -40 $RUNNER_LOG" 2>/dev/null
            echo "SYNCED - the chain has stopped on the pod"
            exit 0
        fi
        echo "pass $pass: runner reads 0 - re-checking before declaring it stopped"
    else
        stopped=0
    fi

    sleep "$SLEEP"
done

echo "watcher hit its pass limit without the runner stopping"
exit 3
