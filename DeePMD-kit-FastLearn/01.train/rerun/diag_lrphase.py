"""Trace the learning-rate schedule against the LES latent charges and gradients.

Why
---
`LearningRateExp` (deepmd 3.1.2, `dpmodel/utils/learning_rate.py`) is a
*piecewise constant* schedule, not a ramp:

    decay_rate = exp( log(stop_lr / start_lr) / (stop_steps / decay_steps) )
    lr(step)   = max(start_lr * decay_rate ** (step // decay_steps), stop_lr)

so with start_lr 1e-3, stop_lr 3.51e-8, stop_steps 10000, decay_steps 5000 the
learning rate is 1e-3 for steps 0-4999 and then 5.921e-6 for the rest of the
run: one cliff, at half the training. Whether the latent charges `q` are
*underfitted* - frozen by that cliff rather than by having converged - is
decided by looking at what `q` and the charge-network gradient do across it.

The two channels are logged by different code and neither records the training
step directly:

* `les.log` (`Les._log_verbose_stats`) prints at forward call 1 and every
  `log_freq` forward calls. Validation runs 3 extra forwards per display step,
  so its counter runs ahead of the training step by 3 per validation, up to 300
  at the end of a 10k run. The label cannot be used as the step.
* `train.log` (`HybridLESModel._register_les_grad_hooks`) prints the backward
  gradient norm of each LES parameter. Backward runs once per training step and
  never during validation, so call n *is* training step n - but the counter, not
  the step, is all that is printed.

Both are recovered here and, more importantly, *checked* against the log's own
line ordering (see `_forward_labels` and `_check_labels`), so a mapping that
drifted would fail loudly instead of silently mislabelling a curve.

A third channel, `train.log`'s `[HybridLES]` lines, carries the same run's
SR/LR decomposition of the LES output term by term (see `read_srlr`).

Usage:
    python diag_lrphase.py [run_dir ...]        # default: both schedules, 6+6 arms
Writes: lrphase_q.tsv, lrphase_grad.tsv, lrphase_srlr.tsv
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

from deepmd.common import j_loader
from deepmd.dpmodel.utils.learning_rate import LearningRateExp

from runname import parse_run

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "extended")

# The six arms of one learning-rate schedule, in the order the runner uses.
ARMS = [
    "run_ordinary_sA",
    "run_hybrid_q_sA",
    "run_hybrid_fixed_sA",
    "run_ordinary_sB",
    "run_hybrid_q_sB",
    "run_hybrid_fixed_sB",
]
LES_ARMS = [a for a in ARMS if "ordinary" not in a]


def default_runs():
    """Every arm of every schedule present on disk, default schedule first."""
    runs = []
    for decay in (None, 7500):
        suffix = "" if decay is None else f"_d{decay}"
        runs += [a + suffix for a in ARMS]
    return [r for r in runs if os.path.isdir(os.path.join(EXT, r))]


def lr_at(rundir):
    """The exact learning-rate schedule the run was trained with.

    Built from the run's own input.yaml through deepmd's own class, so the
    figure shows the schedule that ran rather than a re-derivation of it.
    """
    jdata = j_loader(os.path.join(rundir, "input.yaml"))
    lr = jdata["learning_rate"]
    steps = int(jdata["training"]["numb_steps"])
    sched = LearningRateExp(
        start_lr=float(lr["start_lr"]),
        stop_lr=float(lr["stop_lr"]),
        decay_steps=int(lr["decay_steps"]),
        stop_steps=steps,
        decay_rate=lr.get("decay_rate"),
    )
    return sched, steps


def _forward_labels(rundir, steps):
    """Map each `les.log` print label to the training step it was emitted at.

    Replays the training loop's forward-call order: one forward per training
    step, plus `numb_btch` validation forwards at each display step, all of them
    hitting `Les.forward`. Prints happen at forward 1 and every `log_freq`.

    Returns (labels, steps_for_labels, is_validation) and is verified against
    the log's line ordering by `_check_labels`.
    """
    jdata = j_loader(os.path.join(rundir, "input.yaml"))
    train = jdata["training"]
    disp = int(train.get("disp_freq", 100))
    nbtch = int(train["validation_data"].get("numb_btch", 1))
    log_freq = int(jdata["model"].get("les_params", {}).get("log_freq", 100))

    display_steps = set(range(disp, steps + 1, disp)) | {1}
    counter, labels, lab_steps, lab_kind = 0, [], [], []
    for s in range(1, steps + 1):
        counter += 1  # training forward
        if counter % log_freq == 0 or counter == 1:
            labels.append(counter)
            lab_steps.append(s)
            lab_kind.append("train")
        if s in display_steps:
            for _ in range(nbtch):  # validation forwards
                counter += 1
                if counter % log_freq == 0 or counter == 1:
                    labels.append(counter)
                    lab_steps.append(s)
                    lab_kind.append("valid")
    return labels, lab_steps, lab_kind, display_steps


def _check_labels(rundir, labels, lab_steps, display_steps, log_freq):
    """Check the replayed mapping against the order the log actually wrote.

    The replay is only worth using if it agrees with the file. Two independent
    checks, both read off the log itself:

    1. the number of `Training steps :` blocks equals the replayed print count;
    2. each block sits inside the display-step interval it was replayed into,
       i.e. it appears after the `batch w` line for the largest display step
       `w` below its step and before the line for the next display step.

    Check 1 is exact; check 2 is only as tight as `disp_freq`, so it bounds a
    drift rather than eliminating it. Between them a mapping that was wrong in
    its mechanism (a missing set of validation forwards, a different display
    cadence) fails loudly instead of silently mislabelling a curve.
    """
    log = os.path.join(rundir, "les.log")
    seen, last_wall = [], None
    for line in open(log):
        m = re.search(r"Training steps :(\d+)", line)
        if m:
            seen.append((int(m.group(1)), last_wall))
            continue
        # `batch N: total wall time` is written once per display step, after
        # that step's validation forwards.
        m = re.search(r"batch\s+(\d+): total wall time", line)
        if m:
            last_wall = int(m.group(1))
    if len(seen) != len(labels):
        raise AssertionError(
            f"{os.path.basename(rundir)}: les.log has {len(seen)} log blocks, "
            f"replay predicts {len(labels)}"
        )
    ordered_walls = sorted(display_steps)
    for (label, wall_before), step in zip(seen, lab_steps):
        nxt = min((w for w in ordered_walls if w >= step), default=None)
        if wall_before is not None and wall_before >= step:
            raise AssertionError(
                f"{os.path.basename(rundir)}: label {label} replayed to step "
                f"{step} but follows the display line for step {wall_before}"
            )
        if nxt is not None and wall_before is not None and wall_before >= nxt:
            raise AssertionError(
                f"{os.path.basename(rundir)}: label {label} replayed to step "
                f"{step} sits after display line {wall_before}"
            )


_GRAD_RE = re.compile(r"\[LES-grad\] (\S+): \|grad\|=([\-\d.eE+]+)")
# `HybridLESModel` decomposes the batch's first frame into its two energy terms
# and the two force terms once per forward, next to the same forward counter the
# `les.log` labels use. Only frame 0 is logged, so the ratio is a frame-0 ratio
# and not a batch mean - the RNG decides which frame that is.
_HYB_E = re.compile(
    r"\[HybridLES\] step=(\d+) \| "
    r"E_SR\(frm0\)=([\-\d.eE+]+) E_LR\(frm0\)=([\-\d.eE+]+) \| "
    r"mean\|E_LR/E_SR\|\(batch\)=([\-\d.eE+]+) \| "
    r"E_LR range=\[([\-\d.eE+]+), ([\-\d.eE+]+)\]"
)
_HYB_F = re.compile(
    r"\[HybridLES\] step=(\d+) \| "
    r"RMS F_SR=([\-\d.eE+]+) RMS F_LR=([\-\d.eE+]+) \|"
    r"F_LR\|/\|F_SR\|\(mean\)=([\-\d.eE+]+)"
)
_LESRE = {
    "q_mean": re.compile(r"latent_charges mean: ([\-\d.eE+]+), std: ([\-\d.eE+]+)"),
    "q_el": re.compile(r"les\.les:(\d+)\ttensor\(\[([\-\d.eE+]+)\]"),
    "E_lr": re.compile(r"E_lr = ([\-\d.eE+]+)"),
    "w": re.compile(r"les\.les:\s+(\S+): \|w\|=([\-\d.eE+]+)"),
}


def read_q(rundir, labels, lab_steps):
    """Per-log latent-charge and charge-network state, indexed by training step."""
    rows, cur = [], None
    for line in open(os.path.join(rundir, "les.log")):
        if "Training steps :" in line:
            if cur:
                rows.append(cur)
            cur = {}
            continue
        if cur is None:
            continue
        m = _LESRE["q_mean"].search(line)
        if m:
            cur["q_mean"], cur["q_std"] = float(m.group(1)), float(m.group(2))
        m = _LESRE["q_el"].search(line)
        if m:
            cur[f"q_{m.group(1)}"] = float(m.group(2))
        m = _LESRE["E_lr"].search(line)
        if m:
            cur["E_lr"] = float(m.group(1))
        m = _LESRE["w"].search(line)
        if m:
            cur.setdefault("w", {})[m.group(1)] = float(m.group(2))
    if cur:
        rows.append(cur)
    if len(rows) != len(labels):
        raise AssertionError(f"{rundir}: {len(rows)} q blocks vs {len(labels)} labels")
    for r, step in zip(rows, lab_steps):
        r["step"] = step
    return pd.DataFrame(rows)


def read_grad(rundir, log_freq):
    """Per-parameter LES gradient norm, indexed by training step.

    The hook fires once per backward, which is once per training step (nothing
    backprops during validation), and prints at call 1 and every `log_freq`, so
    print j is training step 1 + log_freq * j. Verified against the surrounding
    `batch` lines before use.
    """
    raw = []
    for line in open(os.path.join(rundir, "train.log")):
        m = _GRAD_RE.search(line)
        if m:
            raw.append((m.group(1), float(m.group(2))))
    if not raw:
        return pd.DataFrame()
    names = list(dict.fromkeys(n for n, _ in raw))
    nprint = len(raw) // len(names)
    if len(raw) != nprint * len(names):
        raise AssertionError(f"{rundir}: ragged LES-grad blocks")
    df = pd.DataFrame(raw, columns=["param", "grad"])
    df["i"] = np.arange(len(df)) // len(names)
    df["step"] = 1 + log_freq * df.i
    return df[["step", "param", "grad"]]


def read_srlr(rundir, labels, lab_steps):
    """The SR/LR split of the LES output, indexed by training step.

    Both lines are emitted by `HybridLESModel` once per forward, keyed on the
    same forward counter as `les.log`. Two lines per print, so the block count
    is checked against the replayed labels exactly as `read_q` does. The
    ordinary arm has no such lines at all, which is why it is skipped in `main`
    before this is reached.
    """
    rows, cur = [], None
    for line in open(os.path.join(rundir, "train.log")):
        m = _HYB_E.search(line)
        if m:
            if cur:
                rows.append(cur)
            cur = {
                "E_SR": float(m.group(2)),
                "E_LR": float(m.group(3)),
                "E_LR_over_E_SR_batch": float(m.group(4)),
                "E_LR_min": float(m.group(5)),
                "E_LR_max": float(m.group(6)),
            }
            continue
        m = _HYB_F.search(line)
        if m and cur is not None:
            cur["RMS_F_SR"] = float(m.group(2))
            cur["RMS_F_LR"] = float(m.group(3))
            cur["F_LR_over_F_SR"] = float(m.group(4))
    if cur:
        rows.append(cur)
    if len(rows) != len(labels):
        raise AssertionError(f"{rundir}: {len(rows)} SR/LR blocks vs {len(labels)} labels")
    for r, step in zip(rows, lab_steps):
        r["step"] = step
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*", default=None)
    args = ap.parse_args()
    runs = args.runs or default_runs()

    q_rows, g_rows, s_rows = [], [], []
    for name in runs:
        rundir = os.path.join(EXT, name)
        meta = parse_run(name)
        jdata = j_loader(os.path.join(rundir, "input.yaml"))
        mtype = jdata["model"]["type"]
        log_freq = int(jdata["model"].get("les_params", {}).get("log_freq", 100))
        sched, steps = lr_at(rundir)

        print(f"{name:28s} d{meta['decay']:<5d} steps={steps} "
              f"lr(0)={sched.value(0):.3e} lr(last)={sched.value(steps - 1):.3e} "
              f"cliff={int(jdata['learning_rate']['decay_steps'])}")
        # The ordinary arm has no LES channel and writes an empty les.log; it is
        # carried through the comparison as the schedule-only control, so it has
        # a learning curve but no charges and no [LES-grad] lines.
        if mtype != "hybrid_ener":
            print("    (no LES channel: control arm for the schedule)")
            continue

        labels, lab_steps, lab_kind, display_steps = _forward_labels(rundir, steps)
        _check_labels(rundir, labels, lab_steps, display_steps, log_freq)

        q = read_q(rundir, labels, lab_steps)
        q["run"], q["model"], q["rep"], q["decay"] = (
            name, meta["model"], meta["rep"], meta["decay"])
        q["lr"] = [sched.value(s) for s in q.step]
        q_rows.append(q)

        g = read_grad(rundir, log_freq)
        g["run"], g["model"], g["rep"], g["decay"] = (
            name, meta["model"], meta["rep"], meta["decay"])
        g["lr"] = [sched.value(s) for s in g.step]
        g_rows.append(g)

        s = read_srlr(rundir, labels, lab_steps)
        s["run"], s["model"], s["rep"], s["decay"] = (
            name, meta["model"], meta["rep"], meta["decay"])
        s["lr"] = [sched.value(st) for st in s.step]
        s_rows.append(s)

        # A one-line summary of the two lr phases, which is the whole point.
        pre = q[q.step <= meta["decay"]]
        post = q[q.step > meta["decay"]]
        for el in ("q_1", "q_8"):
            if el not in q:
                continue
            print(f"    {el}: drift pre-cliff {pre[el].max() - pre[el].min():+.4f}, "
                  f"post-cliff {post[el].max() - post[el].min():+.4f}")
        if len(g):
            tot = g.groupby("step").grad.apply(lambda s: np.sqrt((s ** 2).sum()))
            print(f"    |grad| median pre-cliff {tot[tot.index <= meta['decay']].median():.3e}, "
                  f"post-cliff {tot[tot.index > meta['decay']].median():.3e}")
        # The 20 log entries ending the run: where the SR/LR split and the LR
        # force settled. Both are already tiny, so the absolute numbers matter
        # less than whether the extra full-lr steps moved them.
        late = s.tail(20)
        print(f"    last 20: |E_LR|={late.E_LR.abs().mean():.4e} "
              f"|E_LR/E_SR|={late.E_LR_over_E_SR_batch.mean():.2e} "
              f"RMS F_LR={late.RMS_F_LR.mean():.4e} "
              f"|F_LR|/|F_SR|={late.F_LR_over_F_SR.mean():.4e}")

    if q_rows:
        cols = ["run", "model", "rep", "decay", "step", "lr", "q_mean", "q_std",
                "q_1", "q_8", "E_lr"]
        qall = pd.concat(q_rows)
        qall = qall[[c for c in cols if c in qall]].sort_values(
            ["decay", "run", "step"])
        qall.to_csv(os.path.join(HERE, "lrphase_q.tsv"), sep="\t", index=False)
        print(f"wrote lrphase_q.tsv  ({len(qall)} rows)")
    if g_rows:
        gall = pd.concat(g_rows).sort_values(["decay", "run", "step", "param"])
        gall.to_csv(os.path.join(HERE, "lrphase_grad.tsv"), sep="\t", index=False)
        print(f"wrote lrphase_grad.tsv ({len(gall)} rows)")
    if s_rows:
        cols = ["run", "model", "rep", "decay", "step", "lr", "E_SR", "E_LR",
                "E_LR_over_E_SR_batch", "E_LR_min", "E_LR_max", "RMS_F_SR",
                "RMS_F_LR", "F_LR_over_F_SR"]
        sall = pd.concat(s_rows)[cols].sort_values(["decay", "run", "step"])
        sall.to_csv(os.path.join(HERE, "lrphase_srlr.tsv"), sep="\t", index=False)
        print(f"wrote lrphase_srlr.tsv ({len(sall)} rows)")


if __name__ == "__main__":
    main()
