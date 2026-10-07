"""Compare the deepmd-family arms of the water-interface campaign on ONE protocol.

Why this file exists: the campaign's `valid_metrics.tsv` scores only the four
cace/sea arms (they are scored by cace's own metric code over the whole validation
split). The deepmd-family arms - `deepmd`, `deepmd-les` and now
`deepmd-les-cace-sched` - were compared with a simpler habit (`analysis/mean.py`):
read the final block's `lcurve.out` and average the last 70 logged validation
points. That habit is kept here, but applied to EVERY arm, so the numbers are
like for like, and with a convergence check attached.

The protocol is the same number of steps in every arm, which is what makes it
comparable across two different logging styles. Both sides log every 450 steps
(one epoch) and both validate over the whole 50-frame held-out split:

  * deepmd arms: `disp_freq: 450`, and `validation_data: {batch_size: 1,
    numb_btch: 50}` against a 50-frame system, so each `lcurve.out` row is the
    whole split;
  * cace/sea arms: one `val_e/atom_rmse` / `val_f_rmse` pair per epoch over the
    same split (each metric line is echoed twice in the log, bare and behind an
    `INFO:` timestamp - the parser takes the bare copy, one row per epoch).
  * 70 logged points is therefore the last 31,500 steps of each arm's schedule,
    whatever its block layout. Checked rather than assumed: `sea-lr_sA`'s final
    logged row (1.39e-04 / 0.038936) reproduces `valid_metrics.tsv`'s full-split
    `sea-lr_A` (1.38545690e-04 / 3.89379747e-02) to 0.3%.

The window is the last 70 rows *inclusive* of the final one; `mean.py` used
`data[-70:-1]`, which drops that row. The difference is one row out of 70.

WHAT THE TWO COLUMNS ACTUALLY ARE - they are not the same quantity on both sides.

  * `rmse_f` is comparable across families: every force metric here pools over all
    atoms and components, so a per-batch RMS and a whole-split RMS agree (the new
    arm's final lcurve row 4.23e-02 against its full-split 4.2377e-02 measured with
    `decompose_final.py`).
  * `rmse_e/atom` is NOT comparable across families. deepmd logs the natoms-weighted
    mean of each validation batch's own metric (`pt/train/training.py:952-956`), and
    every arm here validates with `batch_size: 1`. An energy "RMSE" over a one-frame
    batch is that frame's absolute error, so the mean over 50 batches is a mean
    ABSOLUTE error per atom, systematically below the pooled RMS. Measured on the
    new arm's final checkpoint: mean per-frame |dE|/natoms = 1.3251e-04 against a
    pooled RMSE of 1.6159e-04, ratio 0.820 - a final lcurve row of 1.36e-04 is that
    mean-absolute number, not the 1.616e-04 the full-split evaluator reports. The
    cace/sea logs' `val_e/atom_rmse` is a pooled RMSE over the whole split.
    So the energy column compares deepmd-family arms with each other and sea-family
    arms with each other, and NOT across the two. Use `valid_sweep_les.tsv` (this
    arm, full split) against `valid_metrics.tsv` (sea arms, full split) for that.

The convergence check matters because a window mean describes a checkpoint that
may not exist: if the window is still drifting, its mean is not a value any single
checkpoint had. The table prints the first-quarter / last-quarter ratio of each
window so a drifting measure is visible rather than silently averaged away. In the
2026-10-07 table the force column is flat (drift 1.00) for every arm, while the
`deepmd` arms' energy is still descending (drift 1.38-1.40), so their energy window
mean is pessimistic against their own final row.
"""
import argparse
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
RUNS = os.path.join(CAMPAIGN, "deepmd", "runs")

TAIL = 70          # logged validation points; 70 x 450 steps = 31,500 steps

# arm -> (kind, final block). Both figures come from the arm's own logs.
ARMS = {
    "deepmd_sA": ("lcurve", "s7"),
    "deepmd_sB": ("lcurve", "s7"),
    "deepmd-les_sA": ("lcurve", "s8"),
    "deepmd-les_sB": ("lcurve", "s8"),
    "deepmd-les-cace-sched_sA": ("lcurve", "s8"),
    "deepmd-les-cace-sched_sB": ("lcurve", "s8"),
    "sea-sr_sA": ("cace", None),
    "sea-sr_sB": ("cace", None),
    "sea-lr_sA": ("cace", None),
    "sea-lr_sB": ("cace", None),
}


def from_lcurve(armdir, block):
    """(global step, rmse_e_val, rmse_f_val) from a block's lcurve.out.

    The block's own step counter restarts at 0, so the global step is the one the
    rest of the campaign quotes: `segments.json`'s global_offset + the local step.
    """
    path = os.path.join(armdir, block, "lcurve.out")
    data = np.loadtxt(path, usecols=(0, 3, 5))      # step, rmse_e_val, rmse_f_val
    with open(os.path.join(armdir, "segments.json")) as fh:
        offset = {s["n"]: s["global_offset"] for s in json.load(fh)["segments"]}
    return data[:, 0] + offset[int(block[1:])], data[:, 1], data[:, 2]


EPOCH_RE = re.compile(r"Epoch\s+(\d+),\s*Train Loss")
E_RE = re.compile(r"val_e/atom_rmse:\s*([0-9.eE+-]+)")
F_RE = re.compile(r"val_f_rmse:\s*([0-9.eE+-]+)")


def from_train_log(armdir):
    """Per-epoch (global step, val_e/atom_rmse, val_f_rmse) from a cace/sea log.

    The epoch counter restarts at every task (5 tasks of 40 epochs, then 3 of 100),
    so it is not a global step. The rows are sequential over the whole chain and
    every epoch is `steps_per_epoch` = 450 steps, so the row index is: row i is the
    end of step (i + 1) * 450. That is the same quantity the deepmd side's
    `global_offset + local step` produces, which is why the column is comparable.
    """
    rows = []
    cur = None
    with open(os.path.join(armdir, "train.log"), errors="ignore") as fh:
        for line in fh:
            if EPOCH_RE.search(line):
                if cur is not None and None not in cur[1:]:
                    rows.append(cur)
                cur = (int(EPOCH_RE.search(line).group(1)), None, None)
                continue
            if cur is None:
                continue
            m = E_RE.search(line)
            if m and cur[1] is None:
                cur = (cur[0], float(m.group(1)), cur[2])
                continue
            m = F_RE.search(line)
            if m and cur[2] is None:
                cur = (cur[0], cur[1], float(m.group(1)))
    if cur is not None and None not in cur[1:]:
        rows.append(cur)
    arr = np.array(rows, dtype=float)
    steps = (np.arange(len(arr)) + 1) * 450.0
    return steps, arr[:, 1], arr[:, 2]


def window(steps, e, f, tail):
    """Tail window plus a drift ratio (first quarter / last quarter of the window)."""
    e, f, steps = e[-tail:], f[-tail:], steps[-tail:]
    q = max(1, len(e) // 4)
    drift_e = e[:q].mean() / e[-q:].mean()
    drift_f = f[:q].mean() / f[-q:].mean()
    return steps[0], steps[-1], len(e), e.mean(), f.mean(), drift_e, drift_f


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tail", type=int, default=TAIL)
    ap.add_argument("--arms", nargs="*", default=sorted(ARMS))
    args = ap.parse_args()

    print(f"tail window: last {args.tail} logged validation points "
          f"= last {args.tail * 450} steps of each arm's schedule")
    print("rmse_f is comparable across families; rmse_e/atom is a mean ABSOLUTE error "
          "on the deepmd side and a pooled RMSE on the sea side (see the docstring)\n")
    head = (f"{'arm':28s} {'window':>17s} {'rmse_e/atom':>12s} {'rmse_f':>9s} "
            f"{'drift_e':>8s} {'drift_f':>8s}")
    print(head)
    print("-" * len(head))

    table = {}
    drift = {}
    for arm in args.arms:
        kind, block = ARMS[arm]
        armdir = os.path.join(RUNS, arm)
        if not os.path.isdir(armdir):
            print(f"{arm:28s} MISSING")
            continue
        steps, e, f = (from_lcurve(armdir, block) if kind == "lcurve"
                       else from_train_log(armdir))
        lo, hi, n, me, mf, de, df = window(steps, e, f, args.tail)
        table[arm] = (me, mf)
        drift[arm] = (de, df)
        print(f"{arm:28s} {int(lo):>7d}-{int(hi):<7d} {me:12.6e} {mf:9.5f} "
              f"{de:8.3f} {df:8.3f}")

    print("\nreplicate means (the number to quote for an arm):")
    means = {}
    for base in ("deepmd", "deepmd-les", "deepmd-les-cace-sched", "sea-sr", "sea-lr"):
        pair = [table.get(f"{base}_s{r}") for r in ("A", "B")]
        if any(p is None for p in pair):
            continue
        e = np.mean([p[0] for p in pair])
        f = np.mean([p[1] for p in pair])
        means[base] = pair
        print(f"  {base:26s} rmse_e/atom {e:12.6e}  rmse_f {f:9.5f}   "
              f"(A/B spread {max(p[0] for p in pair) / min(p[0] for p in pair):.2f}x / "
              f"{max(p[1] for p in pair) / min(p[1] for p in pair):.2f}x)")

    # The questions this table exists to answer, in the order they get asked.
    print("\npairwise ratios (B/A; < 1 means the first arm is better). 'disjoint' is "
          "whether the two replicates' windows do not overlap at all,\nwhich at n=2 is "
          "the only significance statement available and a weak one:")
    questions = [
        ("setting effect", "deepmd-les-cace-sched", "deepmd-les", False),
        ("implementation effect", "deepmd-les-cace-sched", "sea-lr", True),
        ("LR payoff, cace", "sea-sr", "sea-lr", False),
        ("LR payoff, deepmd", "deepmd", "deepmd-les", False),
    ]
    for label, a, b, cross_family in questions:
        if a not in means or b not in means:
            continue
        # means[arm] is the two replicates' (rmse_e, rmse_f) windows
        ea = [p[0] for p in means[a]]
        fa = [p[1] for p in means[a]]
        eb = [p[0] for p in means[b]]
        fb = [p[1] for p in means[b]]
        de = "yes" if (max(ea) < min(eb) or max(eb) < min(ea)) else "no"
        if cross_family:
            de = "n/a"          # two different measures, so overlap means nothing
        df = "yes" if (max(fa) < min(fb) or max(fb) < min(fa)) else "no"
        # a ratio over a window that is still descending describes no checkpoint,
        # so it is flagged rather than printed as if it were a result
        warn = ""
        for arm in (a, b):
            pair = [f"{arm}_s{r}" for r in ("A", "B")]
            worst = max(drift[p][0] for p in pair if p in drift)
            if worst > 1.05:
                warn += (f"  [energy window of {arm} still descending: "
                         f"first/last quarter {worst:.2f}]")
        print(f"  {label:22s} {a} / {b}: force {np.mean(fa) / np.mean(fb):.4f}  "
              f"energy "
              + ("(not comparable across families, see docstring)"
                 if cross_family else f"{np.mean(ea) / np.mean(eb):.4f}")
              + f"   disjoint F/E: {df}/{de}{warn}")


if __name__ == "__main__":
    main()