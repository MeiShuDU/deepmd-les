"""Every arm of the water-interface campaign on ONE number: pooled RMSE, final ckpt.

`compare_les_sched.py` compares the deepmd-family arms through their `lcurve.out`
logs (the campaign's habit), and its docstring explains why the energy column of
that comparison is not comparable across families: deepmd logs the natoms-weighted
mean of per-batch metrics, and every arm validates with `batch_size: 1`, so a
deepmd energy "RMSE" is really a mean absolute error, systematically below the
pooled RMS the cace/sea arms report.

This script sidesteps the logging difference entirely. Every arm is evaluated by
the same definition - sqrt(mean((dE/natoms)^2)) and sqrt(mean(dF^2)) over the
whole 50-frame validation split, at the arm's final checkpoint:

  * deepmd family: `analysis/data/valid_pooled_deepmd.tsv`, produced by
    `decompose_final.py` (deepmd's own `DeepPot` on the same frames);
  * cace/sea family: `analysis/valid_metrics.tsv`, produced by `score_arms.py`
    (cace's model code on the same frames, same formula).

Checked rather than assumed: on the arm where both numbers exist, the deepmd side
of the pair reproduces to 0.2% in force (deepmd-les-cace-sched_sA final lcurve row
4.23e-02 vs 4.2377e-02 pooled) and the energy difference is exactly the
mean-absolute-vs-RMS gap (1.36e-04 vs 1.616e-04).

Final checkpoints rather than window means: a window mean is a fine smoothing of a
training curve, but a shipped model is one checkpoint, and the final one is what
the campaign's `valid_metrics.tsv` scores for the cace/sea arms. The tail-window
view lives in `compare_les_sched.py`; the two agree on every question below.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# arm -> (family, rep -> (rmse_e_peratom, rmse_f))
DEEP = os.path.join(HERE, "data", "valid_pooled_deepmd.tsv")
METRICS = os.path.join(HERE, "valid_metrics.tsv")

# The questions, in the order they get asked. `cross_family` is gone: on this
# footing every pair is comparable, which is the point of the file.
QUESTIONS = [
    ("setting effect", "deepmd-les-cace-sched", "deepmd-les"),
    ("implementation effect", "deepmd-les-cace-sched", "sea-lr"),
    ("vs sibling SR", "deepmd-les-cace-sched", "sea-sr"),
    ("LR payoff, cace", "sea-sr", "sea-lr"),
    ("LR payoff, deepmd", "deepmd", "deepmd-les"),
]


def load():
    arms = {}
    with open(DEEP) as fh:
        head = fh.readline().split()
        for line in fh:
            row = dict(zip(head, line.split()))
            base = row["arm"].rsplit("_", 1)[0]
            arms.setdefault(base, {})[row["rep"]] = (
                float(row["rmse_e_peratom"]), float(row["rmse_f"]))
    with open(METRICS) as fh:
        head = fh.readline().split()
        for line in fh:
            row = dict(zip(head, line.split("\t")))
            base = row["arm_id"].rsplit("_", 1)[0]
            arms.setdefault(base, {})[row["rep"]] = (
                float(row["rmse_e_peratom"]), float(row["rmse_f"]))
    return arms


def main():
    arms = load()
    print("pooled RMSE over the 50-frame validation split, final checkpoint, "
          "one definition for every arm\n")
    head = f"{'arm':26s} {'family':7s} {'rmse_e/atom':>12s} {'rmse_f':>10s} " \
           f"{'A/B spread E':>12s} {'A/B spread F':>12s}"
    print(head)
    print("-" * len(head))
    family = {"deepmd": "deepmd", "deepmd-les": "deepmd",
              "deepmd-les-cace-sched": "deepmd",
              "cace-sr": "cace", "cace-lr": "cace",
              "sea-sr": "sea", "sea-lr": "sea"}
    order = sorted(arms, key=lambda a: (family[a], np.mean([p[1] for p in arms[a].values()])))
    for arm in order:
        reps = arms[arm]
        e = np.mean([p[0] for p in reps.values()])
        f = np.mean([p[1] for p in reps.values()])
        se = max(p[0] for p in reps.values()) / min(p[0] for p in reps.values())
        sf = max(p[1] for p in reps.values()) / min(p[1] for p in reps.values())
        print(f"{arm:26s} {family[arm]:7s} {e:12.6e} {f:10.5f} "
              f"{se:11.3f}x {sf:11.3f}x   ({', '.join(sorted(reps))})")

    print("\npairwise ratios (< 1 means the first arm is better on that metric). "
          "'disjoint' is whether the two replicates' values\nare separated; at n=2 "
          "that is the only significance statement available, and a weak one:")
    for label, a, b in QUESTIONS:
        ra, rb = arms[a], arms[b]
        ea = [p[0] for p in ra.values()]
        fa = [p[1] for p in ra.values()]
        eb = [p[0] for p in rb.values()]
        fb = [p[1] for p in rb.values()]
        de = "yes" if (max(ea) < min(eb) or max(eb) < min(ea)) else "no"
        df = "yes" if (max(fa) < min(fb) or max(fb) < min(fa)) else "no"
        print(f"  {label:22s} {a} / {b}:  force {np.mean(fa) / np.mean(fb):.4f}  "
              f"energy {np.mean(ea) / np.mean(eb):.4f}   disjoint F/E: {df}/{de}")

    # What each number is made of, for the two arms the mission is about, so a
    # "better" cannot hide "differently biased".
    print("\noffset / scale, deepmd family (an RMSE alone cannot say displaced vs noisy):")
    with open(DEEP) as fh:
        head = fh.readline().split()
        for line in fh:
            row = dict(zip(head, line.split()))
            base, tag = row["arm"].rsplit("_", 1)
            print(f"  {base:26s} {tag}  offset {float(row['energy_offset_peratom']):+.3e} eV/atom"
                  f"  slope {float(row['energy_scale_slope']):.4f}"
                  f"  rmse_e/atom {float(row['rmse_e_peratom']):.6e}")


if __name__ == "__main__":
    main()