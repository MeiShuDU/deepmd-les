"""Score the chained cace-schedule arms and write the campaign's metric tables.

The chained arms differ from every other arm in this campaign in one way that shows
up here: their work is spread over eight directories, each with its own step counter
that starts at 0. So the two scoring steps are

1. score each block directory on its own (``eval_campaign.eval_arm`` for the
   validation split, ``lr_force_share.measure`` for the long-range decomposition),
   exactly as if it were a stand-alone run; then
2. relabel its step to the GLOBAL step, ``segments.json``'s ``global_offset`` plus
   the checkpoint's own number.

That relabeling is what makes the numbers comparable to the other arms: deepmd's
checkpoint number is the count of batches that have been processed, and cace's epoch
counter is the same count divided by 160, so a chained block 6 checkpoint numbered
3200 is global step 35200 - the same amount of training as a single-run arm's
``model.ckpt-35200.pt``. The phase endpoints fall out at 32000 / 48000 / 64000 /
80000, which is where the campaign's other arms are quoted.

Nothing here re-implements the metrics; both come from the campaign's own modules so
the new rows land in the same schema as the old ones.

Usage:
    python score_sea_chain.py --reset                       # fresh valid_sweep_sea.tsv
    python score_sea_chain.py --arms deepmd-les-cace-sched_sA
    python score_sea_chain.py --steps 32000 48000 64000 80000
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lr_force_share
from deepmd.common import j_loader
from eval_campaign import eval_arm, kept_steps

CAMPAIGN = os.path.dirname(HERE)
ACC_COLS = ["arm", "step", "rmse_e_peratom", "rmse_f", "natoms", "nframes"]
DEFAULT_ACC = os.path.join(CAMPAIGN, "analysis", "data", "valid_sweep_sea.tsv")
DEFAULT_LR = os.path.join(CAMPAIGN, "analysis", "data", "lr_mechanism_sea.tsv")


def chains(root):
    """Every chained arm directory under `root`, in a stable order."""
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.exists(os.path.join(d, "segments.json")):
            out.append((name, d))
    return out


def segments(armdir):
    with open(os.path.join(armdir, "segments.json")) as fh:
        return {s["n"]: s for s in json.load(fh)["segments"]}


def blocks(armdir):
    """(block number, block dir, global offset) for the blocks that exist."""
    segs = segments(armdir)
    out = []
    for n in sorted(segs):
        d = os.path.join(armdir, f"s{n}")
        if os.path.isdir(d):
            out.append((n, d, segs[n]["global_offset"]))
    return out


def wanted_global(segs, steps):
    """Which (block, local step) pairs correspond to the requested global steps."""
    if not steps:
        return None
    want = set(steps)
    out = []
    for n, d, off in segs:
        local = sorted(w for w in kept_steps(d) if off + w in want)
        if local:
            out.append((n, d, off, local))
    return out


def write_tsv(path, cols, rows, reset):
    append = os.path.exists(path) and not reset
    seen = {}
    if append:
        with open(path) as fh:
            head = fh.readline().strip().split("\t")
            if head != cols:
                raise SystemExit(f"{path}: header {head} != expected {cols}")
            for line in fh:
                f = line.rstrip("\n").split("\t")
                seen[(f[0], int(f[1]))] = f
    for r in rows:
        seen[(r[0], int(r[1]))] = r
    ordered = [seen[k] for k in sorted(seen, key=lambda k: (k[0], k[1]))]
    with open(path, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in ordered:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"\nwrote {path}: {len(ordered)} rows "
          f"({len(rows)} from this invocation)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--arms", nargs="*", default=None,
                    help="chained arm directory names; default all under --root")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="global steps to score; default every kept checkpoint")
    ap.add_argument("--acc-out", default=DEFAULT_ACC)
    ap.add_argument("--lr-out", default=DEFAULT_LR)
    ap.add_argument("--skip-lr", action="store_true",
                    help="skip the long-range decomposition (SR-only arms have none)")
    ap.add_argument("--reset", action="store_true", help="start the TSVs over")
    args = ap.parse_args()

    all_chains = chains(args.root)
    if args.arms:
        want = set(args.arms)
        all_chains = [(a, d) for a, d in all_chains if a in want]
        missing = want - {a for a, _ in all_chains}
        if missing:
            raise SystemExit(f"no segments.json for {sorted(missing)} under {args.root}")
    if not all_chains:
        raise SystemExit(f"no chained arms under {args.root}")

    acc_rows, lr_rows = [], []
    for arm, armdir in all_chains:
        segs = blocks(armdir)
        limited = wanted_global(segs, args.steps)
        print(f"\n=== {arm} ({len(segs)} blocks) ===", flush=True)
        for n, d, off in segs:
            steps = kept_steps(d)
            if limited is not None:
                steps = [s for s in steps if off + s in args.steps]
            if not steps:
                continue
            for r in eval_arm(d, steps, "valid", None):
                _, step, e, f, nat, nfr = r
                acc_rows.append((arm, off + step, f"{e:.8e}", f"{f:.8e}", nat, nfr))
            if args.skip_lr:
                continue
            cfg = j_loader(os.path.join(d, "input.yaml"))["model"]["type"]
            if cfg != "hybrid_ener":
                continue
            for step in steps:
                row = lr_force_share.measure(d, step, 0)
                if row:
                    row["arm"], row["step"] = arm, off + step
                    lr_rows.append(row)

    if acc_rows:
        def cell(r):
            return [r[0], r[1]] + list(r[2:])

        write_tsv(args.acc_out, ACC_COLS, [cell(r) for r in acc_rows], args.reset)
    if lr_rows:
        def lcell(r):
            out = []
            for c in lr_force_share.COLS:
                v = r[c]
                if isinstance(v, str):
                    out.append(v)
                elif v is None:
                    out.append("")
                elif isinstance(v, bool):
                    out.append(str(int(v)))
                elif isinstance(v, int):
                    out.append(str(v))
                else:
                    out.append(f"{v:.8e}")
            return out

        write_tsv(args.lr_out, lr_force_share.COLS,
                  [lcell(r) for r in lr_rows], args.reset)
    if not acc_rows:
        print("nothing scored", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
