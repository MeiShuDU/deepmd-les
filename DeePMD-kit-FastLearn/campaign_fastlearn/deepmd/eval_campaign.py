"""Evaluate campaign checkpoints over the whole validation split.

Why this exists: the `rmse_*_val` columns in `lcurve.out` are computed over only
`numb_btch: 3` frames per display step, and consecutive logged points come from
different checkpoints, so a single logged point is not an accuracy estimate.
The evaluation protocol for this campaign is to score every kept late checkpoint
over ALL validation frames and average, which is also what `01.train/rerun/
eval_full.py` does for the earlier runs; this is the same math (deepmd's own
`test_ener` + `weighted_average`), pointed at the campaign's run layout.

`no_jit=True` is required: the hybrid model is not scriptable from a checkpoint
(the `element_numbers` buffer is registered `persistent=False`, and TorchScript
drops that flag, so the scripted model asks for a key the checkpoint lacks). See
`HPC_HANDOFF.md` section 6.

One (arm, step) per process: loading several models into one CUDA context has
tripped an allocator internal assert on this hardware before, so the driver
calls this once per pair rather than looping inside one process.

Usage:
    python eval_campaign.py --step 80000 deepmd-les_sA deepmd-les_sB
    python eval_campaign.py --steps auto --out sweep_valid.tsv      # every kept ckpt
Writes/appends a TSV: arm, step, rmse_e_peratom, rmse_f, natoms, nframes.
"""
import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(HERE, "runs")


def kept_steps(rundir):
    steps = []
    for path in glob.glob(os.path.join(rundir, "model.ckpt-*.pt")):
        m = re.search(r"model\.ckpt-(\d+)\.pt$", path)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def eval_arm(rundir, steps, split, out_path, reset=False):
    from deepmd.common import j_loader
    from deepmd.entrypoints.test import test_ener
    from deepmd.infer.deep_pot import DeepPot
    from deepmd.utils.data import DeepmdData
    from deepmd.utils.weight_avg import weighted_average

    arm = os.path.basename(rundir.rstrip("/"))
    jdata = j_loader(os.path.join(rundir, "input.yaml"))
    key = {"train": "training_data", "valid": "validation_data"}[split]
    systems = jdata["training"][key]["systems"]
    if isinstance(systems, str):
        systems = [systems]
    # deepmd resolves a relative `systems` entry against the process CWD, not
    # against the config file, so the configs only work when dp is launched from
    # the run directory (which is what the training runner does). Here the CWD is
    # the parent of runs/, so resolve them against the run dir explicitly.
    systems = [
        s if os.path.isabs(s) else os.path.normpath(os.path.join(rundir, s))
        for s in systems
    ]

    rows = []
    for step in steps:
        ckpt = os.path.join(rundir, f"model.ckpt-{step}.pt")
        if not os.path.exists(ckpt):
            print(f"  skip {arm} step {step}: no checkpoint", flush=True)
            continue
        dp = DeepPot(ckpt, no_jit=True)
        tmap = dp.get_type_map()
        err_coll = []
        natoms = nframes = None
        for system in systems:
            data = DeepmdData(system, set_prefix="set", shuffle_test=False,
                              type_map=tmap, sort_atoms=False)
            err_coll.append(test_ener(dp, data, system, float("inf"), None, False))
            natoms, nframes = data.get_natoms(), int(data.nframes)
        avg = weighted_average(err_coll)
        rows.append((arm, step, avg["rmse_ea"], avg["rmse_f"], natoms, nframes))
        print(f"{arm:34s} step={step:<7d} [{split}] "
              f"rmse_e/Natoms={avg['rmse_ea']:.6e} rmse_f={avg['rmse_f']:.6e} "
              f"({nframes} frames)", flush=True)
        del dp

    if out_path:
        append = os.path.exists(out_path) and not reset
        with open(out_path, "a" if append else "w") as fh:
            if not append:
                fh.write("arm\tstep\trmse_e_peratom\trmse_f\tnatoms\tnframes\n")
            for r in rows:
                fh.write(f"{r[0]}\t{r[1]}\t{r[2]:.8e}\t{r[3]:.8e}\t{r[4]}\t{r[5]}\n")
        print("wrote", out_path, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+",
                    help="run directory names under --runs-root")
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    ap.add_argument("--step", type=int, action="append",
                    help="checkpoint step to score; repeatable")
    ap.add_argument("--steps", choices=["auto"], help="score every kept checkpoint")
    ap.add_argument("--split", choices=["train", "valid"], default="valid")
    ap.add_argument("--out", help="TSV to append to")
    ap.add_argument("--reset", action="store_true", help="truncate --out first")
    args = ap.parse_args()

    if not args.step and not args.steps:
        raise SystemExit("pass --step N (repeatable) or --steps auto")

    reset = args.reset
    for name in args.arms:
        rundir = os.path.join(args.runs_root, name)
        if not os.path.isdir(rundir):
            print(f"no such run: {rundir}", file=sys.stderr)
            continue
        steps = args.step or kept_steps(rundir)
        if not steps:
            print(f"  {name}: no checkpoints yet", file=sys.stderr)
            continue
        eval_arm(rundir, steps, args.split, args.out, reset=reset)
        reset = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
