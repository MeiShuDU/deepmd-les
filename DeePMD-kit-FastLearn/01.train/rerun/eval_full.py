"""Evaluate saved checkpoints over a whole split, one (run, step) per process.

Why this exists
---------------
The `rmse_*_val` columns in lcurve_*.out are computed over only `numb_btch`=3
randomly drawn frames per display step (validation_data batch_size 1,
numb_btch 3). That is a small, noisy subsample. It is also not enough to
explain the tail: evaluating the saved checkpoint on all 80 frames shows the
model's own validation error oscillating by ~1.4-2.3x across consecutive late
checkpoints (e.g. run_hybrid_fixed_sB: 9.91e-4 / 6.78e-4 / 1.55e-3 at steps
8000 / 9000 / 10000, while force falls smoothly). So neither a single logged
point nor a single checkpoint is a reliable energy-accuracy estimate: the
comparison has to average over several late checkpoints.

This script evaluates a checkpoint over a whole split at once (all frames, no
subsampling), reusing the official DeepMD test math
(deepmd.entrypoints.test.test_ener + weighted_average) so the numbers are the
same metric as the lcurve columns.

no_jit=True is required: `dp --pt test` would call torch.jit.script on the
reconstructed model (deepmd/pt/infer/deep_eval.py) and the hybrid model is not
scriptable (the LES library formats f-strings inside the forward path).

Usage:
    python eval_full.py <split> <step> <run_dir_name> [...]
Writes/appends: sweep_<split>.tsv, or sweep_<split>_d<decay>.tsv when the runs
belong to a non-default learning-rate schedule. The decay is taken from the run
names and never mixed: a single file is one schedule, because the rows are only
comparable within a schedule.
"""
import argparse
import os

from deepmd.common import j_loader
from deepmd.entrypoints.test import test_ener
from deepmd.infer.deep_pot import DeepPot
from deepmd.utils.data import DeepmdData
from deepmd.utils.weight_avg import weighted_average

from runname import DEFAULT_DECAY, parse_run

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "extended")
RUNS = [
    "run_ordinary_sA",
    "run_hybrid_q_sA",
    "run_hybrid_fixed_sA",
    "run_ordinary_sB",
    "run_hybrid_q_sB",
    "run_hybrid_fixed_sB",
]


def eval_run(name, split, ckpt):
    rundir = os.path.join(EXT, name)
    jdata = j_loader(os.path.join(rundir, "input.yaml"))
    key = {"train": "training_data", "valid": "validation_data"}[split]
    systems = jdata["training"][key]["systems"]
    if isinstance(systems, str):
        systems = [systems]

    dp = DeepPot(os.path.join(rundir, ckpt), no_jit=True)
    tmap = dp.get_type_map()

    err_coll = []
    natoms = nframes = None
    for system in systems:
        data = DeepmdData(
            system, set_prefix="set", shuffle_test=False, type_map=tmap, sort_atoms=False
        )
        err_coll.append(test_ener(dp, data, system, float("inf"), None, False))
        natoms, nframes = data.get_natoms(), int(data.nframes)

    avg = weighted_average(err_coll)
    return avg["rmse_ea"], avg["rmse_f"], natoms, nframes


def main():
    # One (run, step) per process: loading several models into one CUDA context
    # trips an allocator internal assert on this host.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("split", choices=["train", "valid"])
    ap.add_argument("step", type=int)
    ap.add_argument("names", nargs="*", default=RUNS)
    ap.add_argument(
        "--reset",
        action="store_true",
        help="truncate the sweep file before writing instead of appending; the "
        "caller does not have to know the file name, which depends on the mix "
        "of learning-rate schedules in `names`",
    )
    args = ap.parse_args()
    split, step = args.split, args.step
    names = args.names or RUNS
    ckpt = f"model.ckpt-{step}.pt"

    metas = [parse_run(n) for n in names]
    decays = {m["decay"] for m in metas}
    if len(decays) > 1:
        raise SystemExit(
            f"refusing to mix learning-rate schedules in one sweep file: {decays}"
        )
    decay = decays.pop()
    suffix = "" if decay == DEFAULT_DECAY else f"_d{decay}"
    out = os.path.join(HERE, f"sweep_{split}{suffix}.tsv")
    append = os.path.exists(out) and not args.reset
    rows = []
    for meta in metas:
        name = meta["run"]
        rmse_e, rmse_f, natoms, nframes = eval_run(name, split, ckpt)
        rows.append(
            (name, meta["model"], meta["rep"], meta["decay"], step, rmse_e, rmse_f,
             natoms, nframes)
        )
        print(
            f"{name:26s} {meta['model']:12s} s{meta['rep']} d{meta['decay']} "
            f"step={step} [{split}] "
            f"rmse_e/Natoms={rmse_e:.6e} rmse_f={rmse_f:.6e}",
            flush=True,
        )

    with open(out, "a" if append else "w") as f:
        if not append:
            f.write(
                "run\tmodel\trep\tdecay\tstep\trmse_e_peratom\trmse_f\tnatoms\tnframes\n"
            )
        for r in rows:
            f.write(
                f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t{r[5]:.8e}\t{r[6]:.8e}\t"
                f"{r[7]}\t{r[8]}\n"
            )
    print("wrote", out)


if __name__ == "__main__":
    main()
