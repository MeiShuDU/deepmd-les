"""Decompose a campaign arm's energy error into level (bias) and shape (spread).

The TSV from `eval_campaign.py` is a raw RMSE, and in this project a raw energy
RMSE is dominated by the constant offset the fitting bias absorbs (the SPC/E
atomic-energies reuse, deepmd's own -29945 reference), not by how well the model
follows the geometry. This reads per-frame errors instead and reports

    rmse         sqrt(mean(err^2))
    bias         mean(err)                      the level
    spread       std(err)                       the shape
    bias^2/rmse^2                               share of the error that is level
    rmse_cal     residual after a linear calibration E_pred = a*E_true + b

so a "worse" raw RMSE can be identified as a level shift rather than a fit
failure. Per-atom errors are used throughout (err / natoms).

One (arm, step) per process, same reason as `eval_campaign.py`.

Usage:
    python decompose_energy.py --step 80000 --out energy_decomp.tsv deepmd_sA ...
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(HERE, "runs")


def read_system(system, nmax=None):
    """Return (coords, cells, energy_per_frame) over ALL sets of a deepmd system.

    A system's frames are split across `set.000`, `set.001`, ... (deepmd caps a
    set at 1000 frames and the FastLearn export wrote `data_1` as two sets), so
    reading only `set.000` silently drops frames: `data_1` would contribute 80 of
    its 160 and the train split would come out 240 frames instead of 320, scored
    against a truncated target.
    """
    import re
    sets = sorted(d for d in os.listdir(system) if re.fullmatch(r"set\.\d+", d))
    if not sets:
        raise FileNotFoundError(f"no set.* under {system}")
    nloc = len(np.loadtxt(os.path.join(system, "type.raw")).astype(int))
    coord = np.concatenate(
        [np.load(os.path.join(system, s, "coord.npy")) for s in sets]
    ).reshape(-1, nloc, 3)
    box = np.concatenate(
        [np.load(os.path.join(system, s, "box.npy")) for s in sets]
    ).reshape(-1, 3, 3)
    # energy.npy is float32 on disk. Cast to float64 here and the calibration fit
    # below is the true least-squares optimum; leave it float32 and the fit is
    # silently degenerate. See the note in `decompose`.
    ener = np.concatenate(
        [np.load(os.path.join(system, s, "energy.npy")) for s in sets]
    ).reshape(-1).astype(np.float64)
    if nmax:
        coord, box, ener = coord[:nmax], box[:nmax], ener[:nmax]
    return coord, box, ener


def decompose(epred, etrue, nloc):
    err = (epred - etrue) / nloc
    bias = float(err.mean())
    spread = float(err.std())
    rmse = float(np.sqrt((err**2).mean()))
    # Fit E_pred = a*E_true + b. The dtype is load-bearing, not cosmetic.
    #
    # `np.polyfit` takes its rcond from `finfo(x.dtype).eps`, and `energy.npy` is
    # float32 on disk, so on uncast labels rcond = len(x) * 1.2e-7 ~ 3e-5. The
    # uncentered Vandermonde [E_true, 1] on labels of magnitude 3e4 is far worse
    # conditioned than that (cond ~ 1e9, so 1/cond ~ 8e-10), so lstsq truncates
    # the second singular value and returns the rank-1 minimum-norm solution
    # instead of the least-squares optimum. That solution is arbitrary within the
    # null space: on this data it came out a = 0.500001, whose residual exceeds
    # the raw rmse - which is exactly what the pre-fix train column reported
    # (rmse_cal 1.82e-03 against rmse 5.14e-04). It also explains why the same
    # artefact appeared at nearly the same value for all eight arms: they all got
    # the same degenerate slope. `read_system` now casts the labels to float64,
    # which drops rcond to ~5e-14 and restores the true optimum.
    #
    # Centering the targets is kept as well: the same affine family (only the
    # intercept absorbs the shift), conditioned on the label spread rather than
    # the label magnitude.
    span = float(etrue.max() - etrue.min())
    xc = etrue - etrue.mean()
    a, b = np.polyfit(xc, epred, 1)
    rmse_cal = float(np.sqrt(((((a * xc + b) - epred) / nloc) ** 2).mean()))
    # a=1, b=0 is IN the fitted family, so a correct least-squares residual can
    # never exceed the raw rmse. If it does, the fit degenerated (the float32
    # truncation above is one way) and the number is meaningless - fail loudly
    # rather than write a plausible-looking column.
    if not rmse_cal <= rmse * (1 + 1e-9):
        raise AssertionError(
            f"calibration fit is not optimal: rmse_cal {rmse_cal:.6e} > rmse "
            f"{rmse:.6e} (slope a={a:.6f}, label span {span:.4f} eV, "
            f"n={len(etrue)}, nloc={nloc}, label dtype was the caller's). "
            f"The affine fit is degenerate; do not quote rmse_cal.")
    # A slope far from 1 means the calibration is doing real work (a genuine
    # scale error) rather than soaking up a level offset, so report it.
    return rmse, bias, spread, rmse_cal, a, err, span


def run_arm(rundir, step, split, out_path, append):
    from deepmd.common import j_loader
    from deepmd.infer.deep_pot import DeepPot

    arm = os.path.basename(rundir.rstrip("/"))
    jdata = j_loader(os.path.join(rundir, "input.yaml"))
    key = {"train": "training_data", "valid": "validation_data"}[split]
    systems = jdata["training"][key]["systems"]
    systems = [systems] if isinstance(systems, str) else systems
    systems = [
        s if os.path.isabs(s) else os.path.normpath(os.path.join(rundir, s))
        for s in systems
    ]
    tmap = jdata["model"]["type_map"]
    ckpt = os.path.join(rundir, f"model.ckpt-{step}.pt")
    if not os.path.exists(ckpt):
        print(f"  skip {arm}: no {os.path.basename(ckpt)}", file=sys.stderr)
        return None

    coords, cells, ener, atype, nloc = [], [], [], None, None
    for system in systems:
        c, b, e = read_system(system)
        nloc = c.shape[1]
        coords.extend(list(c))
        cells.extend(list(b))
        ener.append(e)
        if atype is None:
            atype = np.loadtxt(os.path.join(system, "type.raw")).astype(int)
    etrue = np.concatenate(ener)

    dp = DeepPot(ckpt, no_jit=True)
    epred, _, _ = dp.eval(coords, cells, atype, atomic=False)
    epred = np.asarray(epred).reshape(-1)
    del dp

    rmse, bias, spread, rmse_cal, slope, err, span = decompose(epred, etrue, nloc)
    row = (arm, step, split, rmse, bias, spread, rmse_cal, slope, span, len(etrue))
    print(f"{arm:32s} step={step:<6d} [{split}] n={len(etrue):<4d} "
          f"rmse={rmse:.6e} bias={bias:+.6e} spread={spread:.6e} "
          f"bias2/rmse2={bias**2 / rmse**2:.3f} rmse_cal={rmse_cal:.6e} "
          f"slope={slope:.5f} span={span:.4f}eV", flush=True)
    if out_path:
        with open(out_path, "a" if append else "w") as fh:
            if not append:
                fh.write("arm\tstep\tsplit\trmse\tbias\tspread\trmse_cal\tslope"
                         "\tspan_eV\tnframes\n")
            fh.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]:.8e}\t{row[4]:.8e}"
                     f"\t{row[5]:.8e}\t{row[6]:.8e}\t{row[7]:.6f}\t{row[8]:.4f}"
                     f"\t{row[9]}\n")
    np.save(os.path.join(HERE, f"err_{arm}_{split}_{step}.npy"), err)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    ap.add_argument("--step", type=int, default=80000)
    ap.add_argument("--split", choices=["train", "valid"], default="valid")
    ap.add_argument("--out")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    append = bool(args.out) and os.path.exists(args.out) and not args.reset
    for name in args.arms:
        rundir = os.path.join(args.runs_root, name)
        if not os.path.isdir(rundir):
            print(f"no such run: {rundir}", file=sys.stderr)
            continue
        run_arm(rundir, args.step, args.split, args.out, append)
        append = True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
