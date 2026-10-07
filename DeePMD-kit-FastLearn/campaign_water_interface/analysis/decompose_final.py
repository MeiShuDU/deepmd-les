"""Per-frame errors of one checkpoint on the 50 validation frames, decomposed.

Why this exists: `eval_campaign.eval_arm` reports aggregate RMSEs through deepmd's
own `test_ener`, and the `lcurve.out` rows report a per-display-step RMSE. The two
disagree for the arm scored here (final full-split rmse_e 1.616e-04 against the
final lcurve row's 1.36e-04, force agreeing to 1e-3 relative). Rather than pick
one, this computes the errors per frame from the model itself, which is also what
the campaign's `valid_metrics.tsv` columns are built from:

    energy_offset_peratom    the mean bias per atom
    energy_scale_slope       the regression slope of prediction on reference
    energy_rmse_after_offset RMSE after subtracting the mean bias
    energy_rmse_after_scale  RMSE after removing the fitted scale

An RMSE alone cannot say whether an arm is displaced, mis-scaled, or noisy, and the
three call for different fixes, so the split is reported even when the headline
number is fine.

`--lr` additionally reports the long-range branch by the same difference method as
`lr_force_share` (E_LR = E(w=1) - E(w=0)), so the precision of this script's own
evaluation can be checked against `lr_mechanism_les.tsv`.

Run with DP_INTERFACE_PREC=low and CUDA_VISIBLE_DEVICES= to match training.
"""
import argparse
import os
import sys

import numpy as np

VALID = ("/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/"
         "campaign_water_interface/data/water-interface/valid")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rundir")
    ap.add_argument("--block", default=None, help="block dir under rundir, e.g. s8")
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--lr", action="store_true")
    args = ap.parse_args()

    block = args.block or sorted(
        d for d in os.listdir(args.rundir)
        if d.startswith("s") and d[1:].isdigit())[-1]
    blockdir = os.path.join(args.rundir, block)
    steps = sorted(int(p.split("-")[1].split(".")[0])
                   for p in os.listdir(blockdir)
                   if p.startswith("model.ckpt-") and p.endswith(".pt"))
    if steps:
        step = args.step or steps[-1]
        ckpt = os.path.join(blockdir, f"model.ckpt-{step}.pt")
    else:
        # The legacy pod runs kept only the final `model.ckpt.pt` in this block
        # (a full state dict, not the 19-byte pointer the new arms also carry).
        # Date it by the last logged step so the printout still says which
        # checkpoint was scored.
        ckpt = os.path.join(blockdir, "model.ckpt.pt")
        if not os.path.exists(ckpt):
            raise SystemExit(f"no checkpoint in {blockdir}")
        lc = np.loadtxt(os.path.join(blockdir, "lcurve.out"), usecols=(0,))
        step = int(lc[-1])

    coord = np.load(os.path.join(VALID, "set.000", "coord.npy"))
    box = np.load(os.path.join(VALID, "set.000", "box.npy"))
    types = np.loadtxt(os.path.join(VALID, "type.raw")).astype(int)
    ref_e = np.load(os.path.join(VALID, "set.000", "energy.npy")).reshape(-1)
    ref_f = np.load(os.path.join(VALID, "set.000", "force.npy"))
    nframes = coord.shape[0]
    natoms = types.shape[0]
    ref_f = ref_f.reshape(nframes, natoms, 3)

    import deepmd.env  # noqa: E402
    from deepmd.infer.deep_pot import DeepPot  # noqa: E402

    dp = DeepPot(ckpt, no_jit=True)

    def ev():
        e, f, _ = dp.eval(coord, box, types, atomic=False)
        return np.asarray(e).reshape(nframes), np.asarray(f).reshape(nframes, -1, 3)

    e_tot, f_tot = ev()
    print(f"{os.path.basename(args.rundir)} {os.path.relpath(ckpt, args.rundir)}  "
          f"nframes={nframes} natoms={natoms}  step~{step}")

    # --- how each aggregate is defined, checked against eval_campaign's output ---
    per_atom_err = (e_tot - ref_e) / natoms
    rmse_e = float(np.sqrt((per_atom_err ** 2).mean()))
    rmse_e_alt = float(np.sqrt(((e_tot - ref_e) ** 2).mean()) / natoms)
    rmse_f = float(np.sqrt(((f_tot - ref_f) ** 2).mean()))
    print(f"  rmse_e/atom  = {rmse_e:.6e}   (sqrt(mean((dE/natoms)^2)))")
    print(f"  rmse_e alt   = {rmse_e_alt:.6e}   (sqrt(mean(dE^2))/natoms)")
    print(f"  rmse_f       = {rmse_f:.6e}")

    # --- offset / scale / shape ---
    slope, intercept = np.polyfit(ref_e, e_tot, 1)
    after_offset = e_tot - (ref_e + per_atom_err.mean() * natoms)
    after_scale = e_tot - (ref_e * slope + ref_e.mean() * (1 - slope))
    print(f"  offset/atom  = {per_atom_err.mean():+.6e} eV")
    print(f"  scale slope  = {slope:.6f}  (1.0 = correctly scaled)")
    print(f"  rmse after offset = "
          f"{float(np.sqrt(((after_offset / natoms) ** 2).mean())):.6e}")
    print(f"  rmse after scale  = "
          f"{float(np.sqrt(((after_scale / natoms) ** 2).mean())):.6e}")
    print(f"  |E| ref per atom  = {np.abs(ref_e).mean() / natoms:.6e} eV")

    if args.lr:
        model = dp.deep_eval.dp.model["Default"]
        trained = model.lr_weight
        model.lr_weight = 1.0
        e_w1, f_w1 = ev()
        model.lr_weight = 0.0
        e_w0, f_w0 = ev()
        model.lr_weight = trained
        e_lr, f_lr = e_w1 - e_w0, f_w1 - f_w0
        rms = lambda a: float(np.sqrt((a ** 2).mean()))
        print(f"  E_LR mean = {e_lr.mean():+.4f} +- {e_lr.std():.4f}   "
              f"E_SR mean = {e_w0.mean():+.4f}")
        print(f"  |F| tot(w=1) = {rms(f_w1):.4f}  LR = {rms(f_lr):.4f}  "
              f"share = {rms(f_lr) / rms(f_w1):.4f}  (raw branch, w=1)")
        f_tot_trained = f_w0 + trained * f_lr
        print(f"  |F| tot at trained w={trained} = {rms(f_tot_trained):.4f}  "
              f"LR share there = {rms(trained * f_lr) / rms(f_tot_trained):.4f}")


if __name__ == "__main__":
    main()