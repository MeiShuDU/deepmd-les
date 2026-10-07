#!/usr/bin/env python
# coding: utf-8
"""LES magnitude diagnostics for the trained deepmd arms.

The metric table says which arm fits better; this says WHAT the long-range term
is doing in each arm, which is what makes the comparison interpretable:

  * E_lr magnitude per frame (mean/std/range) and its share of the total energy,
    next to the short-range part's share and their correlation. E_lr as large as
    the target while E_sr is flat means the long-range term is carrying the fit,
    not correcting it;
  * the same E_lr with each frame's mean charge removed. This kernel has no
    jellium background (check_ewald_reference.py), so a drifting total charge Q
    adds a Q^2 self-energy term; the difference between the two E_lr series is
    therefore the part of the long-range term that comes from charge drift, not
    from Coulomb physics. If it dominates, the arm is exploiting a kernel
    artifact, and its q are not physical charges;
  * F_lr per-component RMS against the total force RMS;
  * latent-charge statistics: per-type mean/std, max|q|, and the per-frame net
    charge |Q|.

E_lr and q are read by hooking the Les module inside the real forward, so the
numbers come from the path training uses rather than a re-derived copy of it.
The hook value is cross-checked against a second difference below.

F_lr is obtained as E_total - E_SR: the same frame is forwarded twice, once with
lr_weight forced to 0, which zeroes E_lr before the autograd that produces the
long-range force. So the difference is exactly the weighted long-range term, and
the two E_lr estimates must agree - a cheap self-check that lr_weight=0 removes
the long-range part and nothing else.

The cace-side counterparts of these numbers come from cace/inspect_cace_les.py.

Usage:
    python diag_les.py                    # every deepmd arm with a checkpoint
    python diag_les.py hyb_lrw1           # one arm
    python diag_les.py --ckpt 95000       # a specific checkpoint step
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from metrics import (  # noqa: E402 - path is set above
    DEEPMD_ARMS,
    DEEPMD_RUNS,
    late_checkpoints,
    load_deepmd,
    load_valid,
    resolve_device,
)


def charge_layer(model):
    """(mode name, the effective lr_weight) of a loaded arm."""
    les = model.atomic_model.les_model
    weight = float(getattr(model, "lr_weight", 1.0))
    if getattr(les, "is_freeze_mode", False):
        return "freeze_charge", weight
    if getattr(les, "is_local_mode", False):
        return "local_charge", weight
    return "?", weight


def forward(model, coord, atype, box, dtype, device):
    return model(
        torch.tensor(coord, device=device, dtype=dtype),
        torch.tensor(atype, device=device),
        box=torch.tensor(box, device=device, dtype=dtype),
    )


def run_arm(run_dir, device, batch, ckpt=None):
    coord, atype, box, E_ref, F_ref, nat = load_valid()
    model, step = load_deepmd(run_dir, device, ckpt=ckpt)
    if model is None:
        return None
    if not hasattr(model.atomic_model, "les_model"):
        return {"arm": os.path.basename(run_dir), "skip": "no LES (short-range only)"}
    mode, lr_weight = charge_layer(model)
    dtype = next(model.parameters()).dtype
    nf = coord.shape[0]

    captured = {}

    def hook(_module, _inp, out):
        captured["E_lr"] = out["E_lr"].detach()
        captured["q"] = out["latent_charges"].detach()

    handle = model.atomic_model.les_model.register_forward_hook(hook)

    E_tot = np.zeros(nf)
    E_sr = np.zeros(nf)
    E_lr_hook = np.zeros(nf)
    E_lr_neu = np.zeros(nf)
    F_tot = np.zeros((nf, nat, 3))
    F_sr = np.zeros((nf, nat, 3))
    Q_net = np.zeros(nf)
    q_blocks = []
    neu_error = None
    les = model.atomic_model.les_model
    for lo in range(0, nf, batch):
        hi = min(lo + batch, nf)
        nfb = hi - lo
        c, a, b = coord[lo:hi], atype[lo:hi], box[lo:hi]
        out = forward(model, c, a, b, dtype, device)
        E_tot[lo:hi] = out["energy"].detach().reshape(nfb).double().cpu().numpy()
        F_tot[lo:hi] = out["force"].detach().double().cpu().numpy()
        E_lr_hook[lo:hi] = lr_weight * captured["E_lr"].double().cpu().numpy()
        qb = captured["q"].double().cpu().numpy().reshape(nfb, nat)
        q_blocks.append(qb)
        Q_net[lo:hi] = qb.sum(axis=1)

        w = model.lr_weight
        model.lr_weight = 0.0
        out0 = forward(model, c, a, b, dtype, device)
        model.lr_weight = w
        E_sr[lo:hi] = out0["energy"].detach().reshape(nfb).double().cpu().numpy()
        F_sr[lo:hi] = out0["force"].detach().double().cpu().numpy()

        # The same charges with each frame's mean charge removed, fed straight to
        # the kernel. Neutralising cannot be done through the charge layer, so
        # bypass it: the Les module uses latent_charges verbatim when given. Must
        # run after the captures above - this call fires the hook too.
        if neu_error is None:
            try:
                qn = torch.tensor(qb - qb.mean(axis=1, keepdims=True),
                                  device=device, dtype=dtype)
                outn = les(
                    positions=torch.tensor(c, device=device,
                                           dtype=dtype).reshape(-1, 3),
                    cell=torch.tensor(b, device=device,
                                      dtype=dtype).reshape(nfb, 3, 3),
                    batch=torch.arange(nfb, device=device).repeat_interleave(nat),
                    latent_charges=qn.reshape(-1),
                    compute_energy=True,
                )
                E_lr_neu[lo:hi] = lr_weight * (
                    outn["E_lr"].double().cpu().numpy().reshape(nfb))
                del outn
            except Exception as exc:  # noqa: BLE001 - report, do not lose the arm
                neu_error = f"{type(exc).__name__}: {exc}"
        del out, out0
    handle.remove()

    E_lr = E_tot - E_sr
    hook_gap = float(np.abs(E_lr - E_lr_hook).max()) if nf else 0.0
    q = np.concatenate(q_blocks, axis=0)  # [nf, nat]
    F_lr = F_tot - F_sr

    ref = float(E_ref.std())  # per-frame target spread, eV
    return {
        "arm": os.path.basename(run_dir),
        "step": step,
        "mode": mode,
        "lr_weight": lr_weight,
        "E_lr": E_lr,
        "E_lr_neu": E_lr_neu,
        "neu_error": neu_error,
        "E_sr": E_sr,
        "E_tot": E_tot,
        "q": q,
        "atype": atype[0],
        "Q_net": Q_net,
        "F_lr": F_lr,
        "F_tot": F_tot,
        "ref_std": ref,
        "hook_gap": hook_gap,
    }


def report(r):
    if r is None:
        return
    if "skip" in r:
        print(f"\n=== {r['arm']}: {r['skip']}")
        return
    e, q, f_lr, f_tot = r["E_lr"], r["q"], r["F_lr"], r["F_tot"]
    sr = r["E_sr"]
    ref = r["ref_std"]
    print(f"\n=== {r['arm']}  (step {r['step']}, {r['mode']}, "
          f"lr_weight {r['lr_weight']})")
    print(f"  E_lr    mean {e.mean():+.4f}  std {e.std():.4f}  "
          f"range {e.min():+.3f}..{e.max():+.3f} eV/frame")
    print(f"  E_lr std / target std = {e.std() / ref:.4f}   (target std "
          f"{ref:.4f} eV/frame)")
    print(f"  E_lr std / |E_tot| mean = "
          f"{e.std() / max(abs(r['E_tot']).mean(), 1e-12):.4e}")
    ne = r["E_lr_neu"]
    if r["neu_error"] is not None:
        print(f"  E_lr neutralized: unavailable ({r['neu_error']})")
    else:
        # Same charges with each frame's mean removed. A neutral frame carries no
        # Q^2 term, so whatever survives here is the Coulomb part; the drop in
        # variance is the drift share.
        drift = 1.0 - ne.var() / max(e.var(), 1e-30)
        print(f"  E_lr neutralized (per-frame mean charge removed): "
              f"mean {ne.mean():+.4f}  std {ne.std():.4f} eV/frame")
        print(f"  drift share of E_lr variance = {drift:+.4f}   "
              f"(0 = pure Coulomb, 1 = all charge-drift artifact)")
    print(f"  E_sr    mean {sr.mean():+.4f}  std {sr.std():.4f}  "
          f"std / target std = {sr.std() / ref:.4f}   "
          f"corr(E_sr, E_lr) = {np.corrcoef(sr, e)[0, 1]:+.4f}")
    print(f"  F_lr RMS {np.sqrt(np.mean(f_lr ** 2)):.4e} eV/A   "
          f"F_tot RMS {np.sqrt(np.mean(f_tot ** 2)):.4e} eV/A   "
          f"ratio {np.sqrt(np.mean(f_lr ** 2)) / np.sqrt(np.mean(f_tot ** 2)):.3e}")
    print("  q per type:")
    for t in np.unique(r["atype"]):
        qt = q[:, r["atype"] == t]
        print(f"    type {int(t)}: n={qt.shape[1]:3d}  mean {qt.mean():+.4f}  "
              f"std {qt.std():.4f}  min {qt.min():+.3f}  max {qt.max():+.3f}")
    print(f"  |q| max {np.abs(q).max():.4f}")
    print(f"  net charge |Q| per frame: mean {np.abs(r['Q_net']).mean():.4e}  "
          f"max {np.abs(r['Q_net']).max():.4e}")
    print(f"  self-check |E_lr(diff) - E_lr(hook)| max = {r['hook_gap']:.3e} eV")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ckpt", type=int, default=None,
                    help="score this exact checkpoint step instead of the latest")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    arms = args.arms or DEEPMD_ARMS
    unknown = [a for a in arms if a not in DEEPMD_ARMS]
    if unknown:
        ap.error(f"unknown arm(s) {unknown}; choose from {DEEPMD_ARMS}")
    device = resolve_device(args.device)

    for arm in arms:
        run_dir = os.path.join(DEEPMD_RUNS, arm)
        ckpt = None
        if args.ckpt is not None:
            cand = os.path.join(run_dir, f"model.ckpt-{args.ckpt}.pt")
            if not os.path.exists(cand):
                print(f"\n=== {arm}: no checkpoint at step {args.ckpt}")
                continue
            ckpt = cand
        elif not late_checkpoints(run_dir, 1):
            print(f"\n=== {arm}: no checkpoint in {run_dir}")
            continue
        try:
            report(run_arm(run_dir, device, args.batch, ckpt=ckpt))
        except Exception as exc:  # noqa: BLE001 - one bad arm must not hide the rest
            print(f"\n=== {arm}: FAILED {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
