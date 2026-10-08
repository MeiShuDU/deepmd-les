#!/usr/bin/env python
# coding: utf-8
"""How big is a FIXED charge table's long-range term, really?

Decides whether a freeze_charge benchmark needs its weight fudged. The peak to
peak of E_lr is NOT the right measure: a constant per-frame E_lr is exactly
absorbable by the SR net's per-type atomic bias (the composition is fixed), so
only the frame-to-frame VARIATION competes with the target residual. A weight w
scales E_lr - and therefore its variation - by w.

Reads the held-out split (the same one metrics.py scores) and reports, for each
candidate table, the E_lr mean/variation and that variation as a multiple of the
target's own standard deviation, at several weights.

Usage: python probe_fixed_table.py
"""
import os

import numpy as np
import torch

from les import Les

HERE = os.path.dirname(os.path.abspath(__file__))
VALID = os.path.join(os.path.dirname(HERE), "water", "valid")

TABLES = {
    "O(-1.0)/H(+0.5)": {0: -1.0, 1: 0.5},
    "SPC/E(-0.8476/+0.4238)": {0: -0.8476, 1: 0.4238},
}
WEIGHTS = (1.0, 0.01, 1e-3)


def load_split():
    coord = np.load(f"{VALID}/set.000/coord.npy")
    box = np.load(f"{VALID}/set.000/box.npy")
    energy = np.load(f"{VALID}/set.000/energy.npy")
    atype = np.array([int(t) for t in open(f"{VALID}/type.raw").read().split()])
    nat = atype.size
    nf = coord.shape[0]
    return coord.reshape(nf, nat, 3), box.reshape(nf, 3, 3), energy, atype, nat, nf


def ewald_series(coord, box, atype, nat, nf, table):
    les = Les(les_arguments={"dim_descrpt": 8, "ntypes": 2, "element_numbers": [8, 1],
                             "sigma": 1.0, "dl": 1.5, "verbose": False})
    les = les.to(device="cpu", dtype=torch.float64)
    q = torch.tensor([table[t] for t in atype], dtype=torch.float64)
    out = np.empty(nf)
    for fr in range(nf):
        res = les(positions=torch.tensor(coord[fr], dtype=torch.float64),
                  cell=torch.tensor(box[fr], dtype=torch.float64).unsqueeze(0),
                  batch=torch.zeros(nat, dtype=torch.int64),
                  latent_charges=q, compute_energy=True)
        out[fr] = float(res["E_lr"].sum())
    return out


def main():
    coord, box, energy, atype, nat, nf = load_split()
    ref = energy.std()
    print(f"held-out split: {nf} frames x {nat} atoms")
    print(f"target residual: mean {energy.mean():+.4f}  std {ref:.4f}  "
          f"range {energy.min():+.3f}..{energy.max():+.3f} eV/frame")

    for name, table in TABLES.items():
        e = ewald_series(coord, box, atype, nat, nf, table)
        print(f"\n{name}: E_lr mean {e.mean():+.4f}  std {e.std():.4f}  "
              f"range {e.min():+.1f}..{e.max():+.1f} eV/frame")
        print("  weight     LR std (eV/frame)   x target std   verdict")
        for w in WEIGHTS:
            ratio = w * e.std() / ref
            verdict = ("dominant" if ratio > 1.0
                       else "meaningful" if ratio > 0.1
                       else "negligible")
            print(f"  {w:>7.0e}   {w * e.std():>16.4f}   {ratio:>12.4f}   {verdict}")
    print("\nthe mean is absorbed by the SR atomic bias; only the spread matters")


if __name__ == "__main__":
    main()
