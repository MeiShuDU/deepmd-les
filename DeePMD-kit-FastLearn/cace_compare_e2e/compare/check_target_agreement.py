#!/usr/bin/env python
# coding: utf-8
"""Verify the cace and deepmd sides share the same validation frames and targets.

Both sides are supposed to come from one prep over cace/water.xyz: the same
90/10 split (valid_fraction 0.1, seed 1) and the same residual target
E - sum_Z ref[Z]. If they ever diverge, every metric comparison built on them is
void, so this is checked explicitly rather than assumed.

Run from this directory with either interpreter (cace is installed in both).
"""
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.dirname(HERE)
CACE_DIR = os.path.join(E2E, "cace")
WATER = os.path.join(E2E, "water")

import cace
from cace.tasks.load_data import get_dataset_from_xyz, load_data_loader

cace.tools.setup_logger(level="WARNING")
torch.set_default_dtype(torch.float32)

REF = {1: -187.6043857100553, 8: -93.80219285502734}
SPLIT = dict(valid_fraction=0.1, seed=1, cutoff=5.5)


def cace_valid():
    col = get_dataset_from_xyz(
        train_path=os.path.join(CACE_DIR, "water.xyz"),
        data_key={"energy": "energy", "forces": "force"},
        atomic_energies=REF,
        **SPLIT,
    )
    dl = load_data_loader(col, "valid", 1)
    e, pos, force = [], [], []
    for b in dl:
        bd = b.to_dict()
        e.append(float(bd["energy"][0]))
        pos.append(bd["positions"].detach().numpy())
        force.append(bd["forces"].detach().numpy())
    return np.array(e), pos, force


def main():
    e_cace, pos_cace, force_cace = cace_valid()
    base = os.path.join(WATER, "valid", "set.000")
    e_dp = np.load(os.path.join(base, "energy.npy"))
    coord_dp = np.load(os.path.join(base, "coord.npy"))
    force_dp = np.load(os.path.join(base, "force.npy"))
    nat = coord_dp.shape[1] // 3

    print(f"cace   valid frames = {len(e_cace)}")
    print(f"deepmd valid frames = {len(e_dp)}  (natoms {nat})")
    if len(e_cace) != len(e_dp):
        print("RESULT: FAIL - valid frame count differs")
        return 1

    dE = np.abs(e_cace - e_dp)
    print(f"energy   max|diff| = {dE.max():.3e} eV   (scale {np.abs(e_dp).max():.3e})")

    dpos = max(
        float(np.abs(p - coord_dp[k].reshape(nat, 3)).max())
        for k, p in enumerate(pos_cace)
    )
    dfrc = max(
        float(np.abs(f - force_dp[k].reshape(nat, 3)).max())
        for k, f in enumerate(force_cace)
    )
    print(f"coords   max|diff| = {dpos:.3e} A")
    print(f"forces   max|diff| = {dfrc:.3e} eV/A")

    ok = dE.max() < 1e-5 and dpos < 1e-5 and dfrc < 1e-4
    print(
        "RESULT:",
        "PASS - same frames, same targets, same order"
        if ok
        else "FAIL - the two sides disagree",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
