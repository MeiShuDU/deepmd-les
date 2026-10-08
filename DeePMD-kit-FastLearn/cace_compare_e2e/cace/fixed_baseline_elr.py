#!/usr/bin/env python
# coding: utf-8
"""Magnitude of the fixed-baseline LES energy for the water benchmark.

Answers the design question: if deepmd (and our cace LesFixed path) add E_LR at
weight 1.0 with the shared fixed-baseline (O -1.0, H +0.5) at dl=2, sigma=1,
how large is the LR term relative to the residual target?

Uses the same les.Les path both codes wrap: latent_charges = zeros (no learned
offset), so E_lr is purely the fixed-charge Ewald sum.
"""
import os
import numpy as np
import torch

from les import Les

HERE = os.path.abspath(os.path.dirname(__file__))
VALID = os.path.join(HERE, "..", "water", "valid")

LES_ARGS = {
    "use_atomwise": False,
    "use_fixed_atomic_charges": True,
    "fixed_atomic_charges_scaling_factor": 0.5,
    "sigma": 1.0,
    "dl": 2.0,
    "verbose": False,
}

les = Les(les_arguments=LES_ARGS)
les = les.double()

coord = np.load(os.path.join(VALID, "set.000", "coord.npy"))
box = np.load(os.path.join(VALID, "set.000", "box.npy"))
e_res = np.load(os.path.join(VALID, "set.000", "energy.npy"))
atype = np.loadtxt(os.path.join(VALID, "type.raw")).astype(np.int64)
nat = atype.shape[0]

# atomic numbers, same order as deepmd type_map O=8, H=1
zmap = np.array([8, 1])[atype]

Elr, E_use = [], []
for f in range(8):
    pos = torch.from_numpy(coord[f * nat:(f + 1) * nat]).double().requires_grad_(True)
    cell = torch.from_numpy(box[f]).reshape(1, 3, 3).double()
    z = torch.from_numpy(zmap).long()
    latent0 = torch.zeros(nat, dtype=torch.double)
    batch = torch.zeros(nat, dtype=torch.long)
    out = les(
        positions=pos, cell=cell, latent_charges=latent0,
        batch=batch, compute_energy=True, atomic_numbers=z,
    )
    Elr.append(float(out["E_lr"].item()))
    E_use.append(float(e_res[f]))

Elr = np.array(Elr)
E_use = np.array(E_use)
print(f"valid frames (first 8):")
print(f"  fixed-baseline E_lr : min={Elr.min():+.4f} max={Elr.max():+.4f} "
      f"mean={Elr.mean():+.4f} std={Elr.std():.4f} eV/frame")
print(f"  residual E_ref      : min={E_use.min():+.4f} max={E_use.max():+.4f} "
      f"mean={E_use.mean():+.4f} eV/frame")
print(f"  |E_lr / E_ref| mean = {np.abs(Elr / E_use).mean():.3f}")
print(f"  per H2O: E_lr = {Elr.mean()/64:+.6f} eV/mol, "
      f"E_ref = {E_use.mean()/64:+.6f} eV/mol")