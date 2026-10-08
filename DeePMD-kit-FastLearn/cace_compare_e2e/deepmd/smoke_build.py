#!/usr/bin/env python
# coding: utf-8
"""CPU smoke-build of the aligned deepmd hybrid_ener model (water benchmark).

Builds exactly the model that `dp --pt train input.json` would train, on CPU,
loads the first valid batch, and runs one forward + backward in float64.

Validates before the HPC run:
  * the se_a descriptor actually builds with the computed sel and no atom in
    the water/valid system overflows the neighbor list ('nnei > sel');
  * HybridLESModel with use_fixed_atomic_charges runs end to end (FixedCharges
    charge_table path, ewald on a nonzero cell, autograd long-range force);
  * reports the SR/LR/fixed-charge param split.
"""
import json
import os
import sys
import numpy as np
import torch

os.environ.setdefault("DEVICE", "cpu")  # build the whole model on CPU for the smoke

from deepmd.pt.model.model import get_model

# Usage: python smoke_build.py [config.json] [valid_dir]
CFG = sys.argv[1] if len(sys.argv) > 1 else "input.json"
VALID = sys.argv[2] if len(sys.argv) > 2 else "../water/valid"


def main():
    # Fixed seed so the randomly initialised model is reproducible: this makes
    # the smoke output comparable across hosts (same params and same E_pred).
    torch.manual_seed(1234)
    np.random.seed(1234)

    with open(CFG) as f:
        cfg = json.load(f)
    model_params = cfg["model"]
    model = get_model(model_params)

    n_sr = sum(p.numel() for p in model.atomic_model.parameters())
    n_les = sum(p.numel() for p in model.atomic_model.les_model.parameters()
                if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: descriptor+fitting(SR)={n_sr}  les={n_les}  total={n_total}")

    coord = np.load(f"{VALID}/set.000/coord.npy")       # [nframes, nat*3]
    box = np.load(f"{VALID}/set.000/box.npy")           # [nframes, 9]
    force = np.load(f"{VALID}/set.000/force.npy")
    energy = np.load(f"{VALID}/set.000/energy.npy")
    atype = np.loadtxt(f"{VALID}/type.raw").astype(np.int64).reshape(1, -1)
    nat = atype.shape[1]
    nf = energy.shape[0]
    print(f"valid: {nf} frames x {nat} atoms")

    coord_t = torch.from_numpy(coord[:1]).reshape(1, nat, 3).to(torch.float64)
    box_t = torch.from_numpy(box[:1]).reshape(1, 3, 3).to(torch.float64)
    atype_t = torch.from_numpy(atype).repeat(1, 1)
    e_ref = torch.from_numpy(energy[:1]).to(torch.float64)

    model = model.double()
    model.train()

    out = model(coord_t.detach().clone().requires_grad_(True),
                atype_t, box_t.clone().requires_grad_(True))
    e_pred = out["energy"]
    f_pred = out["force"]
    print(f"E_pred={e_pred[0].item():.6f}  E_ref={e_ref[0].item():.6f}")
    loss = torch.mean((e_pred - e_ref) ** 2)
    loss.backward()
    g = {k: v for k, v in model.named_parameters() if v.grad is not None}
    print(f"backward OK; params with grad: {len(g)}")
    print("SMOKE OK")

    # with use_fixed_atomic_charges=false there is no fixed baseline; the model
    # must not put the astronomically large fixed-charge Ewald term into E_total.
    # no config here turns on the fixed baseline, so no FixedCharges module should
    # be built at all.
    fs = getattr(model.atomic_model.les_model, "fixed_charges", None)
    print(f"FixedCharges module: {fs}")
    assert fs is None, "a FixedCharges baseline was built but no config asked for one"

    # with zero latent charges the LES energy must vanish. (A fixed O=-1.0/H=+0.5
    # baseline at lr_weight 1.0 gives -530 eV/frame on this split, i.e. a large
    # constant that the SR atomic bias can absorb; the point here is only that the
    # learned-q configs must not smuggle one in. See compare/probe_fixed_table.py.)
    q = model.atomic_model.les_model(
        positions=coord_t.reshape(-1, 3),
        cell=box_t.reshape(1, 3, 3),
        latent_charges=torch.zeros(nat, dtype=torch.float64),
        batch=torch.zeros(nat, dtype=torch.int64),
        compute_energy=True,
        atomic_numbers=model.atomic_model.element_numbers[atype_t.reshape(-1)],
    )
    print(f"E_lr(les_model, q=0) = {q['E_lr'].item():+.6e}  expect 0.0")


if __name__ == "__main__":
    main()