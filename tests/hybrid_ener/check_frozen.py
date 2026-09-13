"""Freeze check: torch.jit.script must not change the model's numerics.

`dp freeze` scripts the model with torch.jit.script and saves the result, so the
scripted module must produce the same numbers as the eager model. This script
scripts a hybrid_ener model (the same call dp freeze makes) and compares
energy / force / virial / atom_virial for one frame, run on CPU so that scripting
and inference are self-contained.

Usage: python check_frozen.py [ckpt]
"""
import sys

import torch

from _common import build_model, build_water, load_checkpoint

KEYS = ["energy", "force", "virial", "atom_virial"]
TOL = 1e-9


def main():
    coord, atype, box = build_water(device="cpu")
    if len(sys.argv) > 1:
        model = load_checkpoint(sys.argv[1], device="cpu")
    else:
        model = build_model(device="cpu")
    model.eval()

    try:
        scripted = torch.jit.script(model)
    except Exception as e:
        print("RESULT: FAIL - torch.jit.script raised", type(e).__name__)
        print(e)
        return 1
    print("torch.jit.script ->", type(scripted).__name__)

    eager = model(coord.clone(), atype, box.clone(), do_atomic_virial=True)
    frozen = scripted(coord.clone(), atype, box.clone(), do_atomic_virial=True)

    worst = 0.0
    for key in KEYS:
        a = eager[key].detach()
        b = frozen[key].detach()
        if a.shape != b.shape:
            print(f"{key:12s} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
            return 1
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        print(f"{key:12s} max|eager - frozen| = {d:.3e}   max|eager| = {a.abs().max().item():.6e}")

    ok = worst < TOL
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
