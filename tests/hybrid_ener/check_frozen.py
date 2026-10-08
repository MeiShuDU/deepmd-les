"""Freeze check: torch.jit.script must not change the model's numerics.

`dp freeze` scripts the model with torch.jit.script and saves the result, so the
scripted module must produce the same numbers as the eager model. This script
scripts a hybrid_ener model (the same call dp freeze makes) and compares
energy / force / virial / atom_virial for one frame, run on CPU so that scripting
and inference are self-contained.

Every charge-layer mode is scripted, because each one takes a different branch in
forward (local: the Atomwise NN; freeze: the constant table). A mode that only
ever ran eagerly would hide a compile error until someone tried to freeze that
run.

Usage: python check_frozen.py [ckpt]
"""
import sys

import torch

from _common import build_model, build_water, load_checkpoint

KEYS = ["energy", "force", "virial", "atom_virial"]
TOL = 1e-9

MODES = [
    ("local", None),  # BASE_CONFIG default: use_atomwise
    ("local+fixed", {"use_atomwise": True, "use_fixed_charges": True}),
    ("local+guess+neutral", {"use_atomwise": True, "initial_guess": [-0.82, 0.41],
                             "claim_neutral": True}),
    ("freeze", {"freeze_charge": [-0.82, 0.41]}),
]


def compare(model, coord, atype, box, label):
    """Script `model` and report the largest eager-vs-scripted difference."""
    try:
        scripted = torch.jit.script(model)
    except Exception as e:  # noqa: BLE001 - report the compile error, keep going
        return None, f"{label}: torch.jit.script raised {type(e).__name__}: {e}"

    eager = model(coord.clone(), atype, box.clone(), do_atomic_virial=True)
    frozen = scripted(coord.clone(), atype, box.clone(), do_atomic_virial=True)

    worst = 0.0
    detail = []
    for key in KEYS:
        a = eager[key].detach()
        b = frozen[key].detach()
        if a.shape != b.shape:
            return None, f"{label}: {key} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}"
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        detail.append(f"{key}={d:.1e}")
    return worst, f"{label}: " + " ".join(detail)


def main():
    coord, atype, box = build_water(device="cpu")

    if len(sys.argv) > 1:
        model = load_checkpoint(sys.argv[1], device="cpu")
        model.eval()
        worst, detail = compare(model, coord, atype, box, "ckpt")
        print(detail)
        print("RESULT:", "PASS" if worst is not None and worst < TOL else "FAIL")
        return 0 if worst is not None and worst < TOL else 1

    worst_overall = 0.0
    for label, overrides in MODES:
        model = build_model(device="cpu", les_overrides=overrides)
        worst, detail = compare(model, coord, atype, box, label)
        print(detail)
        if worst is None:
            print("RESULT: FAIL")
            return 1
        worst_overall = max(worst_overall, worst)

    ok = worst_overall < TOL
    print(f"worst eager-vs-scripted difference over {len(MODES)} modes: "
          f"{worst_overall:.3e} (tol {TOL:.0e})")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
