"""What latent-charge equalization does to the Born effective charges.

``check_bec.py`` reports the Born charges of a ``deepmd_cace`` checkpoint as
trained. This script reports them again with ``ChargeEqLatent`` applied to the
latent charges at inference time:

    q_r  ->  q_eq  =  argmin 1/2 q^T A q + w ||q - q_r||^2   s.t.  1^T q = 0

The model's own parameters are untouched (``ChargeEqLatent`` is parameter-free,
so the same state dict loads), and every convention in ``check_bec.py`` is
reused unchanged - only the charges that enter ``Polarization`` are swapped from
``q`` to ``q_eq``.

This is an operator-level diagnostic, NOT the BEC of a trained ``sea-lr-eq``
run: the charge head here was trained without the equalization term, so its
proposal ``q_r`` still carries the non-neutral drift the equalizer then removes.
A model trained with the term in the loss would put its proposal somewhere else
and the numbers below would move. What the script measures exactly is the
mapping ``BEC(q_r) -> BEC(q_eq)`` at fixed weights.

Usage:
    python check_bec_eq.py --ckpt <...>/sea-lr_sA/best_model.pth --w 10 --frames 5
"""
import argparse
import copy
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("DEEPMD_CACE_DEVICE", "cpu")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check_bec as CB  # noqa: E402


def build_eq_model(ckpt, w):
    """Same checkpoint, with ``ChargeEqLatent`` swapped in for ``EwaldPotential``."""
    from deepmd.pt.model.descriptor.se_a import DescrptSeA
    from deepmd_cace.model import build_model

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(blob["config"])
    cfg["model"]["long_range"]["charge_eq_latent"] = {
        "enabled": True,
        "regularization_weight": w,
        "total_charge": 0.0,
    }
    desc = DescrptSeA(
        **{k: v for k, v in cfg["model"]["descriptor"].items() if k != "type"}
    )
    model = build_model(cfg["model"], desc, "cpu")
    model.load_state_dict(blob["state_dict"], strict=True)
    return model.eval(), cfg


def eq_module_of(model):
    from deepmd_cace.charge_eq_latent import ChargeEqLatent

    return next(m for m in model.modules() if isinstance(m, ChargeEqLatent))


def charges_eq(model, head, eq_mod, pos, cell, batch, numbers):
    """Channel-summed per-atom charge after equalization, connected to ``pos``."""
    data = {
        "positions": pos,
        "cell": cell,
        "batch": batch,
        "atomic_numbers": numbers,
    }
    for module in model.input_modules:
        data = module(data, compute_stress=False, compute_virials=False)
    data = head(data, training=False)
    data = eq_mod(data, training=True)
    return data["q_eq"].sum(dim=1, keepdim=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CB.DEFAULT_CKPT)
    parser.add_argument("--data", default=CB.DEFAULT_DATA)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--w", type=float, default=10.0,
                        help="regularization_weight of ChargeEqLatent")
    args = parser.parse_args()

    model, cfg = build_eq_model(args.ckpt, args.w)
    head = CB.charge_head_of(model)
    eq_mod = eq_module_of(model)
    elem, numbers = CB.read_system(args.data)
    is_o, is_h = elem == "O", elem == "H"
    dtype = CB.model_dtype(model)
    numbers = torch.tensor(numbers)

    # the same model without equalization, for the side-by-side column
    plain = CB.load_model(args.ckpt)
    plain_head = CB.charge_head_of(plain)

    print(f"checkpoint : {os.path.relpath(args.ckpt)}")
    print(f"w          : {args.w}")
    print(f"atoms      : {len(elem)}  (O {int(is_o.sum())}, H {int(is_h.sum())})")

    rows = {k: [] for k in ("raw", "eq_raw", "eq_frozen")}
    dup = {k: [] for k in ("raw", "eq_raw", "eq_frozen")}
    for index in range(args.frames):
        coord, box = CB.frame_tensors(args.data, index, dtype)
        n = coord.shape[0]
        # one graph for both raw and equalized charges: same leaf tensor
        pos = torch.tensor(coord, dtype=dtype, requires_grad=True)
        cell = torch.tensor(box, dtype=dtype).reshape(1, 3, 3)
        batch = torch.zeros(n, dtype=torch.int64)

        q_raw = CB.charges(plain, plain_head, pos, cell, batch, numbers)
        q_eq = charges_eq(model, head, eq_mod, pos, cell, batch, numbers)

        rows["raw"].append(CB.summarise(CB.born_charges(pos, cell, batch, q_raw, True), is_o, is_h))
        rows["eq_raw"].append(CB.summarise(CB.born_charges(pos, cell, batch, q_eq, True), is_o, is_h))
        rows["eq_frozen"].append(CB.summarise(CB.born_charges(pos, cell, batch, q_eq.detach(), True), is_o, is_h))
        dup["raw"].append(CB.summarise(CB.born_charges(pos, cell, batch, q_raw, False), is_o, is_h))
        dup["eq_raw"].append(CB.summarise(CB.born_charges(pos, cell, batch, q_eq, False), is_o, is_h))
        dup["eq_frozen"].append(CB.summarise(CB.born_charges(pos, cell, batch, q_eq.detach(), False), is_o, is_h))

        if index == 0:
            u = CB.CHARGE_UNIT
            print(
                "charges (e), per frame:  raw net = %+.3f  ->  eq net = %+.3f"
                % (float(q_raw.sum()) * u, float(q_eq.sum()) * u)
            )
            print(
                "                          q_O raw %+.4f -> eq %+.4f    q_H raw %+.4f -> eq %+.4f"
                % (
                    float(q_raw[is_o].mean()) * u,
                    float(q_eq[is_o].mean()) * u,
                    float(q_raw[is_h].mean()) * u,
                    float(q_eq[is_h].mean()) * u,
                )
            )
            move = (q_eq - q_raw).norm() / q_raw.norm()
            print("rms movement ||q_eq - q_r|| / ||q_r|| = %.4f" % float(move))

    def table(rows, title):
        print(f"\n{title}  (mean over {args.frames} frames):")
        for key, label in (
            ("raw", "PBC dephased, raw q_r   "),
            ("eq_raw", "PBC dephased, q_eq       "),
            ("eq_frozen", "PBC dephased, q_eq frozen"),
        ):
            o = np.array([r["o"] for r in rows[key]])
            hh = np.array([r["h"] for r in rows[key]])
            sr = np.array([r["sum_rule"] for r in rows[key]])
            print(
                "  %-25s O = %+.4f   H = %+.4f   max|Z*| = %.1f   sum_i Z* max = %.2e"
                % (label, o.mean(), hh.mean(),
                   np.array([r["max"] for r in rows[key]]).max(), sr.max())
            )

    table(rows, "Born effective charges, Z*_iso in e, dephased periodic")
    table(dup, "Born effective charges, Z*_iso in e, direct sum (no dephasing)")


if __name__ == "__main__":
    main()