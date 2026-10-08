"""Charges and Born effective charges of one trained arm, as the kernel sees it.

``check_bec.py`` reports the BEC of a checkpoint's *latent charge proposal*
(``q``, the head's output). For a plain ``sea-lr`` arm that proposal is exactly
what the ``EwaldPotential`` consumes, so the distinction does not arise. For an
arm with ``charge_eq_latent`` the kernel consumes the *equalized* charge
(``q_eq``) instead, and the proposal ``q`` is only an intermediate.

This script reports both, with every convention of ``check_bec.py`` reused
unchanged (raw / neutralized / frozen, dephased periodic and direct sum), so an
eq arm and a plain arm can be read off the same table.

Usage:
    python check_bec_arm.py --ckpt <...>/sea-lr-eq_sA/best_model.pth --frames 5
    python check_bec_arm.py --ckpt <...>/sea-lr_sA/best_model.pth --frames 5
"""
import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("DEEPMD_CACE_DEVICE", "cpu")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check_bec as CB  # noqa: E402


def eq_module_of(model):
    from deepmd_cace.charge_eq_latent import ChargeEqLatent

    for module in model.modules():
        if isinstance(module, ChargeEqLatent):
            return module
    return None


def kernel_charges(model, head, pos, cell, batch, numbers):
    """The charge set the long-range kernel consumes, connected to ``pos``.

    Runs the input modules and the charge head, then the equalizer when the
    checkpoint carries one, so the returned tensor is ``q_eq`` for an eq arm and
    the head's ``q`` for a plain one.
    """
    data = {
        "positions": pos,
        "cell": cell,
        "batch": batch,
        "atomic_numbers": numbers,
    }
    for module in model.input_modules:
        data = module(data, compute_stress=False, compute_virials=False)
    data = head(data, training=False)
    raw = data["q"].sum(dim=1, keepdim=True)
    eq_mod = eq_module_of(model)
    if eq_mod is None:
        return raw, raw
    data = eq_mod(data, training=True)
    return raw, data["q_eq"].sum(dim=1, keepdim=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default=CB.DEFAULT_DATA)
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    model = CB.load_model(args.ckpt)
    head = CB.charge_head_of(model)
    elem, numbers = CB.read_system(args.data)
    is_o, is_h = elem == "O", elem == "H"
    dtype = CB.model_dtype(model)
    numbers = torch.tensor(numbers)

    has_eq = eq_module_of(model) is not None
    print(f"checkpoint : {os.path.relpath(args.ckpt)}")
    print(f"equalizer  : {'ChargeEqLatent' if has_eq else 'none (plain EwaldPotential)'}")
    if has_eq:
        eq = eq_module_of(model)
        print(f"             w = {eq.regularization_weight}  "
              f"remove_self_interaction = {eq.ep.remove_self_interaction}")
    print(f"atoms      : {len(elem)}  (O {int(is_o.sum())}, H {int(is_h.sum())})")

    # columns: (charge set, convention). raw = the kernel's charges; pro = proposal.
    cols = ["pro_PBC", "pro_dir", "ker_PBC", "ker_dir", "ker_frz_PBC", "ker_frz_dir"]
    rows = {k: [] for k in cols}
    charge_stats = {k: [] for k in ("q_pro", "q_ker", "net_pro", "net_ker")}
    for index in range(args.frames):
        coord, box = CB.frame_tensors(args.data, index, dtype)
        n = coord.shape[0]
        pos = torch.tensor(coord, dtype=dtype, requires_grad=True)
        cell = torch.tensor(box, dtype=dtype).reshape(1, 3, 3)
        batch = torch.zeros(n, dtype=torch.int64)

        q_pro, q_ker = kernel_charges(model, head, pos, cell, batch, numbers)
        rows["pro_PBC"].append(CB.summarise(CB.born_charges(pos, cell, batch, q_pro, True), is_o, is_h))
        rows["pro_dir"].append(CB.summarise(CB.born_charges(pos, cell, batch, q_pro, False), is_o, is_h))
        rows["ker_PBC"].append(CB.summarise(CB.born_charges(pos, cell, batch, q_ker, True), is_o, is_h))
        rows["ker_dir"].append(CB.summarise(CB.born_charges(pos, cell, batch, q_ker, False), is_o, is_h))
        rows["ker_frz_PBC"].append(CB.summarise(CB.born_charges(pos, cell, batch, q_ker.detach(), True), is_o, is_h))
        rows["ker_frz_dir"].append(CB.summarise(CB.born_charges(pos, cell, batch, q_ker.detach(), False), is_o, is_h))

        u = CB.CHARGE_UNIT
        charge_stats["q_pro"].append(
            (float(q_pro[is_o].mean()) * u, float(q_pro[is_h].mean()) * u))
        charge_stats["q_ker"].append(
            (float(q_ker[is_o].mean()) * u, float(q_ker[is_h].mean()) * u))
        charge_stats["net_pro"].append(float(q_pro.sum()) * u)
        charge_stats["net_ker"].append(float(q_ker.sum()) * u)

    u = CB.CHARGE_UNIT
    qp = np.array(charge_stats["q_pro"])
    qk = np.array(charge_stats["q_ker"])
    npro = np.array(charge_stats["net_pro"])
    nker = np.array(charge_stats["net_ker"])
    print(f"\ncharges in e (mean over {args.frames} frames, pm std over frames):")
    print("  proposal  q_O = %+.4f +- %.4f   q_H = %+.4f +- %.4f   net/frame = %+.4f"
          % (qp[:, 0].mean(), qp[:, 0].std(), qp[:, 1].mean(), qp[:, 1].std(), npro.mean()))
    print("  kernel    q_O = %+.4f +- %.4f   q_H = %+.4f +- %.4f   net/frame = %+.4f"
          % (qk[:, 0].mean(), qk[:, 0].std(), qk[:, 1].mean(), qk[:, 1].std(), nker.mean()))
    if has_eq:
        move = np.abs(qk - qp).max() / max(np.abs(qp).max(), 1e-12)
        print("  max per-element |q_eq - q_r| / max|q_r| = %.3f" % move)

    def table(title, keys):
        print(f"\n{title}  (mean over {args.frames} frames):")
        for label, key in keys:
            o = np.array([r["o"] for r in rows[key]])
            hh = np.array([r["h"] for r in rows[key]])
            mx = np.array([r["max"] for r in rows[key]])
            sr = np.array([r["sum_rule"] for r in rows[key]])
            print("  %-34s O = %+8.4f   H = %+8.4f   max|Z*| = %7.2f   sum_i Z* = %.2e"
                  % (label, o.mean(), hh.mean(), mx.max(), sr.max()))

    table(
        "Born effective charges, Z*_iso in e, dephased periodic",
        [
            ("proposal q_r", "pro_PBC"),
            ("kernel charges (q_eq)", "ker_PBC"),
            ("kernel charges, frozen", "ker_frz_PBC"),
        ],
    )
    table(
        "Born effective charges, Z*_iso in e, direct sum (no dephasing)",
        [
            ("proposal q_r", "pro_dir"),
            ("kernel charges (q_eq)", "ker_dir"),
            ("kernel charges, frozen", "ker_frz_dir"),
        ],
    )


if __name__ == "__main__":
    main()