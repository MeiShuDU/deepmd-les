"""Born effective charges of a ``deepmd_cace`` long-range checkpoint.

The ``deepmd_cace`` model has no BEC head: its long-range branch is CACE's
latent-charge ``Atomwise`` head plus CACE's ``EwaldPotential``. A Born charge is
therefore assembled from the charge head, using CACE's own
``Polarization`` -> ``Grad`` -> ``Dephase`` chain (the kernel that is
bit-identical to the ``les`` one).

Wiring rule (see ``les/module/bec.py``): the charges must be autograd-connected
to the *same* positions tensor that is handed to ``Grad``. A fresh forward, a
reshape or a slice in between silently drops the charge-response term and
leaves the frozen-charge value.

Three charge conventions are reported, because the checkpoint's charges are not
neutral and each convention answers a different question:

* ``raw``     - the model's charges exactly as the Ewald kernel sees them;
* ``neutral`` - charges shifted to zero mean (a uniform offset is a null mode
                of the periodic Ewald energy, so this removes an arbitrary
                gauge);
* ``frozen``  - ``q`` detached, i.e. the no-response limit ``q_i * I``.

The direct-sum value is the finite-box-robust one; the dephased periodic value
carries the dephasing correction ``T`` that violates the acoustic sum rule by
``~1/L^2`` for a charge-response model.

Usage:
    python check_bec.py                      # default checkpoint and data
    python check_bec.py --frames 5 --fd 3
"""
import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("DEEPMD_CACE_DEVICE", "cpu")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.join(
    HERE, "..", "DeePMD-kit-FastLearn", "campaign_water_interface"
)
DEFAULT_CKPT = os.path.join(
    CAMPAIGN, "deepmd", "runs", "sea-lr_sA", "best_model.pth"
)
DEFAULT_DATA = os.path.join(CAMPAIGN, "data", "water-interface", "valid")

# model internal charge units are scaled by sqrt(90.0474) for the norm_factor=1
# Ewald kernel; 1/9.48933 restores the physical electron charge.
CHARGE_UNIT = 1.0 / 9.48933


def load_model(path):
    sys.path.insert(0, HERE)
    from deepmd_cace.model import load_checkpoint

    return load_checkpoint(path, device="cpu").eval()


def charge_head_of(model):
    from cace.modules.atomwise import Atomwise

    return next(
        mod
        for mod in model.modules()
        if isinstance(mod, Atomwise) and getattr(mod, "output_key", None) == "tot_q"
    )


def model_dtype(model):
    """The dtype the model's graph runs in, i.e. the dtype to give positions.

    Not ``next(model.parameters()).dtype``: a checkpoint may mix precisions. The
    first parameter in iteration order belongs to the descriptor, and deepmd's
    ``DeepmdSeAInput`` casts its input to ``GLOBAL_PT_FLOAT_PRECISION`` on the
    way in but casts its output *back* to ``positions.dtype`` on the way out. So
    the dtype that has to match everything downstream - the charge head, cace's
    ``Polarization``/``Forces`` - is the head's. Feeding float64 positions to a
    float64-descriptor, float32-head model (``sea_seam.SE_A`` omits ``precision``,
    so that is what ``cace-sea-lr`` is) makes the head's first linear layer raise
    "expected m1 and m2 to have the same dtype, but got: double != float".
    """
    for parameter in charge_head_of(model).parameters():
        return parameter.dtype
    return next(model.parameters()).dtype


def read_system(data_dir):
    type_map = open(os.path.join(data_dir, "type_map.raw")).read().split()
    from ase.data import atomic_numbers as z_lut

    type_ids = np.loadtxt(os.path.join(data_dir, "type.raw"), dtype=int)
    elem = np.array([type_map[i] for i in type_ids])
    numbers = np.array([z_lut[type_map[i]] for i in type_ids], dtype=np.int64)
    return elem, numbers


def frame_tensors(data_dir, index, dtype):
    coord = np.load(os.path.join(data_dir, "set.000", "coord.npy"))[index]
    coord = coord.reshape(-1, 3).astype(np.float64)
    box = np.load(os.path.join(data_dir, "set.000", "box.npy"))[index]
    box = box.reshape(3, 3).astype(np.float64)
    return coord, box


def charges(model, head, pos, cell, batch, numbers):
    """Channel-summed per-atom charge, connected to ``pos``."""
    data = {
        "positions": pos,
        "cell": cell,
        "batch": batch,
        "atomic_numbers": numbers,
    }
    for module in model.input_modules:
        data = module(data, compute_stress=False, compute_virials=False)
    data = head(data, training=False)
    return data["q"].sum(dim=1, keepdim=True)


def born_charges(pos, cell, batch, q, pbc):
    """Z*[i, alpha, beta] = d P_alpha / d r_{i,beta} in physical e."""
    from cace.modules.grad import Grad
    from cace.modules.polarization import Dephase, Polarization

    data = {"positions": pos, "cell": cell, "batch": batch, "q": q}
    Polarization(
        charge_key="q",
        output_key="polarization",
        phase_key="phase",
        remove_mean=False,
        pbc=pbc,
        normalization_factor=CHARGE_UNIT,
    )(data, training=True)
    Grad(y_key="polarization", x_key="positions", output_key="grad_pol")(
        data, training=True
    )
    if pbc:
        Dephase(input_key="grad_pol", phase_key="phase", output_key="bec")(
            data, training=True
        )
    else:
        data["bec"] = data["grad_pol"]
    return data["bec"].transpose(1, 2).detach().numpy()


def isotropic(bec):
    return np.trace(bec, axis1=1, axis2=2) / 3.0


def summarise(bec, is_o, is_h):
    iso = isotropic(bec)
    return dict(
        o=float(iso[is_o].mean()),
        h=float(iso[is_h].mean()),
        max=float(np.abs(bec).max()),
        sum_rule=float(np.abs(bec.sum(0)).max()),
    )


def fixed_charge_sanity(model, head, pos, cell, batch, numbers, elem):
    """A neutral SPC/E charge set must give Z* = q * I through this pipeline."""
    spc = {"O": -0.8476, "H": 0.4238}
    q = torch.tensor(
        [[spc[e]] for e in elem], dtype=pos.dtype
    ) * (1.0 / CHARGE_UNIT)  # back to model units
    bec = born_charges(pos, cell, batch, q, pbc=False)
    diag_o = np.diagonal(bec[elem == "O"], axis1=1, axis2=2).mean(0)
    diag_h = np.diagonal(bec[elem == "H"], axis1=1, axis2=2).mean(0)
    off = np.abs(bec - bec * np.eye(3)).max()
    return diag_o, diag_h, off


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument(
        "--fd",
        type=int,
        default=0,
        help="finite-difference check on this many atoms of frame 0",
    )
    args = parser.parse_args()

    model = load_model(args.ckpt)
    head = charge_head_of(model)
    elem, numbers = read_system(args.data)
    is_o, is_h = elem == "O", elem == "H"
    dtype = model_dtype(model)
    numbers = torch.tensor(numbers)

    print(f"checkpoint: {os.path.relpath(args.ckpt)}")
    print(f"atoms: {len(elem)}  (O {int(is_o.sum())}, H {int(is_h.sum())})")

    rows = {k: [] for k in ("A_raw", "A_neu", "A_frozen", "B_raw", "B_neu")}
    spc_done = False
    for index in range(args.frames):
        coord, box = frame_tensors(args.data, index, dtype)
        n = coord.shape[0]
        pos = torch.tensor(coord, dtype=dtype, requires_grad=True)
        cell = torch.tensor(box, dtype=dtype).reshape(1, 3, 3)
        batch = torch.zeros(n, dtype=torch.int64)
        q_raw = charges(model, head, pos, cell, batch, numbers)
        q_neu = q_raw - q_raw.mean(0, keepdim=True)
        rows["A_raw"].append(summarise(born_charges(pos, cell, batch, q_raw, True), is_o, is_h))
        rows["A_neu"].append(summarise(born_charges(pos, cell, batch, q_neu, True), is_o, is_h))
        rows["A_frozen"].append(summarise(born_charges(pos, cell, batch, q_raw.detach(), True), is_o, is_h))
        rows["B_raw"].append(summarise(born_charges(pos, cell, batch, q_raw, False), is_o, is_h))
        rows["B_neu"].append(summarise(born_charges(pos, cell, batch, q_neu, False), is_o, is_h))
        if index == 0:
            print(
                "learned charges (e): q_O=%+.4f  q_H=%+.4f  net per frame=%+.3f"
                % (
                    float(q_raw[is_o].mean()) * CHARGE_UNIT,
                    float(q_raw[is_h].mean()) * CHARGE_UNIT,
                    float(q_raw.sum()) * CHARGE_UNIT,
                )
            )
            diag_o, diag_h, off = fixed_charge_sanity(
                model, head, pos, cell, batch, numbers, elem
            )
            print(
                "pipeline sanity (fixed SPC/E): Z*_O diag=%s  Z*_H diag=%s  offdiag max=%.2e"
                % (np.round(diag_o, 4), np.round(diag_h, 4), off)
            )
            if args.fd:
                bec_direct = born_charges(pos, cell, batch, q_raw, False)
                h = 1e-3

                def charge_at(r):
                    p = torch.tensor(r, dtype=dtype)
                    q = charges(model, head, p, cell, batch, numbers)
                    return q.detach().numpy().ravel()

                print(f"finite difference of the direct-sum BEC (atoms 0..{args.fd - 1}):")
                for i in range(args.fd):
                    fd = np.zeros((3, 3))
                    for beta in range(3):
                        rp = coord.copy()
                        rp[i, beta] += h
                        rm = coord.copy()
                        rm[i, beta] -= h
                        pp = (charge_at(rp)[:, None] * rp).sum(0) * CHARGE_UNIT
                        pm = (charge_at(rm)[:, None] * rm).sum(0) * CHARGE_UNIT
                        fd[:, beta] = (pp - pm) / (2 * h)
                    print(
                        "  atom %d (%s): analytic max %.3f  fd max %.3f  max|diff| %.2e"
                        % (
                            i,
                            elem[i],
                            np.abs(bec_direct[i]).max(),
                            np.abs(fd).max(),
                            np.abs(bec_direct[i] - fd).max(),
                        )
                    )
            spc_done = True

    print("\nBorn effective charges, Z*_iso in e  (mean over %d frames):" % args.frames)
    labels = {
        "A_raw": "PBC dephased, raw charges",
        "A_neu": "PBC dephased, neutral charges",
        "A_frozen": "PBC dephased, frozen charges",
        "B_raw": "direct sum, raw charges",
        "B_neu": "direct sum, neutral charges",
    }
    for key, label in labels.items():
        o = np.array([r["o"] for r in rows[key]])
        hh = np.array([r["h"] for r in rows[key]])
        sr = np.array([r["sum_rule"] for r in rows[key]])
        print(
            "  %-32s O = %+.4f +- %.4f   H = %+.4f +- %.4f   sum_i Z* max = %.2e"
            % (label, o.mean(), o.std(), hh.mean(), hh.std(), sr.max())
        )


if __name__ == "__main__":
    main()
