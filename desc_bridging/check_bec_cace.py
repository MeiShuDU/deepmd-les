"""Born effective charges of the native-cace campaign arms on the BEC/ VASP set.

``campaign_fastlearn/cace/runs/cace-lr`` and ``cace-sea-lr`` are NOT
``deepmd_cace`` models: they are plain cace ``NeuralNetworkPotential``s, so the
neighbour list comes from ``cace.data.AtomicData`` (the dataset transform the
training used) rather than from a deepmd input module, and the checkpoint is
either cace's own whole-module pickle (``cace-lr``) or the seam state dict
(``cace-sea-lr``, see ``sea_seam.load_sea_checkpoint``).

Both arms put cace's ``EwaldPotential`` (``norm_factor = 1.0``) in the energy at
weight 1 (``FeatureAdd``), so their learned charge converts to physical e with
``Polarization``'s default ``normalization_factor = 1/9.48933`` and no further
weight factor. ``PHYS_SCALE`` is the extra ``sqrt(eps_inf)`` the water-interface
work needed; it is reported as a separate variant, not folded in silently,
because whether it belongs here is exactly what the VASP comparison tests.

Usage:
    python check_bec_cace.py --arm cace-sea-lr --frames 20
    python check_bec_cace.py --dump cace_bec.npz --frames 100

``--dump`` exists because these checkpoints CANNOT be scored in the same
process as a ``deepmd_cace`` arm: ``deepmd_cace`` imports whatever ``cace`` the
editable install points at (``/root/app/cace``), and ``cace-lr`` is a
whole-module pickle written under ``/root/app/cace-ts``, whose internal
``lxlylz_dict`` uses string keys (``'1_0_0'``). Unpickled against the other
checkout the keys are strings but the forward does ``l - 1`` on them, so it
raises ``TypeError: unsupported operand type(s) for -: 'str' and 'int'``.
Python module identity is global, so the two arms need two processes; the
notebook shells out to this script and reads the ``.npz``.
"""
import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("DEEPMD_CACE_DEVICE", "cpu")

CACE_ROOT = "/root/app/cace-ts"
CAMPAIGN = (
    "/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_fastlearn"
)
SEAM_DIR = os.path.join(CAMPAIGN, "cace")
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (SEAM_DIR, HERE, CACE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cace  # noqa: E402  (the cace-ts checkout, per CACE_ROOT)
from ase import Atoms  # noqa: E402
from cace.data import AtomicData  # noqa: E402
from sea_seam import load_sea_checkpoint  # noqa: E402

RUNS = os.path.join(CAMPAIGN, "cace", "runs")
ARMS = ("cace-lr", "cace-sea-lr")
CUTOFF = 5.5
LR_WEIGHT = 1.0  # FeatureAdd, not CombinePotential: no weight on the Ewald term
EPS_INF = 1.78


def load_arm(arm, device="cpu", dtype=torch.float64):
    path = os.path.join(RUNS, arm, "best_model.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if arm.startswith("cace-sea"):
        model, _ = load_sea_checkpoint(cace, path, device)
    else:
        model = torch.load(path, map_location=device, weights_only=False)
    return model.to(device=device, dtype=dtype).eval()


def make_data(coord, cell, elem, cutoff=CUTOFF, dtype=torch.float32):
    """The graph the cace dataset would have built for this frame."""
    atoms = Atoms(symbols=list(elem), positions=coord, cell=cell, pbc=True)
    data = AtomicData.from_atoms(
        atoms, cutoff=cutoff,
        data_key={"energy": "E_total", "forces": "force"},
        atomic_energies=None,
    ).to_dict()
    n = len(elem)
    data["batch"] = torch.zeros(n, dtype=torch.int64)
    data["ptr"] = torch.tensor([0, n], dtype=torch.int64)
    for key in ("positions", "cell", "shifts"):
        if key in data and torch.is_floating_point(data[key]):
            data[key] = data[key].to(dtype)
    data["positions"] = data["positions"].clone().detach().requires_grad_(True)
    return data


def charges(model, data):
    """Per-atom charge [n,1], autograd-connected to ``data['positions']``.

    The forward is stopped the moment ``q`` exists: the trailing ``Forces``
    module differentiates the total energy again and is not needed here.
    """
    for module in model.input_modules:
        data = module(data, compute_stress=False, compute_virials=False)
    if model.representation is not None:
        data = model.representation(data)
    for module in model.output_modules:
        data = module(data, training=True)
        if "q" in data:
            break
    return data["q"]


def born_charges(data, q, pbc=True):
    """Z*[i, alpha, beta] = d P_alpha / d r_{i,beta} in physical e.

    ``Polarization``'s default normalisation is ``1/9.48933``, the cace-kernel
    charge-unit conversion, so the result is already in physical e. ``q`` is
    handed over explicitly: some cace modules return a fresh dict rather than
    mutating the one they are given, so ``data['q']`` cannot be relied on.
    """
    from cace.modules.grad import Grad
    from cace.modules.polarization import Dephase, Polarization

    data["q"] = q

    Polarization(
        charge_key="q", output_key="polarization", phase_key="phase",
        remove_mean=False, pbc=pbc,
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


def load_arm_auto(arm, coord, cell, elem, device="cpu"):
    """The arm in the widest dtype its own modules accept.

    Returns ``(model, dtype)``. cace's node embedding is hardcoded float32, so
    ``cace-lr`` cannot be widened; the seam arm's descriptor is deepmd and runs
    in double, which the BEC's response term wants.
    """
    last = None
    for candidate in (torch.float64, torch.float32):
        model = load_arm(arm, device, candidate)
        try:
            charges(model, make_data(coord, cell, elem, dtype=candidate))
        except RuntimeError as exc:
            if "dtype" not in str(exc):
                raise
            last = exc
            continue
        return model, candidate
    raise RuntimeError(f"{arm}: no working dtype") from last


def arm_diag(model, dtype, coord, cell, elem):
    """Diagonal Z*_aa [n,3] in physical e, plus the per-atom charge."""
    data = make_data(coord, cell, elem, dtype=dtype)
    q = charges(model, data)
    bec = born_charges(data, q, pbc=True)
    return np.diagonal(bec, axis1=1, axis2=2), q.detach().numpy().reshape(-1)


def rmse_r2(pred, dft):
    rmse = float(np.sqrt(np.mean((pred - dft) ** 2)))
    r2 = float(1.0 - np.sum((pred - dft) ** 2) / np.sum((dft - dft.mean()) ** 2))
    return rmse, r2


def collect(arm, vasp, frames, device="cpu"):
    """Per-atom diagonal Z* (sign-fixed, unscaled) and the frame-0 charge."""
    from eval_vasp_bec import read_poscar, read_outcar_bec

    indices = sorted(
        int(f.split(".")[1]) for f in os.listdir(os.path.join(vasp, "poscars"))
    )[:frames]
    coord0, cell0, elem0 = read_poscar(
        os.path.join(vasp, "poscars", f"POSCAR.{indices[0]}")
    )
    model, dtype = load_arm_auto(arm, coord0, cell0, elem0, device)

    pred, dft, elem_all, q_all = [], [], [], []
    for idx in indices:
        coord, cell, elem = read_poscar(
            os.path.join(vasp, "poscars", f"POSCAR.{idx}")
        )
        ref = read_outcar_bec(os.path.join(vasp, "outcars", f"OUTCAR.{idx}"))
        diag, q = arm_diag(model, dtype, coord, cell, elem)
        pred.append(diag)
        dft.append(np.diagonal(ref, axis1=1, axis2=2))
        elem_all.append(elem)
        q_all.append(q)

    pred = np.concatenate([d.ravel() for d in pred])
    dft = np.concatenate([d.ravel() for d in dft])
    elem_flat = np.concatenate([np.repeat(e, 3) for e in elem_all])
    q0 = q_all[0]
    q_mean = np.mean(
        [q[[e == "O" for e in el]].mean() - q[[e == "H" for e in el]].mean()
         for q, el in zip(q_all, elem_all)]
    )

    sign = -1.0 if pred[elem_flat == "O"].mean() > 0 else 1.0
    return dict(
        arm=arm, dtype=str(dtype), sign=sign, signed=pred * sign, pred=pred,
        dft=dft, elem=elem_flat, q0=q0, q0_elem=elem_all[0], q_mean=q_mean,
    )


def report(r):
    """The console table for one collected arm."""
    signed, dft, elem_flat = r["signed"], r["dft"], r["elem"]
    print(f"arm {r['arm']}   dtype {r['dtype']}   "
          f"q_O-q_H (cace units) = {r['q_mean']:+.3f}   "
          f"sign gauge x{r['sign']:+.0f}")
    print(f"  {'scale':>26s}{'Z*_O':>9s}{'Z*_H':>9s}{'RMSE':>9s}{'R^2':>9s}"
          f"{'O: RMSE/R^2':>18s}{'H: RMSE/R^2':>18s}")
    for label, scale in (
        ("x1 (plain conversion)", 1.0),
        ("x sqrt(w*eps_inf)", np.sqrt(LR_WEIGHT * EPS_INF)),
    ):
        p = signed * scale
        o, h = elem_flat == "O", elem_flat == "H"
        rm, r2 = rmse_r2(p, dft)
        rmo, r2o = rmse_r2(p[o], dft[o])
        rmh, r2h = rmse_r2(p[h], dft[h])
        print(f"  {label:>26s}{p[o].mean():+9.3f}{p[h].mean():+9.3f}"
              f"{rm:9.4f}{r2:9.4f}{rmo:8.3f}/{r2o:<9.3f}{rmh:8.3f}/{r2h:<9.3f}")
    vo, vh = dft[elem_flat == "O"].mean(), dft[elem_flat == "H"].mean()
    print(f"  {'VASP reference':>26s}{vo:+9.3f}{vh:+9.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="cace-sea-lr", choices=ARMS)
    parser.add_argument("--vasp", default="/root/app/deepmd-les/BEC")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dump", default=None,
                        help="write every arm's arrays to this .npz and exit")
    args = parser.parse_args()

    if args.dump:
        # the shared DFT reference is arm-independent, so dump it once
        out = {"lr_weight": LR_WEIGHT, "eps_inf": EPS_INF, "cutoff": CUTOFF}
        for arm in ARMS:
            r = collect(arm, args.vasp, args.frames, args.device)
            report(r)
            for key in ("signed", "dft", "elem", "q0", "q0_elem"):
                out[f"{arm}_{key}"] = r[key]
            out[f"{arm}_sign"] = r["sign"]
            out[f"{arm}_dtype"] = r["dtype"]
        np.savez(args.dump, **out)
        print(f"\nwrote {args.dump}")
        return

    report(collect(args.arm, args.vasp, args.frames, args.device))


if __name__ == "__main__":
    main()