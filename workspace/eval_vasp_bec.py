"""Born effective charges of a ``deepmd_cace`` checkpoint on the BEC/ VASP set.

``BEC/`` holds 100 revPBE snapshots of bulk liquid water (64 H2O, 192 atoms,
cubic 12.429 A) with ``LEPSILON=.TRUE.``: ``outcars/OUTCAR.i`` (the reference
Born effective charges) and ``poscars/POSCAR.i`` (the geometry), element order
H(128) then O(64).

For every snapshot the checkpoint's latent charge is fed through CACE's
``Polarization -> Grad -> Dephase`` chain in two conventions - the dephased
periodic Z* and the direct sum - and compared against the VASP tensor. The
model's overall charge sign is a gauge, so it is flipped to the physical
O-negative one before reporting.

The reported Z* is scaled by ``sqrt(w * eps_inf)``, where ``w`` is the weight the
long-range term carries in the energy. That weight is read off the checkpoint's
own config, so an arm that adds ``E_LR`` through ``FeatureAdd`` (unit weight,
``combine_potentials: false``) is scaled by ``sqrt(1.78) = 1.3342`` and one that
puts a weight on a ``CombinePotential`` branch by ``sqrt(w * 1.78)``; passing
``--lr-weight`` overrides the lookup for checkpoints that carry no config.

``--dump`` writes the dephased per-atom diagonal Z* and the DFT reference to an
``.npz`` in the same schema ``check_bec_cace.py --dump`` uses, for
``bec_scatter_pred_dft.ipynb``.

Usage:
    python eval_vasp_bec.py --frames 100
    python eval_vasp_bec.py --ckpt <arm>/best_model.pth --frames 100 \
        --dump vasp_bec_<arm>.npz
"""
import argparse
import os
import re
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("DEEPMD_CACE_DEVICE", "cpu")

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check_bec as CB  # noqa: E402
import check_bec_arm as CBA  # noqa: E402

VASP_DIR = "/root/app/deepmd-les/BEC"
ATOMIC_NUMBER = {"H": 1, "O": 8}
W_LR = 0.02
EPS_INF = 1.78
PHYS_SCALE = np.sqrt(W_LR * EPS_INF)


def lr_weight_of(path):
    """The weight the checkpoint's own config puts on the long-range energy.

    ``combine_potentials: true`` scales the Ewald branch by ``long_range.weight``
    in ``CombinePotential``; the default path instead sums ``SR_energy`` and
    ``ewald_potential`` with ``FeatureAdd``, which carries no weight at all.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    long_range = checkpoint["config"]["model"].get("long_range", {})
    if long_range.get("combine_potentials", False):
        return float(long_range.get("weight", 1.0)), "long_range.weight"
    return 1.0, "FeatureAdd (unit weight)"


def cutoff_of(model):
    """The descriptor's rcut, for the ``cutoff`` key ``check_bec_cace.py --dump`` writes.

    Read off the model rather than imported from ``check_bec_cace``: that module
    lives in the cace-ts checkout, and importing it here would pull a second
    ``cace`` package into the process alongside the one ``deepmd_cace`` uses.
    """
    for module in model.input_modules:
        descriptor = getattr(module, "descriptor", None)
        if descriptor is not None and hasattr(descriptor, "get_rcut"):
            return float(descriptor.get_rcut())
    return float("nan")


def read_poscar(path):
    lines = open(path).read().splitlines()
    scale = float(lines[1].split()[0])
    cell = np.array(
        [[float(x) for x in lines[2 + i].split()[:3]] for i in range(3)]
    ) * scale
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    n = sum(counts)
    frac = np.array([[float(x) for x in lines[8 + i].split()[:3]] for i in range(n)])
    elem = np.array([s for s, c in zip(symbols, counts) for _ in range(c)])
    return frac @ cell, cell, elem


def read_outcar_bec(path):
    txt = open(path).read()
    at = txt.rindex("BORN EFFECTIVE CHARGES (including local field effects)")
    ions = re.findall(
        r"^ ion\s+\d+\s*$\n"
        r"((?:\s+\d+\s+[-0-9.]+\s+[-0-9.]+\s+[-0-9.]+\s*\n){3})",
        txt[at:],
        flags=re.M,
    )
    return np.array(
        [
            [[float(x) for x in ln.split()[1:]] for ln in m.strip("\n").splitlines()]
            for m in ions
        ]
    )


def isotropic(bec):
    return np.trace(bec, axis1=1, axis2=2) / 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=os.path.join(
            HERE, "..", "DeePMD-kit-FastLearn", "campaign_water_interface",
            "deepmd", "runs", "sea-lr-eq_sA", "best_model.pth",
        ),
    )
    parser.add_argument("--vasp", default=VASP_DIR)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--lr-weight", type=float, default=None,
                        help="weight on the long-range energy; default: read "
                             "from the checkpoint's own config")
    parser.add_argument("--dump", default=None,
                        help="write the dephased diagonal Z* and the DFT "
                             "reference to this .npz and exit")
    args = parser.parse_args()

    if args.lr_weight is None:
        lr_weight, source = lr_weight_of(args.ckpt)
    else:
        lr_weight, source = args.lr_weight, "--lr-weight"
    scale = np.sqrt(lr_weight * EPS_INF)

    model = CB.load_model(args.ckpt)
    head = CB.charge_head_of(model)
    dtype = CB.model_dtype(model)

    keys = ("pbc", "dir", "frz", "vasp")
    mean = {k: dict(o=[], h=[]) for k in keys}
    atoms = {k: dict(o=[], h=[]) for k in keys}
    sumrule = {k: [] for k in keys}
    q_iso = []
    diag_pbc, diag_dft, elem_all, q_all = [], [], [], []

    indices = sorted(
        int(f.split(".")[1]) for f in os.listdir(os.path.join(args.vasp, "poscars"))
    )[: args.frames]

    for idx in indices:
        coord, cell, elem = read_poscar(
            os.path.join(args.vasp, "poscars", f"POSCAR.{idx}")
        )
        ref = read_outcar_bec(os.path.join(args.vasp, "outcars", f"OUTCAR.{idx}"))
        numbers = torch.tensor([ATOMIC_NUMBER[e] for e in elem])
        pos = torch.tensor(coord, dtype=dtype, requires_grad=True)
        cellt = torch.tensor(cell, dtype=dtype).reshape(1, 3, 3)
        batch = torch.zeros(len(elem), dtype=torch.int64)

        _, q_ker = CBA.kernel_charges(model, head, pos, cellt, batch, numbers)
        diag_pbc.append(np.diagonal(CB.born_charges(pos, cellt, batch, q_ker, True),
                                   axis1=1, axis2=2))
        diag_dft.append(np.diagonal(ref, axis1=1, axis2=2))
        elem_all.append(elem)
        q_all.append(q_ker.detach().numpy().reshape(-1))
        for key, q, pbc in (
            ("pbc", q_ker, True),
            ("dir", q_ker, False),
            ("frz", q_ker.detach(), False),
        ):
            bec = CB.born_charges(pos, cellt, batch, q, pbc)
            iso = isotropic(bec)
            sumrule[key].append(float(np.abs(bec.sum(0)).max()))
            for name, mask in (("o", elem == "O"), ("h", elem == "H")):
                mean[key][name].append(iso[mask].mean())
                atoms[key][name].append(iso[mask])

        sumrule["vasp"].append(float(np.abs(ref.sum(0)).max()))
        for name, mask in (("o", elem == "O"), ("h", elem == "H")):
            mean["vasp"][name].append(isotropic(ref)[mask].mean())
            atoms["vasp"][name].append(isotropic(ref)[mask])

        q_iso.append(
            (
                float(q_ker[elem == "O"].mean()) * CB.CHARGE_UNIT,
                float(q_ker[elem == "H"].mean()) * CB.CHARGE_UNIT,
            )
        )

    for k in keys:
        for name in ("o", "h"):
            mean[k][name] = np.array(mean[k][name])
            atoms[k][name] = np.concatenate(atoms[k][name])
    q = np.array(q_iso)
    sign = -1.0 if mean["pbc"]["o"].mean() > 0 else 1.0

    print(f"checkpoint : {os.path.relpath(args.ckpt)}")
    print(f"dataset    : {args.vasp}  ({len(indices)} snapshots, 192 atoms, cubic 12.429 A)")
    print(f"charges (e): q_O = {sign * q[:, 0].mean():+.3f}  "
          f"q_H = {sign * q[:, 1].mean():+.3f}  "
          f"sign gauge flipped x{sign:+.0f}")

    labels = {
        "pbc": "model  PBC dephased",
        "dir": "model  direct sum",
        "frz": "model  direct, frozen",
        "vasp": "VASP   (revPBE)",
    }
    vo, vh = mean["vasp"]["o"].mean(), mean["vasp"]["h"].mean()

    print(f"\nZ*_iso (e), mean over frames")
    print(f"  {'':22s}{'O':>9s}{'H':>9s}   {'per-atom std O/H':>16s}   "
          f"{'RMSE O/H':>10s}   {'sum_rule':>9s}")
    for k in keys:
        o = sign * mean[k]["o"] if k != "vasp" else mean[k]["o"]
        h = sign * mean[k]["h"] if k != "vasp" else mean[k]["h"]
        if k == "vasp":
            so, sh = atoms["vasp"]["o"].std(), atoms["vasp"]["h"].std()
            ro = rh = 0.0
        else:
            so, sh = (sign * atoms[k]["o"]).std(), (sign * atoms[k]["h"]).std()
            ro = np.sqrt(((sign * atoms[k]["o"] - atoms["vasp"]["o"]) ** 2).mean())
            rh = np.sqrt(((sign * atoms[k]["h"] - atoms["vasp"]["h"]) ** 2).mean())
        print(f"  {labels[k]:22s}{o.mean():+9.3f}{h.mean():+9.3f}   "
              f"{so:7.3f} /{sh:7.3f}   {ro:4.2f} /{rh:4.2f}   "
              f"{np.mean(sumrule[k]):9.1e}")

    print(f"\n  frame-to-frame correlation with VASP (Pearson r)")
    for k in ("pbc", "dir", "frz"):
        ro = np.corrcoef(sign * mean[k]["o"], mean["vasp"]["o"])[0, 1]
        rh = np.corrcoef(sign * mean[k]["h"], mean["vasp"]["h"])[0, 1]
        print(f"    {labels[k]:22s} O r={ro:+.2f}   H r={rh:+.2f}")

    print(f"\n  x sqrt(w*eps_inf) = {scale:.4f}  (w={lr_weight:g} from {source}, "
          f"eps_inf={EPS_INF})")
    print(f"  {'':22s}{'Z*_O':>9s}{'Z*_H':>9s}   {'O/VASP':>7s}")
    for k in keys:
        o = (sign * mean[k]["o"].mean()) if k != "vasp" else mean[k]["o"].mean()
        h = (sign * mean[k]["h"].mean()) if k != "vasp" else mean[k]["h"].mean()
        s = scale if k != "vasp" else 1.0
        print(f"  {labels[k]:22s}{o * s:+9.3f}{h * s:+9.3f}   {o * s / vo:7.2f}")

    print(f"\n  per-atom agreement at the x{scale:.4f} scale (O / H)")
    for k in ("pbc", "dir", "frz"):
        line = []
        for name in ("o", "h"):
            m = sign * scale * atoms[k][name]
            v = atoms["vasp"][name]
            r = np.corrcoef(m, v)[0, 1]
            line.append(f"r={r:+.2f} RMSE={np.sqrt(((m - v) ** 2).mean()):.3f} "
                        f"std={m.std():.3f}")
        print(f"    {labels[k]:22s} O: {line[0]}   H: {line[1]}")
    vv = atoms["vasp"]["o"]
    print(f"    {'VASP (for reference)':22s} O: std={vv.std():.3f}   "
          f"H: std={atoms['vasp']['h'].std():.3f}")

    if args.dump:
        pred = np.concatenate([d.ravel() for d in diag_pbc])
        dft = np.concatenate([d.ravel() for d in diag_dft])
        elem_flat = np.concatenate([np.repeat(e, 3) for e in elem_all])
        name = os.path.basename(os.path.dirname(args.ckpt)) or "ckpt"
        np.savez(
            args.dump,
            lr_weight=lr_weight, eps_inf=EPS_INF, cutoff=cutoff_of(model),
            **{
                f"{name}_signed": pred * sign,
                f"{name}_dft": dft,
                f"{name}_elem": elem_flat,
                f"{name}_q0": q_all[0],
                f"{name}_q0_elem": elem_all[0],
                f"{name}_sign": sign,
                f"{name}_dtype": str(dtype),
            },
        )
        print(f"\nwrote {args.dump} ({name}: {pred.size:,} diagonal elements, "
              f"sign gauge x{sign:+.0f})")


if __name__ == "__main__":
    main()
