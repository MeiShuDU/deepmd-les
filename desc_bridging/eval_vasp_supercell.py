"""Finite-size check of the sea-lr-eq dephased BEC on the BEC/ VASP set.

The dephased periodic Z* carries a correction that scales with the cell, so a
match to the VASP reference on the native 12.429 A cell could be cell-specific.
Rebuild each geometry as a 2x2x2 supercell (same local environments, doubled
cell), rerun the same pipeline, and see whether the scaled dephased value stays
at the VASP reference.

Usage:
    python eval_vasp_supercell.py --frames 5
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
import check_bec_arm as CBA  # noqa: E402
from eval_vasp_bec import ATOMIC_NUMBER, PHYS_SCALE, isotropic, read_poscar  # noqa: E402


def supercell(coord, cell, numbers, n):
    shifts = np.array(
        [[i, j, k] for i in range(n) for j in range(n) for k in range(n)]
    ) @ cell
    coord = (coord[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    numbers = np.tile(numbers, n**3)
    src = np.tile(np.arange(len(numbers) // n**3), n**3)
    return coord, cell * n, numbers, src


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=os.path.join(
            HERE, "..", "DeePMD-kit-FastLearn", "campaign_water_interface",
            "deepmd", "runs", "sea-lr-eq_sA", "best_model.pth",
        ),
    )
    parser.add_argument("--vasp", default="/root/app/deepmd-les/BEC")
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    model = CB.load_model(args.ckpt)
    head = CB.charge_head_of(model)
    dtype = CB.model_dtype(model)

    indices = sorted(
        int(f.split(".")[1]) for f in os.listdir(os.path.join(args.vasp, "poscars"))
    )[: args.frames]

    print(f"{'frame':>5s}  {'1x1x1 dephased':>16s}  {'2x2x2 dephased':>16s}  "
          f"{'1x scaled':>10s}  {'2x scaled':>10s}")
    acc = {1: [], 2: []}
    for idx in indices:
        coord, cell, elem = read_poscar(
            os.path.join(args.vasp, "poscars", f"POSCAR.{idx}")
        )
        numbers = np.array([ATOMIC_NUMBER[e] for e in elem])
        out = {}
        sr = {}
        for n in (1, 2):
            c, cl, nm, src = supercell(coord, cell, numbers, n)
            pos = torch.tensor(c, dtype=dtype, requires_grad=True)
            cellt = torch.tensor(cl, dtype=dtype).reshape(1, 3, 3)
            batch = torch.zeros(len(nm), dtype=torch.int64)
            _, q_ker = CBA.kernel_charges(
                model, head, pos, cellt, batch, torch.tensor(nm)
            )
            bec = CB.born_charges(pos, cellt, batch, q_ker, True)
            iso = isotropic(bec)
            is_o = elem[src] == "O"
            out[n] = iso[is_o].mean()
            sr[n] = float(np.abs(bec.sum(0)).max())
            acc[n].append(out[n])

        print(f"  {idx:3d}  {out[1]:+16.4f}  {out[2]:+16.4f}  "
              f"{out[1] * PHYS_SCALE:+10.3f}  {out[2] * PHYS_SCALE:+10.3f}"
              f"   sum_rule {sr[1]:.2e} -> {sr[2]:.2e}")

    a1, a2 = np.array(acc[1]), np.array(acc[2])
    sign = -1.0 if a1.mean() > 0 else 1.0
    print(f"\n  Z*_O (dephased), mean over {args.frames} frames, x{PHYS_SCALE:.4f}, "
          f"sign gauge x{sign:+.0f}")
    print(f"    1x1x1 (12.43 A): {sign * a1.mean() * PHYS_SCALE:+.3f}")
    print(f"    2x2x2 (24.86 A): {sign * a2.mean() * PHYS_SCALE:+.3f}")
    print(f"    VASP reference : -0.903")


if __name__ == "__main__":
    main()