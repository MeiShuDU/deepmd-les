#!/usr/bin/env python
# coding: utf-8
"""Convert the CACE water/slab-interface benchmark (slab-fps-n-500.xyz) into deepmd systems.

Aligns the deepmd hybrid_ener run with the author's cace-LES benchmark for
fit-water-interface:
  * same frames, same 90/10 train/valid split (valid_fraction=0.1, seed=1,
    np.random.default_rng(1).shuffle over frame indices, exactly as cace's
    random_train_valid_split in load_data.py);
  * same supervised residual target: E_target = E_frame - sum_Z ref[Z], with
    the benchmark's references {H: -187.42397696905275, O: -93.71198848452647}
    (cace subtracts these inside AtomicData.from_atoms; deepmd gets the
    residual directly in energy.npy);
  * type order / type_map the same as the deepmd input: ["O", "H"].

The source file is 500 frames of 1566 atoms (522 H2O) in a fixed orthorhombic
25.6 x 25.6 x 65.0 slab cell, and every frame carries the identical species
sequence (O,H,H repeated), so a single type.raw is valid for all frames.

Also computes the se_a sel arrays (max neighbors per type within the chosen
rcut, min-image over PBC) so the descriptor does not silently truncate.

Output layout (under campaign_water_iface/data/):
  train/set.000/{coord,box,force,energy}.npy  train/type.raw   (450 frames)
  valid/set.000/{coord,box,force,energy}.npy  valid/type.raw   (50 frames)
  sel.txt              (se_a sel, ordered as type_map)
  split_manifest.txt   (frame-index ledger: written positions -> xyz index)
  source.json          (provenance: path, sha256, frame count, reference energies)
"""
import hashlib
import json
import os

import ase.io
import numpy as np
from ase.neighborlist import neighbor_list

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(
    HERE,
    "..",
    "cace-lr-fit-datarepo",
    "BingqingCheng-cace-lr-fit-0211150",
    "fit-water-interface",
    "slab-fps-n-500.xyz",
)
OUT = os.path.join(HERE, "data")

REF = {1: -187.42397696905275, 8: -93.71198848452647}  # H, O  (eV), the author's table
TYPE_MAP = ["O", "H"]
Z2TYPE = {8: 0, 1: 1}  # O -> 0, H -> 1
RCUT = 6.0             # deepmd se_a cutoff (the cace arms train at cutoff 5.5)
SEL_MARGIN = 1.10      # se_a pads to sel; keep a safety margin over the max


def write_system(path, frames, frag_energy):
    os.makedirs(os.path.join(path, "set.000"), exist_ok=True)
    nf = len(frames)
    nat = len(frames[0])
    type_mp = np.array([Z2TYPE[z] for z in frames[0].get_atomic_numbers()]).astype(int)
    np.savetxt(os.path.join(path, "type.raw"), type_mp, fmt="%d")
    with open(os.path.join(path, "type_map.raw"), "w") as f:
        f.write("\n".join(TYPE_MAP) + "\n")
    coord = np.zeros((nf, nat, 3))
    fz = np.zeros((nf, nat, 3))
    box = np.zeros((nf, 9))
    energy = np.zeros(nf)
    for r, a in enumerate(frames):
        coord[r] = a.positions
        fz[r] = a.get_forces()
        box[r] = np.array(a.cell.array).reshape(9)
        energy[r] = frag_energy[r]
    s = os.path.join(path, "set.000")
    np.save(os.path.join(s, "coord.npy"), coord.reshape(nf, nat * 3))
    np.save(os.path.join(s, "box.npy"), box)
    np.save(os.path.join(s, "force.npy"), fz.reshape(nf, nat * 3))
    np.save(os.path.join(s, "energy.npy"), energy)
    return nat, nf


def main():
    src = os.path.abspath(SRC)
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
    frames = ase.io.read(src, ":")
    n = len(frames)
    nat = len(frames[0])
    print(f"frames={n} natoms={nat}")
    print(f"sha256={sha}")

    # every frame must share one species sequence, else a single type.raw is wrong
    z0 = frames[0].get_atomic_numbers()
    for r, a in enumerate(frames):
        if len(a) != nat or not np.array_equal(a.get_atomic_numbers(), z0):
            raise SystemExit(f"frame {r} differs in natoms or species order; needs a per-frame type array")
    print("species sequence identical in every frame")

    # --- reproduce the cace split exactly ---
    train_size = n - int(0.1 * n)
    rng = np.random.default_rng(1)
    idx = np.arange(n)
    rng.shuffle(idx)
    train_idx, valid_idx = idx[:train_size], idx[train_size:]
    print(f"train={len(train_idx)} valid={len(valid_idx)}")

    # --- residual target, same as cace AtomicData.from_atoms ---
    e_res = np.array(
        [a.get_potential_energy() - sum(REF[z] for z in a.get_atomic_numbers()) for a in frames]
    )
    print(f"residual E-ref range: [{e_res.min():.4f}, {e_res.max():.4f}] eV/frame")
    print(f"residual  mean: {e_res.mean():.6f} eV/frame -> {e_res.mean()/nat:.6f} eV/atom")

    # --- descriptor sel for the chosen rcut (max neighbors per type, PBC) ---
    maxcnt = {1: 0, 8: 0}
    for a in frames:
        i, j, _ = neighbor_list("ijd", a, RCUT, self_interaction=False)
        z = a.get_atomic_numbers()
        per_center = np.zeros((len(a), 2), dtype=np.int64)  # [nH, nO]
        np.add.at(per_center, i, np.stack([z[j] == 1, z[j] == 8], 1))
        maxcnt[1] = max(maxcnt[1], int(per_center[:, 0].max()))
        maxcnt[8] = max(maxcnt[8], int(per_center[:, 1].max()))
    sel = [int(np.ceil(maxcnt[z] * SEL_MARGIN)) for z in (8, 1)]
    logme = {("H" if z == 1 else "O"): v for z, v in maxcnt.items()}
    print(f"max neighbors within rcut={RCUT}: {logme}")
    print(f"sel (type order O,H) = {sel}")

    # --- write deepmd systems ---
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sel.txt"), "w") as f:
        f.write(" ".join(map(str, sel)) + "\n")
    write_system(os.path.join(OUT, "train"), [frames[i] for i in train_idx], e_res[train_idx])
    write_system(os.path.join(OUT, "valid"), [frames[i] for i in valid_idx], e_res[valid_idx])
    with open(os.path.join(OUT, "split_manifest.txt"), "w") as f:
        f.write(f"slab-fps-n-500.xyz total frames = {n}\n")
        f.write(f"train positions 0..{len(train_idx)-1} -> xyz frames {train_idx.tolist()}\n")
        f.write(f"valid positions 0..{len(valid_idx)-1} -> xyz frames {valid_idx.tolist()}\n")
    with open(os.path.join(OUT, "source.json"), "w") as f:
        json.dump(
            {
                "source": src,
                "sha256": sha,
                "n_frames": n,
                "natoms": nat,
                "cell": [25.6, 25.6, 65.0],
                "atomic_energies": REF,
                "valid_fraction": 0.1,
                "split_seed": 1,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "deepmd_rcut": RCUT,
                "sel": sel,
                "energy_target": "residual (E_frame - sum_Z ref[Z])",
            },
            f,
            indent=2,
        )
    print(f"wrote systems under {OUT}")


if __name__ == "__main__":
    main()