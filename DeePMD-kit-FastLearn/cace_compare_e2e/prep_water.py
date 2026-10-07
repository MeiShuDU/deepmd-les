#!/usr/bin/env python
# coding: utf-8
"""Convert the CACE 64-H2O water benchmark (water.xyz) into deepmd systems.

Aligns the deepmd hybrid_ener run with the author-published cace-LES benchmark:
  * same frames, same 90/10 train/valid split (valid_fraction=0.1, seed=1,
    np.random.default_rng(1).shuffle over frame indices, exactly as cace's
    random_train_valid_split in load_data.py);
  * same supervised residual target: E_target = E_frame - sum_Z ref[Z], with
    the benchmark's isolated-atom references {H: -187.6043857100553,
    O: -93.80219285502734} (cace subtracts these inside AtomicData.from_atoms;
    deepmd gets the residual directly in energy.npy);
  * type order / type_map the same as the deepmd input.json: ["O", "H"].

Also computes the se_a sel arrays (max neighbors per type within the chosen
rcut, min-image over PBC) so the descriptor does not silently truncate.

Output layout (under cace_compare_e2e/water/):
  train/set.000/{coord,box,force,energy}.npy  train/type.raw   (1434 frames)
  valid/set.000/{coord,box,force,energy}.npy  valid/type.raw   (159 frames)
  split_manifest.txt   (frame-index ledger: written positions -> water.xyz index)
"""
import os
import numpy as np
import ase.io
from ase.neighborlist import neighbor_list

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "cace", "water.xyz")
OUT = os.path.join(HERE, "water")

REF = {1: -187.6043857100553, 8: -93.80219285502734}  # H, O  (eV)
TYPE_MAP = ["O", "H"]  # matches deepmd input.json type_map
Z2TYPE = {8: 0, 1: 1}  # O -> 0, H -> 1
RCUT = 5.5             # same SR cutoff as the cace benchmark
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
        fz[r] = a.arrays["force"]
        cell = np.array(a.cell.array)
        box[r] = cell.reshape(9)
        energy[r] = frag_energy[r]
    # deepmd stores per-frame arrays flattened as (nframes, natoms * ndof).
    s = os.path.join(path, "set.000")
    np.save(os.path.join(s, "coord.npy"), coord.reshape(nf, nat * 3))
    np.save(os.path.join(s, "box.npy"), box)
    np.save(os.path.join(s, "force.npy"), fz.reshape(nf, nat * 3))
    np.save(os.path.join(s, "energy.npy"), energy)
    return nat, nf


def main():
    frames = ase.io.read(SRC, ":")
    n = len(frames)
    print(f"frames={n} natoms={len(frames[0])}")

    # --- reproduce the cace split exactly ---
    train_size = n - int(0.1 * n)
    rng = np.random.default_rng(1)
    idx = np.arange(n)
    rng.shuffle(idx)
    train_idx, valid_idx = idx[:train_size], idx[train_size:]
    print(f"train={len(train_idx)} valid={len(valid_idx)}")

    # --- residual target, same as cace AtomicData.from_atoms ---
    def resid(a):
        e = a.get_potential_energy()
        refs = sum(REF[z] for z in a.get_atomic_numbers())
        return e - refs

    e_res = np.array([resid(a) for a in frames])
    print(f"residual range: [{e_res.min():.4f}, {e_res.max():.4f}] eV/frame")

    # --- descriptor sel for the chosen rcut (max neighbors per type, PBC) ---
    # deepmd se_e2_a: sel[j] >= max over all centers of (# type-j neighbors
    # within rcut). Track per-center neighbor-counts per element.
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
        f.write(f"water.xyz total frames = {n}\n")
        f.write(f"train positions 0..{len(train_idx)-1} -> water.xyz frames {train_idx.tolist()}\n")
        f.write(f"valid positions 0..{len(valid_idx)-1} -> water.xyz frames {valid_idx.tolist()}\n")
    print(f"wrote systems under {OUT}")


if __name__ == "__main__":
    main()