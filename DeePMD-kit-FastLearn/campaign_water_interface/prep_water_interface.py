#!/usr/bin/env python
# coding: utf-8
"""Convert the CACE water-interface slab benchmark into deepmd systems.

Source: fit-water-interface/slab-fps-n-500.xyz
  500 frames, 1566 atoms each (522 H2O), fixed orthorhombic cell 25.6 x 25.6 x 65 A,
  species order O,H,H repeated identically in every frame.

Aligns the deepmd run with the cace arms in fit-water-interface/fit-interface-mp{0,1}[-sr]:
  * same frames, same 90/10 split (valid_fraction=0.1, seed=1) reproduced by calling
    cace's own random_train_valid_split, so the ledger is authoritative rather than
    a re-implementation;
  * same supervised residual target E_target = E_frame - sum_Z ref[Z], with the
    benchmark's references {H: -187.42397696905275, O: -93.71198848452647}. cace
    subtracts these inside AtomicData.from_atoms, so deepmd is handed the residual
    directly in energy.npy and no bias_atom_e is written;
  * type order ["O", "H"], matching the deepmd input type_map.

Output layout (under campaign_water_interface/data/water-interface/):
  train/set.000/{coord,box,force,energy}.npy   train/type.raw   (450 frames)
  valid/set.000/{coord,box,force,energy}.npy   valid/type.raw   (50 frames)
  sel.txt              se_a sel for rcut 6.0, measured with a 10% margin
  split_manifest.txt   frame-index ledger (written position -> slab xyz index)

Note the cutoff differs by design: cace trains at cutoff 5.5, the deepmd arms use
rcut 6.0 as specified, so sel is measured at 6.0.
"""
import os
import numpy as np
import ase.io
from ase.neighborlist import neighbor_list

HERE = os.path.dirname(os.path.abspath(__file__))
DATAREPO = os.path.join(
    HERE, "..", "cace-lr-fit-datarepo", "BingqingCheng-cace-lr-fit-0211150"
)
SRC = os.path.abspath(os.path.join(DATAREPO, "fit-water-interface", "slab-fps-n-500.xyz"))
OUT = os.path.join(HERE, "data", "water-interface")

REF = {1: -187.42397696905275, 8: -93.71198848452647}  # H, O  (eV)
TYPE_MAP = ["O", "H"]
Z2TYPE = {8: 0, 1: 1}
RCUT = 6.0             # deepmd descriptor rcut (cace, by contrast, uses 5.5)
SEL_MARGIN = 1.10
VALID_FRACTION = 0.1
SPLIT_SEED = 1


def write_system(path, frames, energy):
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
    for r, a in enumerate(frames):
        coord[r] = a.positions
        fz[r] = a.get_forces()
        box[r] = np.array(a.cell.array).reshape(9)
    s = os.path.join(path, "set.000")
    np.save(os.path.join(s, "coord.npy"), coord.reshape(nf, nat * 3))
    np.save(os.path.join(s, "box.npy"), box)
    np.save(os.path.join(s, "force.npy"), fz.reshape(nf, nat * 3))
    np.save(os.path.join(s, "energy.npy"), energy)
    return nat, nf


def main():
    frames = ase.io.read(SRC, ":")
    n = len(frames)
    nat = len(frames[0])
    print(f"frames={n} natoms={nat}")

    # --- the cace split, taken from cace itself so the ledger cannot drift ---
    from cace.tasks.load_data import random_train_valid_split

    train_idx, valid_idx = random_train_valid_split(
        list(range(n)), VALID_FRACTION, SPLIT_SEED
    )
    train_idx = np.array(train_idx)
    valid_idx = np.array(valid_idx)
    print(f"train={len(train_idx)} valid={len(valid_idx)}")

    def resid(a):
        return a.get_potential_energy() - sum(REF[z] for z in a.get_atomic_numbers())

    e_res = np.array([resid(a) for a in frames])
    e_tot = np.array([a.get_potential_energy() for a in frames])
    print(f"total    E range: [{e_tot.min():.4f}, {e_tot.max():.4f}] eV/frame")
    print(f"residual E range: [{e_res.min():.4f}, {e_res.max():.4f}] eV/frame")

    # --- se_a sel at the deepmd rcut: max over centres of per-type neighbour counts ---
    maxcnt = {1: 0, 8: 0}
    for a in frames:
        i, j, _ = neighbor_list("ijd", a, RCUT, self_interaction=False)
        z = a.get_atomic_numbers()
        per_center = np.zeros((len(a), 2), dtype=np.int64)  # [nH, nO]
        np.add.at(per_center, i, np.stack([z[j] == 1, z[j] == 8], 1))
        maxcnt[1] = max(maxcnt[1], int(per_center[:, 0].max()))
        maxcnt[8] = max(maxcnt[8], int(per_center[:, 1].max()))
    sel = [int(np.ceil(maxcnt[z] * SEL_MARGIN)) for z in (8, 1)]
    print(f"max neighbours within rcut={RCUT}: H={maxcnt[1]} O={maxcnt[8]}")
    print(f"sel (type order O,H) = {sel}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sel.txt"), "w") as f:
        f.write(" ".join(map(str, sel)) + "\n")
    nat_t, nf_t = write_system(
        os.path.join(OUT, "train"), [frames[i] for i in train_idx], e_res[train_idx]
    )
    nat_v, nf_v = write_system(
        os.path.join(OUT, "valid"), [frames[i] for i in valid_idx], e_res[valid_idx]
    )
    assert nat_t == nat_v == nat
    with open(os.path.join(OUT, "split_manifest.txt"), "w") as f:
        f.write(f"slab-fps-n-500.xyz total frames = {n}, natoms = {nat}\n")
        f.write(
            "split = cace.tasks.load_data.random_train_valid_split"
            f"(valid_fraction={VALID_FRACTION}, seed={SPLIT_SEED})\n"
        )
        f.write(f"train positions 0..{len(train_idx)-1} -> xyz frames {train_idx.tolist()}\n")
        f.write(f"valid positions 0..{len(valid_idx)-1} -> xyz frames {valid_idx.tolist()}\n")
        f.write(
            "references (eV): "
            + ", ".join(f"Z={z}: {e}" for z, e in sorted(REF.items()))
            + "\n"
        )
    print(f"wrote {nf_t} train + {nf_v} valid frames under {OUT}")


if __name__ == "__main__":
    main()