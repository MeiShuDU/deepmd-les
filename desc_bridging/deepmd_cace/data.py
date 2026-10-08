"""Load DeepMD-kit raw systems into CACE's native graph batches."""

import glob
import os
import shutil

import numpy as np
import torch
from ase import Atoms
from cace.data import AtomicData
from cace.tools import torch_geometric


def _read_type_map(system, configured_type_map):
    path = os.path.join(system, "type_map.raw")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing DeepMD type map: {path}")
    with open(path, encoding="utf-8") as stream:
        found = [line.strip() for line in stream if line.strip()]
    if found != list(configured_type_map):
        raise ValueError(
            f"{path} contains {found}, but model.type_map is {configured_type_map}"
        )


def _load_system(system, type_map, cutoff, with_data, collect_stats=True,
                 atomic_energies=None):
    system = os.path.abspath(system)
    type_path = os.path.join(system, "type.raw")
    if not os.path.isfile(type_path):
        raise FileNotFoundError(f"missing DeepMD atom types: {type_path}")
    _read_type_map(system, type_map)
    atom_types = np.loadtxt(type_path, dtype=np.int64, ndmin=1).reshape(-1)
    if atom_types.size == 0 or atom_types.min() < 0 or atom_types.max() >= len(type_map):
        raise ValueError(f"invalid type ids in {type_path}")

    set_dirs = sorted(glob.glob(os.path.join(system, "set.*")))
    if not set_dirs:
        raise FileNotFoundError(f"no set.* directories under {system}")

    samples = []
    stat_samples = []
    for set_dir in set_dirs:
        coord_path = os.path.join(set_dir, "coord.npy")
        box_path = os.path.join(set_dir, "box.npy")
        energy_path = os.path.join(set_dir, "energy.npy")
        force_path = os.path.join(set_dir, "force.npy")
        arrays = {"coord": coord_path, "box": box_path, "energy": energy_path,
                  "force": force_path}
        missing = [name for name, path in arrays.items() if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"{set_dir} is missing {', '.join(missing)}")

        coord = np.load(coord_path, mmap_mode="r")
        nframes = coord.shape[0]
        natoms = atom_types.size
        coord = np.asarray(coord).reshape(nframes, natoms, 3)
        box = np.asarray(np.load(box_path, mmap_mode="r")).reshape(nframes, 3, 3)
        energy = np.asarray(np.load(energy_path, mmap_mode="r")).reshape(nframes)
        force = np.asarray(np.load(force_path, mmap_mode="r")).reshape(nframes, natoms, 3)
        if not (len(box) == len(energy) == len(force) == nframes):
            raise ValueError(f"frame count mismatch in {set_dir}")
        if not all(np.isfinite(value).all() for value in (coord, box, energy, force)):
            raise ValueError(f"non-finite values in {set_dir}")

        if with_data:
            symbols = [type_map[index] for index in atom_types]
            for frame in range(nframes):
                cell = np.asarray(box[frame])
                if np.any(np.abs(cell) > 1e-12) and abs(np.linalg.det(cell)) < 1e-10:
                    raise ValueError(f"singular periodic cell in {set_dir}, frame {frame}")
                atoms = Atoms(
                    symbols=symbols,
                    positions=np.asarray(coord[frame]),
                    cell=cell,
                    pbc=bool(np.any(np.abs(cell) > 1e-12)),
                )
                atoms.info["ref_energy"] = float(energy[frame])
                atoms.arrays["ref_forces"] = np.asarray(force[frame]).copy()
                samples.append(AtomicData.from_atoms(
                    atoms, cutoff=cutoff, atomic_energies=atomic_energies
                ))

        if collect_stats:
            stat_samples.append({
                "coord": torch.as_tensor(np.asarray(coord).copy(), dtype=torch.float64),
                "atype": torch.as_tensor(
                    np.broadcast_to(atom_types, (nframes, natoms)).copy(), dtype=torch.long
                ),
                "box": torch.as_tensor(np.asarray(box).reshape(nframes, 9).copy(),
                                        dtype=torch.float64),
                "natoms": natoms,
            })
    return samples, stat_samples


def load_split(systems, type_map, cutoff, batch_size, shuffle, collect_stats=True,
               atomic_energies=None):
    data = []
    stats = []
    for system in systems:
        system_data, system_stats = _load_system(
            system, type_map, cutoff, with_data=True, collect_stats=collect_stats,
            atomic_energies=atomic_energies,
        )
        data.extend(system_data)
        stats.extend(system_stats)
    if not data:
        raise ValueError("the configured split contains no frames")
    loader = torch_geometric.DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )
    return loader, stats


def split_system(source, train_dir, valid_dir, valid_fraction=0.1, seed=1):
    """Create DeepMD-style train/valid systems from one labeled system."""
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    source = os.path.abspath(source)
    type_path = os.path.join(source, "type.raw")
    type_map_path = os.path.join(source, "type_map.raw")
    if not os.path.isfile(type_path) or not os.path.isfile(type_map_path):
        raise FileNotFoundError("source must contain type.raw and type_map.raw")

    frames = {key: [] for key in ("coord", "box", "energy", "force")}
    for set_dir in sorted(glob.glob(os.path.join(source, "set.*"))):
        for key in frames:
            frames[key].append(np.load(os.path.join(set_dir, f"{key}.npy")))
    if not frames["coord"]:
        raise FileNotFoundError(f"no set.* directories under {source}")
    frames = {key: np.concatenate(values, axis=0) for key, values in frames.items()}
    total = len(frames["coord"])
    n_valid = int(valid_fraction * total)
    if n_valid == 0 or n_valid == total:
        raise ValueError("valid_fraction produces an empty train or validation split")
    order = np.arange(total)
    np.random.default_rng(seed).shuffle(order)
    train_indices, valid_indices = order[:-n_valid], order[-n_valid:]

    for output_dir, indices in ((train_dir, train_indices), (valid_dir, valid_indices)):
        output_dir = os.path.abspath(output_dir)
        os.makedirs(os.path.join(output_dir, "set.000"), exist_ok=True)
        shutil.copyfile(type_path, os.path.join(output_dir, "type.raw"))
        shutil.copyfile(type_map_path, os.path.join(output_dir, "type_map.raw"))
        for key, values in frames.items():
            np.save(os.path.join(output_dir, "set.000", f"{key}.npy"), values[indices])
    return len(train_indices), len(valid_indices)