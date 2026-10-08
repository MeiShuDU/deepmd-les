#!/usr/bin/env python
# coding: utf-8
"""Export the DeePMD-kit-FastLearn npy data to extended XYZ, so the cace arms
can train on exactly the same frames as the deepmd arms.

The cace ``fit-water-timing`` scripts read XYZ only. The FastLearn data is the
deepmd-native copy of the same MP-0 64-H2O system, so this is a format change,
not a data change: same frames, same energies, same forces, same cell.

Split, matching what the deepmd arms train and validate on:
  * train  ``data_0`` + ``data_1`` (set.000 + set.001) + ``data_2`` -> 320 frames
  * valid  ``data_3``                                              ->  80 frames

Energies and forces are carried the way the author's own ``water.xyz`` carries
them, because ``cace.data.AtomicData.from_atoms`` looks them up by name: the
energy in ``atoms.info[...]`` and the forces in ``atoms.arrays['force']``. The
array is named ``force``, singular - the fitness script passes
``data_key={'forces': 'force'}`` and that lookup is a plain
``atoms.arrays.get('force')``, so a ``forces`` array would silently come back as
no target at all.

The energy key is ``E_total`` rather than the conventional ``energy`` for a
reason worth knowing about. ``AtomicData.from_atoms`` reads it as
``atoms.info.get(data_key['energy'], None)``, and with ase >= 3.x a bare
``energy=`` in an extended-XYZ header is parsed into a *calculator*, not into
``atoms.info``, so ``info['energy']`` comes back None and the fit proceeds with
no energy target at all - forces only, silently. That is not a property of our
file: the author's own ``water.xyz`` behaves the same way under the ase installed
here (3.26.0), so the shipped script cannot fit energy in this environment. A
non-reserved key such as ``E_total`` does land in ``atoms.info``, so that is what
is written, and the runner passes ``data_key={'energy': 'E_total', ...}``. The
recipe is unchanged; only the name the loader can see is.

Usage:
    python export_data.py                       # writes xyz/{train,valid}.xyz
    python export_data.py --data-root ... --out-dir ...
"""
import argparse
import glob
import os

import numpy as np
from ase import Atoms
from ase.io import write

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(HERE, '..', 'data')
# named ``xyz`` rather than ``data`` on purpose: the deepmd arms read the
# original npy data one directory up, and a same-named sibling invites reading
# the wrong one
DEFAULT_OUT_DIR = os.path.join(HERE, 'xyz')

TRAIN_DIRS = ('data_0', 'data_1', 'data_2')
VALID_DIR = 'data_3'


def systems(data_root, names):
    """Every set.* under each named system dir, in sorted order."""
    for name in names:
        for s in sorted(glob.glob(os.path.join(data_root, name, 'set.*'))):
            yield name, s


def load_frames(data_root, names, type_map):
    """Read coord/box/energy/force/type and build one Atoms per frame."""
    frames = []
    for name, s in systems(data_root, names):
        type_raw = os.path.join(data_root, name, 'type.raw')
        # type.raw is a plain int column; type.npy is a pickled array, so the
        # raw file is the one to read.
        t = np.loadtxt(type_raw).astype(int)
        coord = np.load(os.path.join(s, 'coord.npy'))
        box = np.load(os.path.join(s, 'box.npy'))
        energy = np.load(os.path.join(s, 'energy.npy'))
        force = np.load(os.path.join(s, 'force.npy'))
        nat = len(t)
        nf = coord.shape[0]
        assert coord.shape == (nf, nat * 3), coord.shape
        assert force.shape == (nf, nat * 3), force.shape
        for f in range(nf):
            atoms = Atoms(
                numbers=[type_map[i] for i in t],
                positions=coord[f].reshape(nat, 3),
                cell=box[f].reshape(3, 3),
                pbc=True,
            )
            atoms.info['E_total'] = float(energy[f])
            atoms.arrays['force'] = force[f].reshape(nat, 3)
            atoms.arrays['Z'] = np.array([type_map[i] for i in t], dtype=int)
            frames.append(atoms)
    return frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    # type index -> atomic number, from the type_map the deepmd arms also use.
    symbols = open(os.path.join(args.data_root, 'data_0', 'type_map.raw')).read().split()
    type_map = {i: _symbol_to_z(s) for i, s in enumerate(symbols)}
    print(f'type_map {symbols} -> {type_map}')

    os.makedirs(args.out_dir, exist_ok=True)
    for split, names in (('train', TRAIN_DIRS), ('valid', (VALID_DIR,))):
        frames = load_frames(args.data_root, names, type_map)
        path = os.path.join(args.out_dir, f'{split}.xyz')
        write(path, frames, format='extxyz')
        e = np.array([a.info['E_total'] for a in frames])
        fmax = max(np.abs(a.arrays['force']).max() for a in frames)
        print(f'{split:5s} {len(frames):4d} frames x {len(frames[0])} atoms -> {path}')
        print(f'      E {e.min():.4f} .. {e.max():.4f} eV/frame (std {e.std():.4f}),'
              f' max|F| {fmax:.4f} eV/A')

        # read back the way the cace loader will: info['E_total'] and
        # arrays['force'], through ase, from the file on disk. extxyz is text at
        # a fixed number of decimals, so forces return to ~5e-9 eV/A rather than
        # exactly; that is the format's floor, 7 orders below the F RMSE of
        # interest, and it is reported rather than hidden.
        from ase.io import read
        back = read(path, ':')
        assert len(back) == len(frames), (len(back), len(frames))
        assert all('E_total' in a.info for a in back), 'loader would read no energy'
        assert np.allclose([a.info['E_total'] for a in back], e, atol=1e-6)
        dfmax = 0.0
        for a, b in zip(back, frames):
            assert 'force' in a.arrays, 'cace would read no forces from this file'
            dfmax = max(dfmax, float(np.abs(a.arrays['force'] - b.arrays['force']).max()))
            assert np.allclose(a.get_cell(), b.get_cell(), atol=1e-9)
            # Z comes back as ase's own numbers array, which is what cace reads
            assert np.array_equal(a.numbers, b.numbers)
        assert dfmax < 1e-8, dfmax
        print(f'      read-back OK (cell, species, Z, E exact; max|dF| {dfmax:.1e} eV/A)')
    return 0


def _symbol_to_z(symbol):
    from ase.data import atomic_numbers
    return int(atomic_numbers[symbol])


if __name__ == '__main__':
    raise SystemExit(main())
