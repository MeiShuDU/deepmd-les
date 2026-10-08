#!/usr/bin/env python
# coding: utf-8
"""Inspect a trained cace-LES checkpoint on the water benchmark.

Answers, empirically, the design question: with NO fixed-charge baseline, what is
the actual magnitude of the learned latent charges q and of the long-range (Ewald)
energy E_lr that a trained cace-LES produces on the 64-H2O data?

Runs on the same split/keys as the author's training scripts (valid_fraction 0.1,
seed 1, data_key {'energy':'energy','forces':'force'}, refs {H,O}). Extracts:
  * q: the per-atom latent feature that feeds EwaldPotential (shape [n, n_out])
  * E_lr_raw: ewald_potential (before the mix weight), per frame
  * E_tot: CACE_energy, per frame. With the pre-update script this is the
    COMBINED prediction, i.e. short-range + 0.01 * E_lr_raw (CombinePotential
    maps it to 'CACE_energy'); with the FeatureAdd checkpoint it is
    short-range + 1.0 * E_lr_raw (FeatureAdd sums its inputs at unit weight).
    The short-range-only arm is therefore E_tot - w * E_lr_raw for the matching
    w, which is a parameter here for exactly that reason.
  * ref residual target per frame, for scale comparison

The cace package is imported lazily, inside the functions, and from a
caller-supplied root. That is deliberate: a checkpoint pickles its module classes
by name, so which cace checkout is imported decides what code runs, and two
checkpoints can require different branches. Importing at module scope would cache
whichever branch happened to be first on sys.path and silently score the other
checkpoint with the wrong code.

Usage:
    python inspect_cace_les.py                     # the default checkpoint
    python inspect_cace_les.py --lr-weight 1.0     # FeatureAdd-style mixing
"""
import argparse
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WATER = os.path.join(HERE, 'water.xyz')
MODEL = os.path.join(HERE, 'best_model.pth')

# The pre-update script wraps its two potentials in a CombinePotential that mixes
# the long-range one in at 0.01; the updated FeatureAdd checkpoint adds it at 1.0.
DEFAULT_LR_WEIGHT = 0.01
CUTOFF = 5.5
ATOMIC_ENERGIES = {1: -187.6043857100553, 8: -93.80219285502734}


def _import_cace(cace_root):
    """Import cace from ``cace_root``, asserting that is where it came from.

    Inserting ``cace_root`` here rather than at module scope is what makes the
    branch choice take effect: a module already imported under another branch
    stays cached, so the assert is what turns a silent wrong-branch run into an
    error.
    """
    if cace_root:
        root = os.path.abspath(cace_root)
        sys.path.insert(0, root)
        expected = root + os.sep
    else:
        expected = None

    import cace
    if expected is not None and not os.path.abspath(cace.__file__).startswith(expected):
        raise RuntimeError(
            f'cace imported from {cace.__file__}, not from {cace_root}; '
            f'another cace is already on sys.path')
    return cace


def _loader(cace_root=None, water=None, batch_size=2):
    """(cace, valid loader, collection) for the split this checkpoint trained on."""
    cace = _import_cace(cace_root)

    from cace.tasks.load_data import get_dataset_from_xyz, load_data_loader

    cace.tools.setup_logger(level='WARNING')
    torch.set_default_dtype(torch.float32)

    collection = get_dataset_from_xyz(
        train_path=water or WATER,
        valid_fraction=0.1,
        seed=1,
        cutoff=CUTOFF,
        data_key={'energy': 'energy', 'forces': 'force'},
        atomic_energies=ATOMIC_ENERGIES,
    )
    return cace, load_data_loader(collection, 'valid', batch_size), collection


def _load(cace_root=None, model_path=None, water=None, batch_size=2):
    """The loader plus the checkpoint, unpickled against that branch's classes."""
    cace, loader, collection = _loader(cace_root, water, batch_size)
    model = torch.load(model_path or MODEL, map_location='cpu', weights_only=False)
    model.eval()
    return cace, loader, model, collection


def split_identity(cace_root=None, water=None):
    """What the cace loader's valid split actually is, for a cross-check.

    (positions [nf,n,3], cell [nf,3,3], type numbers [nf,n], energy [nf] or None).
    The positions and cell are compared against our own water/valid before any
    metric is written: the loader derives its split from the xyz with its own
    seed, and a metric against a different split is worse than no metric.

    Deliberately does not load a checkpoint: a checkpoint from one cace branch
    does not unpickle against another branch's classes, and the split is a
    property of the loader alone.
    """
    _cace, _ldr, collection = _loader(cace_root, water)
    frames = list(collection.valid)
    pos = np.array([f.get_positions() for f in frames], dtype=np.float64)
    cell = np.array([f.get_cell()[:] for f in frames], dtype=np.float64)
    z = np.array([f.get_atomic_numbers() for f in frames], dtype=np.int64)
    return pos, cell, z, None


def _long_range_forces(model, bd, pred):
    """The long-range forces, for the diagnostic "long range removed" arm.

    The pre-update script built these with a second potential whose ``forces_lr``
    was a ``Forces`` module reading ``ewald_potential``; the updated FeatureAdd
    design deliberately has no such module, because deriving the total force from
    one autograd on the summed energy is the whole optimization. So where the
    model does not emit ``ewald_forces``, recover it the way that module did:
    differentiate ``ewald_potential`` instead of the total energy.
    """
    if 'ewald_forces' in pred:
        return pred['ewald_forces'].detach().cpu().numpy()

    swapped = [(m, m.energy_key) for m in model.modules()
               if type(m).__name__ == 'Forces' and getattr(m, 'energy_key', None)]
    if not swapped:
        raise RuntimeError('no Forces module to derive long-range forces with')
    try:
        for m, _old in swapped:
            m.energy_key = 'ewald_potential'
        # a fresh forward: the model's own Forces call already consumed the graph
        return model(bd, training=False)['CACE_forces'].detach().cpu().numpy()
    finally:
        for m, old in swapped:
            m.energy_key = old


def predict(max_batches=10**9, lr_weight=DEFAULT_LR_WEIGHT, cace_root=None,
            model_path=None, water=None, batch_size=2):
    """Run the checkpoint over the valid split; returns the raw arrays.

    ``max_batches`` caps the number of loader batches (default: all of them).
    ``lr_weight`` is the mix weight the checkpoint's long-range term enters its
    reported energy with, and is what the SR-only arm subtracts.
    """
    _cace, loader, model, _collection = _load(
        cace_root, model_path, water, batch_size)

    results = {'q': [], 'E_lr': [], 'E_tot': [], 'E_ref': [], 'F_lr': [], 'F_tot': []}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        bd = batch.to_dict()
        bd['positions'] = bd['positions'].detach().clone().requires_grad_(True)
        pred = model(bd, training=False)
        results['q'].append(pred['q'].detach().cpu().numpy())
        results['E_lr'].append(np.array(pred['ewald_potential'].detach()))
        results['E_tot'].append(np.array(pred['CACE_energy'].detach()))
        if 'energy' in bd:
            results['E_ref'].append(np.array(bd['energy'].detach()))
        results['F_lr'].append(_long_range_forces(model, bd, pred))
        results['F_tot'].append(pred['CACE_forces'].detach().cpu().numpy())

    q = np.concatenate(results['q'])
    E_lr = np.concatenate(results['E_lr'])
    E_tot = np.concatenate(results['E_tot'])
    E_ref = (np.concatenate(results['E_ref']) if results['E_ref']
             else np.full(E_tot.shape, np.nan))
    F_lr = np.concatenate(results['F_lr'])
    F_tot = np.concatenate(results['F_tot'])
    # The mix weight is what the long-range term enters the reported energy with,
    # so the short-range-only arm is the combined prediction minus that much.
    E_sr = E_tot - lr_weight * E_lr
    F_sr = F_tot - lr_weight * F_lr
    return q, E_lr, E_tot, E_sr, E_ref, F_lr, F_sr, F_tot


def predict_sr(max_batches=10**9, cace_root=None, model_path=None, water=None,
               batch_size=2):
    """Run a short-range-only checkpoint over the valid split; (E_tot, E_ref, F_tot).

    The ``fit-water-timing/fit-water-mp0-sr`` control is the same representation
    and fitting net as the long-range checkpoints with the Ewald head simply
    absent, so it emits ``CACE_energy`` / ``CACE_forces`` and no ``q`` or
    ``ewald_potential`` at all. That makes it a genuinely independent
    short-range model, unlike the ``<tag>_sr`` rows elsewhere in this harness,
    which are a long-range checkpoint with its Ewald term subtracted.
    """
    _cace, loader, model, _collection = _load(
        cace_root, model_path, water, batch_size)

    E_tot, E_ref, F_tot = [], [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        bd = batch.to_dict()
        bd['positions'] = bd['positions'].detach().clone().requires_grad_(True)
        pred = model(bd, training=False)
        if i == 0 and 'ewald_potential' in pred:
            raise RuntimeError(
                'this checkpoint emits ewald_potential, so it has a long-range '
                'head: score it with predict(), not predict_sr()')
        E_tot.append(np.array(pred['CACE_energy'].detach()))
        if 'energy' in bd:
            E_ref.append(np.array(bd['energy'].detach()))
        F_tot.append(pred['CACE_forces'].detach().cpu().numpy())

    E_tot = np.concatenate(E_tot)
    F_tot = np.concatenate(F_tot)
    E_ref = (np.concatenate(E_ref) if E_ref else np.full(E_tot.shape, np.nan))
    return E_tot, E_ref, F_tot


def report(lr_weight=DEFAULT_LR_WEIGHT, cace_root=None, model_path=None):
    q, E_lr, E_tot, E_sr, E_ref, F_lr, F_sr, F_tot = predict(
        lr_weight=lr_weight, cace_root=cace_root, model_path=model_path)

    print(f'\nq shape (atoms, n_out) = {q.shape}')
    for c in range(q.shape[1]):
        qc = q[:, c]
        print(f'  q[ch {c}]: min={qc.min():+.4e} max={qc.max():+.4e} '
              f'mean={qc.mean():+.4e} std={qc.std():.4e}')
    print(f'q |.|max overall = {np.abs(q).max():+.4e}')

    print(f'\nE_lr raw (unweighted) per frame: '
          f'min={E_lr.min():+.4f} max={E_lr.max():+.4f} '
          f'mean={E_lr.mean():+.4f} std={E_lr.std():.4f} eV')
    print(f'E_lr * {lr_weight:g} (as combined):        '
          f'mean={(lr_weight*E_lr).mean():+.4f} std={(lr_weight*E_lr).std():.4f} eV')
    print(f'E_tot (combined, = CACE_energy):    '
          f'min={E_tot.min():+.4f} max={E_tot.max():+.4f} mean={E_tot.mean():+.4f}')
    print(f'E_sr  (= E_tot - {lr_weight:g}*E_lr, the SR-only arm): '
          f'min={E_sr.min():+.4f} max={E_sr.max():+.4f} mean={E_sr.mean():+.4f}')
    print(f'E_ref (residual) :  min={E_ref.min():+.4f} max={E_ref.max():+.4f} '
          f'mean={E_ref.mean():+.4f} eV/frame')

    # Long-range FORCE: this is what the mix weight attenuates in the combined
    # force, and it is what a deepmd-les run must reproduce.
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    print(f'\nF_lr raw      per-component RMS = {rms(F_lr):.4e} eV/A '
          f'max|F| = {np.abs(F_lr).max():.4e}')
    print(f'F_lr * {lr_weight:g}   per-component RMS = {rms(lr_weight * F_lr):.4e} '
          f'eV/A max|F| = {np.abs(lr_weight * F_lr).max():.4e}')
    print(f'F_sr          per-component RMS = {rms(F_sr):.4e} eV/A '
          f'max|F| = {np.abs(F_sr).max():.4e}')
    print(f'F_tot         per-component RMS = {rms(F_tot):.4e} eV/A')
    print(f'ratio RMS({lr_weight:g}*F_lr)/RMS(F_tot) = '
          f'{rms(lr_weight * F_lr) / rms(F_tot):.4e}')

    # per-atom count for per-atom scales
    nat = q.shape[0] // len(E_lr)
    print(f'\nper-atom: |E_lr|/frame ~ {np.abs(E_lr).mean()/nat:+.4e} eV/atom '
          f'(raw), {(lr_weight*np.abs(E_lr)).mean()/nat:+.4e} eV/atom (weighted)')
    print(f'ref residual/atom  ~ {np.abs(E_ref).mean()/nat:+.4e} eV/atom')
    print(f'for scale: fixed-baseline |q| = 1.0 (O), 0.5 (H)  vs  learned q std ~ '
          f'{q.std():.3f}, max |q| ~ {np.abs(q).max():.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--cace-root', default=None,
                    help='cace checkout to import (default: the installed one)')
    ap.add_argument('--lr-weight', type=float, default=DEFAULT_LR_WEIGHT,
                    help='weight the long-range term enters CACE_energy with')
    args = ap.parse_args()
    report(lr_weight=args.lr_weight, cace_root=args.cace_root,
           model_path=args.model)


if __name__ == '__main__':
    main()
