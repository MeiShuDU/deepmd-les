#!/usr/bin/env python
# coding: utf-8
"""Where does the author's cace model lose its energy accuracy - and is it a
reference-convention artefact or the model itself?

``record_cace_valid.py`` records what the model scores. This records *why* the
score looks the way it does, by decomposing the energy error and by running the
same forward pass over the frames the model was trained on.

Three decompositions of the energy error, per frame and per atom:

  raw             as reported, ``mean((pred - ref)^2) ** 0.5``
  zero-offset     after subtracting the mean error. A constant reference-energy
                  mismatch (or a model that never learned the mean level) shows up
                  here as a large drop and nothing else moving.
  calibrated      after a least-squares ``pred = a * ref + b``. What is left when
                  both the level and the range are allowed to be off.

The train/valid pair is the discriminator: if the offset is a reference
convention, or a level the model never fit, then the model will show the *same*
offset on frames it was trained on. If instead the valid offset were a
generalization failure, train would be near zero. Zero train/valid gap + a large
offset = an under-fitting model, not an over-fitting one.

Writes ``cace/cace_offset_probe.json``. Refuses to write if its valid-split
numbers disagree with the already-recorded ``cace/cace_valid_metrics.json``,
which is the check that both used the same split, same model, same loader.

Usage:
    python probe_cace_offset.py                       # full probe on train+valid
    python probe_cace_offset.py --train-batches 40    # cheaper train sample
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.dirname(HERE)
CACE_DIR = os.path.join(E2E, 'cace')
sys.path.insert(0, HERE)
sys.path.insert(0, CACE_DIR)

DEFAULT_MODEL = os.path.join(CACE_DIR, 'best_model.pth')
DEFAULT_OUT = os.path.join(CACE_DIR, 'cace_offset_probe.json')
RECORDED = os.path.join(CACE_DIR, 'cace_valid_metrics.json')

# Same loader arguments as fit-cace-nnp.py / prep_water.py: the reference
# energies cace subtracts (and the deepmd targets were built with).
REF = {1: -187.6043857100553, 8: -93.80219285502734}
CUTOFF = 5.5


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def decompose(E_ref, E_pred, nat):
    """The three error levels, per frame and per atom (meV/atom for the per-atom)."""
    d = E_pred - E_ref
    raw = float(np.sqrt(np.mean(d ** 2)))
    zeroed = float(d.std())            # = RMSE after removing the mean error
    a, b = (float(v) for v in np.polyfit(E_ref, E_pred, 1))
    cal = float(np.std(E_pred - (a * E_ref + b)))
    r = float(np.corrcoef(E_pred, E_ref)[0, 1])
    return {
        'n_frames': int(len(E_ref)),
        'bias_ev_per_frame': float(d.mean()),
        'bias_ev_per_atom': float(d.mean() / nat),
        'rmse_raw_ev_per_frame': raw,
        'rmse_zero_offset_ev_per_frame': zeroed,
        'rmse_calibrated_ev_per_frame': cal,
        'rmse_raw_mev_per_atom': 1000.0 * raw / nat,
        'rmse_zero_offset_mev_per_atom': 1000.0 * zeroed / nat,
        'rmse_calibrated_mev_per_atom': 1000.0 * cal / nat,
        'ols_slope': a,
        'ols_intercept_ev_per_frame': b,
        'pearson_r': r,
        'pearson_r2': r ** 2,
        'target_mean_ev_per_frame': float(E_ref.mean()),
        'target_std_ev_per_frame': float(E_ref.std()),
        'pred_std_ev_per_frame': float(E_pred.std()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--train-batches', type=int, default=100,
                    help='loader batches of 4 train frames to score (default 100 '
                         '= 400 frames); 0 = all 1434')
    ap.add_argument('--valid-batches', type=int, default=10 ** 9)
    args = ap.parse_args()

    import cace
    from cace.tasks.load_data import get_dataset_from_xyz, load_data_loader
    cace.tools.setup_logger(level='ERROR')
    torch.set_default_dtype(torch.float32)

    collection = get_dataset_from_xyz(
        train_path=os.path.join(CACE_DIR, 'water.xyz'), valid_fraction=0.1,
        seed=1, cutoff=CUTOFF, data_key={'energy': 'energy', 'forces': 'force'},
        atomic_energies=REF)
    nat = int(collection.train[0].get_global_number_of_atoms())

    model = torch.load(args.model, map_location='cpu', weights_only=False)
    model.eval()

    def score(split, maxb):
        E_ref, E_pred = [], []
        for i, batch in enumerate(load_data_loader(collection, split, 4)):
            if i >= maxb:
                break
            bd = batch.to_dict()
            bd['positions'] = bd['positions'].detach().clone().requires_grad_(True)
            pred = model(bd, training=False)
            E_ref.append(np.asarray(bd['energy'].detach()))
            E_pred.append(np.asarray(pred['CACE_energy'].detach()))
        return np.concatenate(E_ref), np.concatenate(E_pred)

    train_b = args.train_batches if args.train_batches > 0 else 10 ** 9
    print(f'model  {args.model}\n       sha256[:12] = {sha256(args.model)[:12]}')
    print(f'data   {len(collection.train)} train / {len(collection.valid)} valid '
          f'frames x {nat} atoms\n')
    splits = {}
    for split, maxb in (('train', train_b), ('valid', args.valid_batches)):
        E_ref, E_pred = score(split, maxb)
        splits[split] = decompose(E_ref, E_pred, nat)
        s = splits[split]
        print(f'--- {split} ({s["n_frames"]} frames) ---')
        print(f'  bias            {s["bias_ev_per_frame"]:+8.4f} eV/frame  '
              f'({1000 * s["bias_ev_per_atom"]:+.3f} meV/atom)')
        print(f'  RMSE raw        {s["rmse_raw_mev_per_atom"]:8.3f} meV/atom')
        print(f'  RMSE -bias      {s["rmse_zero_offset_mev_per_atom"]:8.3f} meV/atom')
        print(f'  RMSE calibrated {s["rmse_calibrated_mev_per_atom"]:8.3f} meV/atom')
        print(f'  OLS pred = {s["ols_slope"]:.4f} * ref {s["ols_intercept_ev_per_frame"]:+.4f}'
              f'   Pearson r {s["pearson_r"]:.5f}  (r2 {s["pearson_r2"]:.5f})\n')

    # Guard: this probe's valid numbers must be the ones already on record.
    if os.path.exists(RECORDED):
        with open(RECORDED) as fh:
            rec = json.load(fh)
        # the record stores e_rmse in eV/atom; this probe reports meV/atom
        want = 1000.0 * rec['metrics']['cace']['e_rmse']
        got = splits['valid']['rmse_raw_mev_per_atom']
        if abs(got - want) > 0.01 * want:
            print(f'FAIL: probe valid {got:.3f} meV/atom vs recorded '
                  f'{want:.3f} meV/atom - different loader, split or model')
            return 1
        print(f'guard  probe valid = recorded cace e_rmse ({got:.3f} vs '
              f'{want:.3f} meV/atom)')

    out = {
        'recorded_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'model': os.path.relpath(args.model, E2E),
        'model_sha256': sha256(args.model),
        'n_atoms': nat,
        'atomic_energies': {str(k): v for k, v in REF.items()},
        'note': ('Energy error decomposition for the author\'s cace checkpoint. '
                 'The train row is scored on the frames the model was trained on, '
                 'so a valid row that looks no better than the train row means the '
                 'model is under-fitting rather than over-fitting.'),
        'splits': splits,
    }
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
        fh.write('\n')
    print(f'wrote  {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
