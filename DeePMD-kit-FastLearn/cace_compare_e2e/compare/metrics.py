#!/usr/bin/env python
# coding: utf-8
"""Per-atom energy / force metrics for every arm of the cace comparison.

All arms are scored against the SAME held-out split - the 159 frames of
``cace_compare_e2e/water/valid`` - so the numbers are directly comparable:

  sr            deepmd, plain short-range (no LES at all)
  hyb_lrw1      deepmd hybrid_ener, local_charge (NN), lr_weight 1.0
  hyb_lrw001    deepmd hybrid_ener, local_charge (NN), lr_weight 0.01
  freeze_spce   deepmd hybrid_ener, freeze_charge = SPC/E table, lr_weight 1.0
  cace          the author's published best_model.pth, COMBINED prediction
  cace_sr       the same model with the long-range term removed
                (CACE_energy - 0.01 * ewald_potential)

hyb_lrw1 vs freeze_spce is the charge-layer benchmark: both run at lr_weight 1.0,
so the only difference is whether q comes from a learned function of the local
descriptor or from a fixed per-type SPC/E table. freeze_spce is the classical
Ewald end of the spectrum, so it is the honest floor for a learned charge layer
to beat; hyb_lrw001 is the cace-matched arm (lr_weight 0.01), not that floor.

The energy target is the residual (total minus the per-element reference), the
same quantity both codes were trained on.

Metrics are the standard per-atom energy RMSE/MAE (so a model with a large frame
size is not unfairly penalised) plus force RMSE/MAE over all components, and R^2
for both. R^2 is computed against the target's own variance, so it is 0 for the
"always predict the mean" model and 1 for a perfect fit.

A single checkpoint is not a converged value: on the full set the energy RMSE
still swings by 1.1x-2.3x between adjacent late checkpoints while force moves
under 1%. So ``--late N`` scores the N highest-step checkpoints and reports the
mean, which is the only stable way to compare arms.

Usage:
    python metrics.py                 # every arm whose model file exists
    python metrics.py sr cace         # only the named arms
    python metrics.py --batch 32      # frames per forward (deepmd arms)
    python metrics.py --late 3        # mean over the 3 latest checkpoints
"""
import argparse
import copy
import glob
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # .../cace_compare_e2e
VALID = os.path.join(ROOT, 'water', 'valid')
DEEPMD_RUNS = os.path.join(ROOT, 'deepmd', 'runs100k')
CACE_DIR = os.path.join(ROOT, 'cace')

DEEPMD_ARMS = ['sr', 'hyb_lrw1', 'hyb_lrw001', 'freeze_spce']
CACE_ARMS = ['cace', 'cace_sr']
ALL_ARMS = DEEPMD_ARMS + CACE_ARMS


def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------------------------------------------------------- #
# targets: the shared held-out split
# --------------------------------------------------------------------------- #
def load_valid():
    """(coord [nf,n,3], atype [nf,n], box [nf,9], energy [nf], force [nf,n,3])."""
    d = os.path.join(VALID, 'set.000')
    coord = np.load(os.path.join(d, 'coord.npy'))
    box = np.load(os.path.join(d, 'box.npy'))
    energy = np.load(os.path.join(d, 'energy.npy'))
    force = np.load(os.path.join(d, 'force.npy'))
    atype = np.array(
        [int(t) for t in open(os.path.join(VALID, 'type.raw')).read().split()],
        dtype=np.int64,
    )
    nat = atype.size
    nf = coord.shape[0]
    return (
        coord.reshape(nf, nat, 3).astype(np.float64),
        np.tile(atype, (nf, 1)),
        box.reshape(nf, 9).astype(np.float64),
        energy.astype(np.float64),
        force.reshape(nf, nat, 3).astype(np.float64),
        nat,
    )


# --------------------------------------------------------------------------- #
# deepmd arms
# --------------------------------------------------------------------------- #
def late_checkpoints(run_dir, n):
    """The n highest-step checkpoints in a run dir, highest first.

    Returns [(step, path)]; step is None for a bare ``model.ckpt.pt``.
    """
    steps = []
    for path in glob.glob(os.path.join(run_dir, 'model.ckpt-*.pt')):
        m = re.search(r'model\.ckpt-(\d+)\.pt$', path)
        if m:
            steps.append((int(m.group(1)), path))
    if steps:
        return sorted(steps, reverse=True)[:n]
    final = os.path.join(run_dir, 'model.ckpt.pt')
    if os.path.exists(final):
        return [(None, final)]
    return []


def latest_checkpoint(run_dir):
    """The highest-step model.ckpt-*.pt in a run directory, or model.ckpt.pt."""
    found = late_checkpoints(run_dir, 1)
    if not found:
        return None, None
    step, path = found[0]
    return path, step


def load_deepmd(run_dir, device, ckpt=None):
    """Rebuild a trained model from its own checkpoint metadata.

    The model params stored in the checkpoint are used rather than input.json:
    they are the normalised params the run actually trained under, so a later
    edit to the config file cannot silently change what is being evaluated.
    """
    from deepmd.pt.model.model import get_model
    from deepmd.pt.train.wrapper import ModelWrapper

    if ckpt is None:
        ckpt, step = latest_checkpoint(run_dir)
    else:
        m = re.search(r'model\.ckpt-(\d+)\.pt$', ckpt)
        step = int(m.group(1)) if m else None
    if ckpt is None:
        return None, None
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    if 'model' in state:
        state = state['model']
    model = get_model(copy.deepcopy(state['_extra_state']['model_params']))
    ModelWrapper(model).load_state_dict(state)
    model.to(device).eval()
    return model, step


def predict_deepmd(run_dir, device, batch=16, ckpt=None):
    data = load_valid()
    coord, atype, box, _, _, nat = data
    model, step = load_deepmd(run_dir, device, ckpt=ckpt)
    if model is None:
        return None
    nf = coord.shape[0]
    E = np.zeros(nf, dtype=np.float64)
    F = np.zeros((nf, nat, 3), dtype=np.float64)
    # The data is float64; the model follows its own precision setting, which is
    # float64 for these runs but is a config knob. Follow the model, not the data.
    dtype = next(model.parameters()).dtype
    # No torch.no_grad() here: the model produces forces with an internal
    # autograd.grad, so the graph has to stay enabled. Outputs are detached.
    for lo in range(0, nf, batch):
        hi = min(lo + batch, nf)
        out = model(
            torch.tensor(coord[lo:hi], device=device, dtype=dtype),
            torch.tensor(atype[lo:hi], device=device),
            box=torch.tensor(box[lo:hi], device=device, dtype=dtype),
        )
        E[lo:hi] = out['energy'].detach().reshape(hi - lo).double().cpu().numpy()
        F[lo:hi] = out['force'].detach().double().cpu().numpy()
    return {'E': E, 'F': F, 'step': step, 'label': os.path.basename(run_dir)}


# --------------------------------------------------------------------------- #
# cace arms
# --------------------------------------------------------------------------- #
def predict_cace(nat):
    """Both cace arms in one pass: the loader gives the combined prediction.

    cace reports forces flattened as (n_frames * nat, 3), so they are reshaped to
    the frame layout the deepmd arms use.
    """
    sys.path.insert(0, CACE_DIR)
    from inspect_cace_les import predict

    q, E_lr, E_tot, E_sr, E_ref, F_lr, F_sr, F_tot = predict()
    reshape = lambda a: np.asarray(a).reshape(-1, nat, 3)
    return {
        'cace': {'E': np.asarray(E_tot), 'F': reshape(F_tot)},
        'cace_sr': {'E': np.asarray(E_sr), 'F': reshape(F_sr)},
        'E_ref': np.asarray(E_ref),
    }


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metrics(E_pred, F_pred, E_ref, F_ref, nat):
    e_res = np.asarray(E_pred) - np.asarray(E_ref)
    f_res = (np.asarray(F_pred) - np.asarray(F_ref)).reshape(-1)
    e_tgt = np.asarray(E_ref)
    f_tgt = np.asarray(F_ref).reshape(-1)
    return {
        'e_rmse': float(np.sqrt(np.mean(e_res ** 2)) / nat),
        'e_mae': float(np.mean(np.abs(e_res)) / nat),
        'e_r2': float(1.0 - np.sum(e_res ** 2) / np.sum((e_tgt - e_tgt.mean()) ** 2)),
        'f_rmse': float(np.sqrt(np.mean(f_res ** 2))),
        'f_mae': float(np.mean(np.abs(f_res))),
        'f_r2': float(1.0 - np.sum(f_res ** 2) / np.sum((f_tgt - f_tgt.mean()) ** 2)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('arms', nargs='*', default=None,
                    help=f'arms to score (default: all). one of {ALL_ARMS}')
    ap.add_argument('--batch', type=int, default=16, help='frames per forward')
    ap.add_argument('--late', type=int, default=1,
                    help='score the N highest-step checkpoints and average them '
                         '(default 1)')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    arms = args.arms or ALL_ARMS
    unknown = [a for a in arms if a not in ALL_ARMS]
    if unknown:
        ap.error(f'unknown arm(s) {unknown}; choose from {ALL_ARMS}')

    device = resolve_device(args.device)
    coord, _, _, E_ref, F_ref, nat = load_valid()
    print(f'valid split: {E_ref.size} frames x {nat} atoms '
          f'(residual-energy target) on {device}')
    print(f'  target per-atom energy RMSE scale = '
          f'{np.sqrt(np.mean((E_ref - E_ref.mean()) ** 2)) / nat:.4e} eV/atom')

    rows = []
    for arm in arms:
        try:
            if arm in DEEPMD_ARMS:
                run_dir = os.path.join(DEEPMD_RUNS, arm)
                cks = late_checkpoints(run_dir, args.late)
                if not cks:
                    print(f'[skip] {arm}: no checkpoint in {run_dir}')
                    continue
                per_ckpt = []
                for _step, ckpt in cks:
                    res = predict_deepmd(run_dir, device, args.batch, ckpt=ckpt)
                    per_ckpt.append(
                        (res['step'],
                         metrics(res['E'], res['F'], E_ref, F_ref, nat))
                    )
                # Report the mean, but print the per-checkpoint values too: one
                # number invites a ranking the checkpoints do not support.
                print(f'[arm] {arm}: ' + '  '.join(
                    f'{s}: e_rmse={mm["e_rmse"]:.4e} f_rmse={mm["f_rmse"]:.4e}'
                    for s, mm in per_ckpt))
                m = {k: float(np.mean([x[1][k] for x in per_ckpt]))
                     for k in per_ckpt[0][1]}
                note = ('step %s' % per_ckpt[0][0] if len(per_ckpt) == 1
                        else f'mean of {len(per_ckpt)} ckpt')
                rows.append((arm, m, note))
            else:
                pred = predict_cace(nat)
                # The loader carries its own copy of the target. If it is not the
                # same array in the same order, scoring against ours would be
                # meaningless, so say so instead of printing a nice-looking number.
                if not np.allclose(np.sort(pred['E_ref']), np.sort(E_ref)):
                    print(f'[fail] {arm}: cace loader targets differ from '
                          f'{VALID}/set.000/energy.npy')
                    continue
                if not np.allclose(pred['E_ref'], E_ref):
                    print(f'[note] {arm}: cace target order differs from the npy '
                          f'order (same set, sorted values match)')
                res = pred[arm]
                rows.append((arm, metrics(res['E'], res['F'], E_ref, F_ref, nat),
                             'best_model.pth'))
        except Exception as exc:  # noqa: BLE001 - one bad arm must not hide the rest
            print(f'[fail] {arm}: {type(exc).__name__}: {exc}')

    if not rows:
        print('\nno arm could be scored')
        return 1

    print()
    hdr = (f'{"arm":<12} {"E/atom RMSE":>12} {"E/atom MAE":>11} {"E R2":>9} '
           f'{"F RMSE":>10} {"F MAE":>10} {"F R2":>9}  {"source":<16}')
    print(hdr)
    print('-' * len(hdr))
    for arm, m, note in rows:
        print(f'{arm:<12} {m["e_rmse"]:>12.4e} {m["e_mae"]:>11.4e} '
              f'{m["e_r2"]:>9.5f} {m["f_rmse"]:>10.4e} {m["f_mae"]:>10.4e} '
              f'{m["f_r2"]:>9.5f}  {note:<16}')
    missing = [a for a in ALL_ARMS if a not in [r[0] for r in rows]]
    if missing:
        print(f'\nnot scored: {missing}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
