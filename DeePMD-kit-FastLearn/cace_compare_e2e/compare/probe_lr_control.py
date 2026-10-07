#!/usr/bin/env python
# coding: utf-8
"""Does the author's cace long-range term buy accuracy, or is the checkpoint just
bigger?

``record_cace_valid.py`` records two rows per long-range checkpoint: the combined
prediction, and the same model with its Ewald term subtracted (``<tag>_sr``). That
second row is *not* a short-range model - it is a long-range model whose
short-range half was only ever trained with the long-range term present - so it
answers "how much of this prediction is the Ewald term" and not "what does a
short-range model of this representation reach".

The short-range control is a separate published run:
``fit-water-timing/fit-water-mp0-sr`` is the identical ``Cace`` representation,
the identical ``Atomwise`` fitting net and the identical 40+100+100+100 epoch
schedule on the identical 1593-frame file, with the Ewald head simply absent
(``fit-cace-nnp.py`` against ``fit_cace_new.py``; 24,572 parameters against
41,072). So this probe scores the matched pair and decomposes the difference:

  * how much the long-range term buys in total (energy and force RMSE), which is
    the honest version of the headline long-range gain;
  * whether that gain scales with the long-range term itself, per frame. A
    correction that carries real physics should help most on the frames where
    that correction is largest; extra capacity should help uniformly. The
    correlation of the per-frame gain with ``|F_lr|`` is that test, and it is the
    one that separates "learned long-range physics" from "a bigger network".

Reads only the two recorded ``_pred.npz`` files, so it loads no checkpoint and
needs no cace branch. Refuses to run if the two files disagree on the targets,
which is what would mean they were scored on different splits.

Usage:
    python probe_lr_control.py
    python probe_lr_control.py --lr-npz cace/cace_lrnew_valid_pred.npz
"""
import argparse
import datetime as dt
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.dirname(HERE)
CACE_DIR = os.path.join(E2E, 'cace')

DEFAULT_LR = os.path.join(CACE_DIR, 'cace_lrnew_valid_pred.npz')
DEFAULT_SR = os.path.join(CACE_DIR, 'cace_timing_sr_valid_pred.npz')
DEFAULT_OUT = os.path.join(CACE_DIR, 'lr_control_probe.json')


def rmse(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lr-npz', default=DEFAULT_LR,
                    help='long-range checkpoint prediction (needs E_lr and F_lr)')
    ap.add_argument('--sr-npz', default=DEFAULT_SR,
                    help='matched short-range-only checkpoint prediction')
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()

    for p in (args.lr_npz, args.sr_npz):
        if not os.path.exists(p):
            print(f'not found: {p}')
            return 1

    lr, sr = np.load(args.lr_npz), np.load(args.sr_npz)
    E_ref = lr['E_ref'].ravel()
    nf = len(E_ref)
    nat = lr['F_ref'].size // (nf * 3)

    # Both files must have been scored against the same frames, in the same order.
    if not np.allclose(E_ref, sr['E_ref'].ravel(), atol=1e-6):
        print('FAIL: the two files disagree on the targets, so they were not '
              'scored on the same split; the comparison would be meaningless')
        return 1
    for need in ('E_lr', 'F_lr'):
        if need not in lr:
            print(f'FAIL: {args.lr_npz} has no {need}; it was recorded without a '
                  f'long-range term, so there is nothing to test')
            return 1

    E_lr = lr['E_lr'].ravel()
    E_lr_arm, E_sr_arm = lr['E_cace'].ravel(), sr['E_cace'].ravel()
    F_ref = lr['F_ref'].reshape(nf, -1)
    F_lr = lr['F_lr'].reshape(nf, -1)
    F_lr_arm, F_sr_arm = lr['F_cace'].reshape(nf, -1), sr['F_cace'].reshape(nf, -1)

    e_lr, e_sr = rmse(E_lr_arm - E_ref), rmse(E_sr_arm - E_ref)
    f_lr, f_sr = rmse(F_lr_arm - F_ref), rmse(F_sr_arm - F_ref)

    # Per-frame gain: positive means the long-range term reduced that frame's error.
    g_e = np.abs(E_sr_arm - E_ref) - np.abs(E_lr_arm - E_ref)
    g_f = np.abs(F_sr_arm - F_ref).mean(1) - np.abs(F_lr_arm - F_ref).mean(1)

    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    tests = {
        'abs_E_lr': np.abs(E_lr),
        'abs_F_lr_max': np.abs(F_lr).max(1),
        'abs_F_lr_rms': np.sqrt((F_lr ** 2).mean(1)),
    }

    print(f'frames {nf} x {nat} atoms\n')
    print(f'{"arm":<22} {"E/atom RMSE":>12} {"F RMSE":>12}')
    print('-' * 48)
    for name, e, f in (('long-range (lr-npz)', e_lr, f_lr),
                       ('short-range (sr-npz)', e_sr, f_sr)):
        print(f'{name:<22} {e / nat:>12.4e} {f:>12.4e}')
    print(f'\nlong-range gain: energy {(e_lr - e_sr) / e_sr * 100:+.2f}%, '
          f'force {(f_lr - f_sr) / f_sr * 100:+.2f}%  '
          f'({e_sr / e_lr:.3f}x / {f_sr / f_lr:.3f}x better with the term)\n')

    print(f'the gain is not uniform: energy helps on {int((g_e > 0).sum())}/{nf} '
          f'frames, force on {int((g_f > 0).sum())}/{nf}')
    print('does the gain scale with the long-range term itself?')
    for k, x in tests.items():
        print(f'  corr(energy gain, {k:<14}) = {corr(g_e, x):+.3f}   '
              f'corr(force gain, {k:<14}) = {corr(g_f, x):+.3f}')
    print(f'  for reference, corr(gain, |E_ref|) = {corr(g_e, np.abs(E_ref)):+.3f} '
          f'(energy), {corr(g_f, np.abs(F_ref).max(1)):+.3f} (force): a nonzero '
          f'value there would just mean hard frames are hard for both arms')

    out = {
        'recorded_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'lr_npz': os.path.relpath(args.lr_npz, E2E),
        'sr_npz': os.path.relpath(args.sr_npz, E2E),
        'n_frames': nf,
        'n_atoms': nat,
        'note': ('The sr-npz arm is an independently trained short-range model of '
                 'the same representation and schedule; the <tag>_sr row recorded '
                 'by record_cace_valid.py is the long-range checkpoint with its '
                 'Ewald term subtracted, which is a different and much weaker '
                 'control. This probe measures the former.'),
        'long_range': {'e_rmse_ev_per_atom': e_lr / nat, 'f_rmse_ev_per_a': f_lr},
        'short_range': {'e_rmse_ev_per_atom': e_sr / nat, 'f_rmse_ev_per_a': f_sr},
        'gain_vs_short_range': {
            'energy_pct': (e_lr - e_sr) / e_sr * 100.0,
            'force_pct': (f_lr - f_sr) / f_sr * 100.0,
            'energy_ratio': e_sr / e_lr,
            'force_ratio': f_sr / f_lr,
        },
        'per_frame': {
            'energy_frames_helped': int((g_e > 0).sum()),
            'force_frames_helped': int((g_f > 0).sum()),
            'corr_energy_gain_vs_abs_E_lr': corr(g_e, np.abs(E_lr)),
            'corr_force_gain_vs_abs_F_lr_max': corr(g_f, np.abs(F_lr).max(1)),
            'corr_force_gain_vs_abs_F_lr_rms': corr(g_f, np.sqrt((F_lr ** 2).mean(1))),
            'corr_energy_gain_vs_abs_E_ref': corr(g_e, np.abs(E_ref)),
            'corr_force_gain_vs_abs_F_ref_max': corr(g_f, np.abs(F_ref).max(1)),
        },
    }
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
        fh.write('\n')
    print(f'\nwrote  {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
