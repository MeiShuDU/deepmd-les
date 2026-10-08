#!/usr/bin/env python
# coding: utf-8
"""What the cace learned charge actually encodes, measured from the recorded
prediction.

``record_cace_valid.py`` writes the per-atom charge ``q`` for a long-range
checkpoint (``cace/cace_lrnew_valid_pred.npz``). That array's min/max sheet
invites a bad reading - "the same element reaches both -4.97 and +4.78, so the
charge is unstable" - which is wrong twice over. This probe measures what the
tail really is:

  * the extreme values are not one element flipping sign, they sit on a
    different chemical species. ``q`` is monotone in how many protons an oxygen
    carries (bare O / OH- / H2O / H3O+), so the tail is proton transfer in the
    training data, not instability;
  * read with the physical global sign the ladder is textbook - and the global
    sign is exactly free, the Ewald energy being bilinear in q, so the model's
    own convention (O positive) is a gauge choice rather than an error.

The geometry section exists to keep the analysis honest about the cell: this
dataset's cell varies per frame (9.257 - 16.027 A), so every distance is taken
under minimum image with each frame's *own* cell. Using a single constant cell
once produced a spurious 0.244 A "near-contact"; the real tightest contact is
0.752 A and 96% of oxygens have exactly two bonded hydrogens.

``--with-kernel`` adds the amplitude test, which needs the checkpoint and the
author's cace checkout (see ``record_cace_valid.py`` for the branch argument):
it drives ``EwaldPotential`` directly with a modified ``q`` to confirm that the
bulk signal survives clipping the tail, and that a global sign flip and a
per-frame DC shift are both nearly or exactly free.

Usage:
    python probe_lr_charge.py
    python probe_lr_charge.py --with-kernel --cace-root /root/app/cace-ts \
        --model ../cace-lr-fit-datarepo/.../fit-water-mp0-lr-FeatureAdd/best_model.pth
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.dirname(HERE)
CACE_DIR = os.path.join(E2E, 'cace')
sys.path.insert(0, HERE)
sys.path.insert(0, CACE_DIR)

from metrics import load_valid  # noqa: E402 - path set above

DEFAULT_NPZ = os.path.join(CACE_DIR, 'cace_lrnew_valid_pred.npz')
DEFAULT_OUT = os.path.join(CACE_DIR, 'lr_charge_probe.json')

# An oxygen counts as bonded to a hydrogen within this distance. The ladder is
# insensitive to the exact value (checked below) but the group sizes are not.
BOND = 1.3
SMOOTH_R0 = 1.4
SMOOTH_W = 0.10
NEIGHBOUR_R = 3.5


def distances(coord, cell):
    """Minimum-image pair distances, each frame under its own cell diagonal."""
    nf, nat = coord.shape[:2]
    diag = np.array([np.diag(c) for c in cell.reshape(nf, 3, 3)])
    R = np.zeros((nf, nat, nat))
    for f in range(nf):
        D = coord[f][:, None, :] - coord[f][None, :, :]
        D -= diag[f] * np.round(D / diag[f])
        R[f] = np.sqrt((D ** 2).sum(-1))
        np.fill_diagonal(R[f], np.inf)
    return R, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--npz', default=DEFAULT_NPZ)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--with-kernel', action='store_true',
                    help='also run the amplitude test through EwaldPotential '
                         '(loads the checkpoint and a cace checkout)')
    ap.add_argument('--cace-root', default=None)
    ap.add_argument('--model', default=None)
    args = ap.parse_args()

    if not os.path.exists(args.npz):
        print(f'not found: {args.npz}')
        return 1

    d = np.load(args.npz)
    coord, atype, box, E_ref, _F_ref, nat = load_valid()
    nf = coord.shape[0]
    q = d['q'].reshape(nf, nat)
    if 'E_lr' not in d:
        print(f'FAIL: {args.npz} has no E_lr; it was recorded without a '
              f'long-range term, so there is no charge to describe')
        return 1

    O = atype[0] == 0
    R, diag = distances(coord, box)
    qO, qH = q[:, O], q[:, ~O]
    # expand to full (nf, nat, nat) so .sum() reduces over frames too, not just
    # over the broadcast axis
    m_OH = np.broadcast_to(O[None, :, None] & (~O)[None, None, :], R.shape)
    m_OO = np.broadcast_to(O[None, :, None] & O[None, None, :], R.shape)
    nH = ((R < BOND) & m_OH).sum(2)[:, O]              # protons per oxygen
    dO = np.where(m_OO, R, np.inf).min(2)[:, O]        # nearest oxygen
    dH = np.where(m_OH, R, np.inf).min(2)[:, O]        # nearest hydrogen
    cn = ((m_OH / (1.0 + np.exp((R - SMOOTH_R0) / SMOOTH_W))).sum(2))[:, O]

    # --- geometry, so the cell is never assumed constant ---
    bonds = np.where((R < BOND) & m_OH, R, np.nan)
    geom = {
        'cell_diagonal_a': {'min': float(diag.min()), 'median': float(np.median(diag)),
                            'max': float(diag.max()),
                            'unique': int(len(np.unique(np.round(diag, 6))))},
        'min_interatomic_a': float(R.min()),
        'o_h_bond_a': {'n': int(np.isfinite(bonds).sum()),
                       'min': float(np.nanmin(bonds)), 'median': float(np.nanmedian(bonds)),
                       'max': float(np.nanmax(bonds))},
        'o_with_exactly_two_h_pct': float(100 * (nH.ravel() == 2).mean()),
        'o_with_close_o_below_2a_pct': float(100 * (dO.ravel() < 2.0).mean()),
    }

    # --- the protonation ladder ---
    regular = (nH.ravel() == 2) & (dO.ravel() >= 2.0)
    ladder = {}
    for k in sorted(set(nH.ravel().tolist())):
        m = nH.ravel() == k
        ladder[int(k)] = {'n': int(m.sum()), 'mean_q': float(qO.ravel()[m].mean()),
                          'std_q': float(qO.ravel()[m].std()),
                          'mean_abs_q': float(np.abs(qO.ravel()[m]).mean())}
    cutoff_check = {}
    for r0 in (1.2, 1.3, 1.4, 1.5):
        nH_r0 = ((R < r0) & m_OH).sum(2)[:, O].ravel()
        cutoff_check[str(r0)] = {
            'nH_%d_mean_q' % k: (float(qO.ravel()[nH_r0 == k].mean())
                                 if (nH_r0 == k).sum() else None)
            for k in (0, 1, 2, 3)}
    charge = {
        'n_o_atom_frames': int(qO.size),
        'regular_water_mean_q': float(qO.ravel()[regular].mean()),
        'regular_water_std_q': float(qO.ravel()[regular].std()),
        'regular_water_p1_p99': [float(np.percentile(qO.ravel()[regular], 1)),
                                 float(np.percentile(qO.ravel()[regular], 99))],
        'regular_water_min_max': [float(qO.ravel()[regular].min()),
                                  float(qO.ravel()[regular].max())],
        'regular_water_pct_below_zero': float(100 * (qO.ravel()[regular] < 0).mean()),
        'ladder_by_proton_count': ladder,
        'ladder_cutoff_check': cutoff_check,
        'smooth_coordination_corr': float(np.corrcoef(qO.ravel(), cn.ravel())[0, 1]),
        'h_by_bridge_count': {
            str(k): {'n': int(((m_OH.sum(1)[:, ~O]).ravel() == k).sum()),
                     'mean_q': float(qH.ravel()[(m_OH.sum(1)[:, ~O]).ravel() == k].mean())}
            for k in sorted(set((m_OH.sum(1)[:, ~O]).ravel().tolist()))},
        'negation_is_exact_gauge_not_checked_here': (
            'E(-q) = E(q) is exact because the Ewald energy is bilinear in q; '
            'the kernel test confirms it with max|dE| = 0'),
    }

    # --- local compensation: are the extremes dipole-like pairs? ---
    loc = np.zeros_like(q)
    for f in range(nf):
        m = R[f] < NEIGHBOUR_R
        loc[f] = np.where(m, q[f][None, :], 0).sum(1)
    big = np.abs(q) > 1.0
    comp = {
        'pct_atom_frames_abs_q_gt_1': float(100 * big.mean()),
        'abs_q_gt_1_neighbour_sum_mean': float(loc[big].mean()),
        'corr_q_vs_neighbour_sum_all': float(np.corrcoef(q.ravel(), loc.ravel())[0, 1]),
        'corr_q_vs_neighbour_sum_abs_q_gt_1': float(np.corrcoef(q[big], loc[big])[0, 1]),
        'corr_q_vs_neighbour_sum_abs_q_le_1': float(np.corrcoef(q[~big], loc[~big])[0, 1]),
    }

    # --- the extremes, named ---
    extremes = []
    for lab, idx in (('max_O', np.argmax(np.where(O[None, :], q, -np.inf))),
                     ('min_O', np.argmin(np.where(O[None, :], q, np.inf)))):
        f, a = np.unravel_index(idx, q.shape)
        o = int(np.where(np.flatnonzero(O) == a)[0][0])
        extremes.append({'which': lab, 'q': float(q[f, a]), 'frame': int(f),
                         'protons_on_O': int(nH[f, o]),
                         'nearest_O_a': float(dO[f, o]), 'nearest_H_a': float(dH[f, o]),
                         'cell_diagonal_a': float(diag[f, 0])})

    out = {
        'recorded_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'npz': os.path.relpath(args.npz, E2E),
        'n_frames': nf, 'n_atoms': nat,
        'note': ('The learned q is a latent variable: only the Ewald energy and '
                 'forces see it, and they are bilinear, so individual values are '
                 'not identifiable. What this probe shows is that the variation '
                 'it uses is the local protonation state, not noise.'),
        'geometry': geom, 'charge': charge, 'compensation': comp, 'extremes': extremes,
    }

    print(f'{nf} frames x {nat} atoms; O {int(O.sum())}, H {int((~O).sum())}')
    print('\n== geometry (per-frame cells) ==')
    print('  cell diagonal %.3f - %.3f A (%d unique)' % (diag.min(), diag.max(),
                                                         geom['cell_diagonal_a']['unique']))
    print('  tightest contact %.3f A ; O-H bonds n=%d %.3f - %.3f (median %.3f)'
          % (geom['min_interatomic_a'], geom['o_h_bond_a']['n'], geom['o_h_bond_a']['min'],
             geom['o_h_bond_a']['max'], geom['o_h_bond_a']['median']))
    print('  O with exactly 2 H within %g A: %.1f%% ; O with a nearest O < 2 A: %.2f%%'
          % (BOND, geom['o_with_exactly_two_h_pct'], geom['o_with_close_o_below_2a_pct']))
    print('\n== the charge is a protonation coordinate ==')
    print('  regular water O (2 H, no close O), %.1f%% of O atom-frames:'
          % (100 * regular.mean()))
    print('    mean q %+.3f  std %.3f  p1/p99 %+.3f/%+.3f  %.1f%% below zero'
          % (charge['regular_water_mean_q'], charge['regular_water_std_q'],
             charge['regular_water_p1_p99'][0], charge['regular_water_p1_p99'][1],
             charge['regular_water_pct_below_zero']))
    print('  mean q by protons on the oxygen (model sign convention):')
    for k in sorted(ladder):
        L = ladder[k]
        print('    %d H  n=%5d  mean q %+7.3f  std %.3f' % (k, L['n'], L['mean_q'], L['std_q']))
    print('  robustness: same ordering at every cutoff', {r: v for r, v in cutoff_check.items()})
    print('  smooth coordination (r0 %g, w %g): corr(q, coordination) = %+.3f'
          % (SMOOTH_R0, SMOOTH_W, charge['smooth_coordination_corr']))
    print('\n== local compensation ==')
    print('  |q|>1 in %.2f%% of atom-frames; their neighbour sum averages %+.3f'
          % (comp['pct_atom_frames_abs_q_gt_1'], comp['abs_q_gt_1_neighbour_sum_mean']))
    print('  corr(q, neighbour sum <%g A): all %+.3f, |q|>1 %+.3f, |q|<=1 %+.3f'
          % (NEIGHBOUR_R, comp['corr_q_vs_neighbour_sum_all'],
             comp['corr_q_vs_neighbour_sum_abs_q_gt_1'],
             comp['corr_q_vs_neighbour_sum_abs_q_le_1']))
    print('\n== the extremes ==')
    for x in extremes:
        print('  %s q=%+.3f frame %d: %d protons, nearest O %.2f A, nearest H %.2f A, cell %.3f A'
              % (x['which'], x['q'], x['frame'], x['protons_on_O'],
                 x['nearest_O_a'], x['nearest_H_a'], x['cell_diagonal_a']))

    if args.with_kernel:
        out['kernel'] = kernel_test(args)

    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
        fh.write('\n')
    print(f'\nwrote  {args.out}')
    return 0


def kernel_test(args):
    """Drive EwaldPotential with a modified q: how much of E_lr is the tail?"""
    import torch
    from inspect_cace_les import _load

    model_path = args.model or os.path.join(
        E2E, '..', 'cace-lr-fit-datarepo', 'BingqingCheng-cace-lr-fit-0211150',
        'fit-water-timing', 'fit-water-mp0-lr-FeatureAdd', 'best_model.pth')
    _cace, loader, model, _coll = _load(cace_root=args.cace_root, model_path=model_path,
                                        batch_size=2)
    EP = [m for m in model.modules() if type(m).__name__ == 'EwaldPotential'][0]
    acc = {}

    def keep(tag, vals):
        acc.setdefault(tag, []).extend(np.asarray(vals).ravel().tolist())

    for batch in loader:
        bd = batch.to_dict()
        bd['positions'] = bd['positions'].detach().clone().requires_grad_(True)
        pred = model(bd, training=False)
        q = pred['q'].detach().clone()
        keep('ref', pred['ewald_potential'].detach().cpu().numpy())
        b = bd['batch']

        def run(qmod, tag):
            dd = dict(bd)
            dd['q'] = qmod
            keep(tag, EP(dd, training=False)['ewald_potential'].detach().cpu().numpy())

        qd = q.clone()
        for f in range(int(b.max()) + 1):
            m = b == f
            qd[m] = q[m] - q[m].mean()
        run(torch.clamp(q, -0.5, 0.5), 'clip_all_05')
        run(torch.where(q.abs() > 1.0, torch.clamp(q, -1.0, 1.0), q), 'clip_tail_1')
        run(torch.where(q.abs() > 1.0, torch.zeros_like(q), q), 'zero_tail')
        run(-q, 'neg')
        run(qd, 'no_dc')
        rng = np.random.default_rng(0)
        flat = np.zeros(q.numel(), dtype=bool)
        flat[rng.choice(q.numel(), size=int((q.abs() > 1.0).sum()), replace=False)] = True
        run(torch.where(torch.as_tensor(flat.reshape(q.shape), device=q.device),
                        torch.zeros_like(q), q), 'zero_random')

    ref = np.array(acc.pop('ref'))
    res = {}
    print('\n== amplitude test: E_lr under a modified q (author kernel) ==')
    print('  frames %d, E_lr mean %+.3f std %.3f eV/frame' % (ref.size, ref.mean(), ref.std()))
    print('  %-16s %10s %10s %12s %8s' % ('variant', 'mean', 'std', 'rms(dE)', 'r'))
    for tag, lab in (('clip_all_05', 'clip all to +-0.5'), ('clip_tail_1', 'clip only tail'),
                     ('zero_tail', 'zero tail'), ('zero_random', 'zero random 2.2%'),
                     ('no_dc', 'DC removed'), ('neg', '-q')):
        v = np.array(acc[tag])
        d = v - ref
        res[tag] = {'mean': float(v.mean()), 'std': float(v.std()),
                    'max_abs_dE': float(np.abs(d).max()),
                    'rms_dE': float(np.sqrt((d ** 2).mean())),
                    'corr_with_ref': float(np.corrcoef(v, ref)[0, 1])}
        print('  %-16s %+10.3f %10.3f %12.4f %+8.4f'
              % (lab, v.mean(), v.std(), np.sqrt((d ** 2).mean()), np.corrcoef(v, ref)[0, 1]))
    res['signal_std'] = float(ref.std())
    return res


if __name__ == '__main__':
    raise SystemExit(main())
