#!/usr/bin/env python
# coding: utf-8
"""Aggregate the per-checkpoint `score_cace.py` JSONs into one TSV for the notebook.

Columns mirror `analysis/data/valid_sweep.tsv` (arm, step, rmse_e_peratom,
rmse_f, natoms, nframes) so the cace arms can join the deepmd rows in one figure,
plus the energy decomposition and the long-range mechanism fields.

`step` is the deepmd training step and the cace training EPOCH - different units
of work, which is why `epoch` is written too: the campaign fixed both families at
160 steps/epoch (batch_size 2 over 320 frames), so 80000 steps and 500 epochs are
the same 500 passes over the data.

Checkpoint provenance, from `fit_cace.py`'s PHASE_CKPT: the phases end at epochs
200/300/400/500, and each phase's end is written as model.pth, model-2.pth,
model-3.pth, model-4.pth. `best_model.pth` is the lowest-validation-loss
checkpoint *within the final phase*, i.e. it was selected by a criterion measured
on exactly the 80 frames we score on - so it is reported but flagged, and kept out
of the headline mean.

The sea arms (the se_a descriptor on cace's loop) write their rows, their epoch
traces and their per-epoch decomposition into their OWN files by default -
`cace_sea_valid.tsv`, `cace_sea_epoch_metrics.tsv`, `cace_sea_terms.tsv` - rather
than into the tables above. The reason is in `is_sea_dir`: the campaign's figures
index arm labels by exact key, so appending would break or repaint them.

Usage:
    python collect_sweep.py
        # analysis/data/cace_valid.tsv          cace-{sr,lr} scored checkpoints
        # analysis/data/cace_epoch_metrics.tsv  their training traces
        # analysis/data/cace_sea_valid.tsv      the sea arms' scored checkpoints
        # analysis/data/cace_sea_epoch_metrics.tsv
        # analysis/data/cace_sea_terms.tsv      per-epoch SR/LR/q decomposition
        # analysis/data/cace_element_charge.tsv per-element charge at each checkpoint
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ANALYSIS_DATA = os.path.join(CAMPAIGN, 'analysis', 'data')
DEFAULT_OUT = os.path.join(ANALYSIS_DATA, 'cace_valid.tsv')
DEFAULT_OUT_SEA = os.path.join(ANALYSIS_DATA, 'cace_sea_valid.tsv')
DEFAULT_EPOCHS_OUT = os.path.join(ANALYSIS_DATA, 'cace_epoch_metrics.tsv')
DEFAULT_EPOCHS_SEA_OUT = os.path.join(ANALYSIS_DATA, 'cace_sea_epoch_metrics.tsv')
DEFAULT_TERMS_OUT = os.path.join(ANALYSIS_DATA, 'cace_sea_terms.tsv')
DEFAULT_ELEM_OUT = os.path.join(ANALYSIS_DATA, 'cace_element_charge.tsv')
DEFAULT_VALID_XYZ = os.path.join(CAMPAIGN, 'xyz', 'valid.xyz')

# stem -> (epoch at which the checkpoint was written, val-selected?)
STEMS = {
    'model': (200, False),
    'model-2': (300, False),
    'model-3': (400, False),
    'model-4': (500, False),
    'best_model': (None, True),
}
# `head_n_in` and `seed` are only filled by the sea arms; a cace checkpoint has no
# descriptor dimension of its own to report (and its score json records no seed,
# because those runs were unseeded).
COLS = ['arm', 'step', 'epoch', 'ckpt', 'val_selected', 'rmse_e_peratom', 'rmse_f',
        'natoms', 'nframes', 'e_r2', 'f_r2', 'bias', 'spread', 'bias2_over_rmse2',
        'f_lr_over_f_tot', 'f_self_over_f_tot', 'e_lr_mean_ev_per_frame',
        'e_lr_std_ev_per_frame', 'e_self_ev_per_frame',
        'net_charge_mean_per_frame', 'sum_q2_mean_per_frame',
        'self_term_ev_per_frame', 'self_over_e_lr', 'remove_self_interaction',
        'model_sha256_12', 'head_n_in', 'seed']


def row_from_json(path):
    with open(path) as fh:
        rec = json.load(fh)
    stem = os.path.splitext(os.path.basename(path))[0]
    epoch, vsel = STEMS[stem]
    arm = f"cace-{rec['arm']}"
    m = rec['metrics'][rec['tag']]
    mech = rec.get('mechanism') or {}
    pars = (mech.get('ewald_params') or [{}])[0]
    sea = rec.get('sea_rebuild') or {}  # only the sea arms write this
    return {
        'arm': arm, 'step': epoch, 'epoch': epoch, 'ckpt': stem,
        'val_selected': int(bool(vsel)),
        'rmse_e_peratom': m['e_rmse'], 'rmse_f': m['f_rmse'],
        'natoms': rec['n_atoms'], 'nframes': rec['n_frames'],
        'e_r2': m['e_r2'], 'f_r2': m['f_r2'],
        'bias': m['bias'], 'spread': m['spread'],
        'bias2_over_rmse2': m['bias2_over_rmse2'],
        'f_lr_over_f_tot': mech.get('f_lr_over_f_tot'),
        'f_self_over_f_tot': mech.get('f_self_over_f_tot'),
        'e_lr_mean_ev_per_frame': mech.get('e_lr_mean_ev_per_frame'),
        'e_lr_std_ev_per_frame': mech.get('e_lr_std_ev_per_frame'),
        'e_self_ev_per_frame': mech.get('e_self_ev_per_frame'),
        'net_charge_mean_per_frame': mech.get('net_charge_mean_per_frame'),
        'sum_q2_mean_per_frame': mech.get('sum_q2_mean_per_frame'),
        'self_term_ev_per_frame': mech.get('self_term_ev_per_frame'),
        'self_over_e_lr': mech.get('self_over_e_lr'),
        'remove_self_interaction': pars.get('remove_self_interaction'),
        'model_sha256_12': rec['model_sha256'][:12],
        'head_n_in': sea.get('dim_out'), 'seed': sea.get('seed'),
    }


def is_sea_dir(*paths):
    """True for the holder/arm of a `cace-sea-*` run (the se_a descriptor arms).

    Used to split the two families of output files. They are kept apart on
    purpose: `cace_valid.tsv` and `cace_epoch_metrics.tsv` feed a finished report
    whose figures index each arm's label by exact key, so appending rows for arms
    it has never heard of would either crash those figures (`CACE_LABELS[arm]`) or
    silently repaint them. The sea arms are a descriptor ablation on cace's loop,
    not two more rows of that campaign, so they get their own tables.
    """
    return any('cace-sea-' in p for p in paths)


def write_rows(rows, path, what):
    """Write scored rows as a TSV, or say so when there are none yet."""
    if not rows:
        print(f'\nno {what} scored checkpoints; skipped {path}')
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as fh:
        fh.write('\t'.join(COLS) + '\n')
        for r in rows:
            fh.write('\t'.join(
                '' if r[c] is None else f'{r[c]:.8e}' if isinstance(r[c], float)
                else str(r[c]) for c in COLS) + '\n')
    print(f'\nwrote {path}  ({len(rows)} rows)')


def headline(rows):
    """The per-arm headline: the evenly spaced phase checkpoints, no
    validation-selected one in the mean (best_model.pth was picked on exactly the
    80 frames every arm is scored on, so it is reported but never averaged)."""
    if not rows:
        return
    print(f'{"arm":<14} {"mean rmse_e/at":>14} {"mean rmse_f":>12}   '
          f'{"bias":>11} {"spread":>11}  (phase checkpoints only)')
    for arm in sorted({r['arm'] for r in rows}):
        sel = [r for r in rows if r['arm'] == arm and not r['val_selected']]
        if not sel:
            print(f'{arm:<14} {"":>14} {"":>12}   {"":>11} {"":>11}   n=0')
            continue
        print(f'{arm:<14} {np.mean([r["rmse_e_peratom"] for r in sel]):>14.6e} '
              f'{np.mean([r["rmse_f"] for r in sel]):>12.6e}   '
              f'{np.mean([r["bias"] for r in sel]):>+11.4e} '
              f'{np.mean([r["spread"] for r in sel]):>11.4e}   n={len(sel)}')


def collect_epochs(runs_root, out_path, sea=False):
    """Concatenate the per-arm training traces into one file for the notebook.

    `score_cace.py --epochs-tsv` writes one trace per arm inside the arm's run
    directory; the notebook reads everything out of `analysis/data/`, so the two
    traces are joined here with an `arm` column rather than copied per arm.
    """
    frames = []
    for p in sorted(glob.glob(os.path.join(runs_root, 'cace-*', 'epoch_metrics.tsv'))):
        if is_sea_dir(p) != sea:
            continue
        d = pd.read_csv(p, sep='\t')
        d.insert(0, 'arm', f"cace-{os.path.basename(os.path.dirname(p))[5:]}")
        frames.append(d)
    if not frames:
        print(f'\nno epoch traces under {runs_root}/cace-*/epoch_metrics.tsv'
              f'{" (sea)" if sea else ""}; skipped {out_path}')
        return None
    ep = pd.concat(frames, ignore_index=True)
    ep.to_csv(out_path, sep='\t', index=False)
    print(f'wrote {out_path}  ({len(ep)} epoch rows, '
          f'{ep["arm"].nunique()} arms, epochs {int(ep["global_epoch"].min())}-'
          f'{int(ep["global_epoch"].max())})')
    return ep


def collect_terms(runs_root, out_path):
    """Concatenate the sea arms' per-epoch decomposition (`sea_terms.tsv`).

    `SeaTermLogger` writes one file per sea arm inside its run directory; the
    notebook reads everything out of `analysis/data/`, so they are joined here
    with an `arm` column, the same way the epoch traces are.

    The two arms' columns are NOT the same width: the sr arm has no long-range
    channel, so its `val_e_lr_*`, `val_q_*`, `val_net_charge`, `val_sum_q2`,
    `val_f_lr_*` fields are empty and concatenate as NaN. That is the honest
    encoding - `arm` says which channel exists - and it keeps one file readable
    by one code path.
    """
    frames = []
    for p in sorted(glob.glob(os.path.join(runs_root, 'cace-sea-*', 'sea_terms.tsv'))):
        d = pd.read_csv(p, sep='\t')
        d.insert(0, 'arm', f"cace-{os.path.basename(os.path.dirname(p))[5:]}")
        frames.append(d)
    if not frames:
        print(f'\nno sea term logs under {runs_root}/cace-sea-*/sea_terms.tsv; '
              f'skipped {out_path}')
        return None
    t = pd.concat(frames, ignore_index=True)
    t.to_csv(out_path, sep='\t', index=False)
    print(f'wrote {out_path}  ({len(t)} term rows, {t["arm"].nunique()} arms, '
          f'epochs {int(t["global_epoch"].min())}-{int(t["global_epoch"].max())})')
    return t


def element_order(valid_xyz):
    """The species symbol of every atom, in the order the scorer sees them.

    The per-atom charge split needs the validation set's atom order. Every frame
    in `xyz/valid.xyz` carries the same species order and the same atom count
    (both asserted here), so one frame's order indexes all of them.
    """
    with open(valid_xyz) as fh:
        lines = fh.read().splitlines()
    natoms = int(lines[0])
    stride = natoms + 2
    if len(lines) % stride:
        raise ValueError(f'{valid_xyz}: {len(lines)} lines is not a whole number '
                         f'of {stride}-line frames')
    cols = [lines[2 + i].split() for i in range(natoms)]
    symbols = [c[0] for c in cols]
    expected = {'H': 1, 'O': 8}
    bad = [(s, c[7]) for s, c in zip(symbols, cols) if expected.get(s) != int(c[7])]
    if bad:
        raise ValueError(f'{valid_xyz}: species and Z disagree, e.g. {bad[:3]}')
    nframes = len(lines) // stride
    for f in range(1, nframes):
        got = [lines[f * stride + 2 + i].split()[0] for i in range(natoms)]
        if got != symbols:
            raise ValueError(f'{valid_xyz}: frame {f} has a different species order')
    return symbols, natoms, nframes


def collect_element_charge(runs_root, valid_xyz, out_path):
    """Per-element charge statistics at every scored checkpoint.

    `score_cace.py` already stores the per-atom charge of all validation frames in
    each checkpoint's npz (`q`, natoms*nframes long), so the element split is
    recoverable from what is on disk - no re-scoring, and no change to
    `SeaTermLogger`, which logs only all-atom charge aggregates.

    The mean is stored as MEASURED, on whichever branch of the `q -> -q` gauge the
    arm's optimizer happened to land. That sign is not data - E_lr = q^T K q and
    F_LR = q^T (dK/dr) q are both even in q, and the short-range half never sees q
    - so the figure that compares two arms flips it to a common convention and
    says so; the table keeps the raw number.
    """
    symbols, natoms, nframes = element_order(valid_xyz)
    els = sorted(set(symbols))
    idx = {e: np.array([s == e for s in symbols]) for e in els}
    rows = []
    for arm_dir in sorted(glob.glob(os.path.join(runs_root, 'cace-*'))):
        arm = f'cace-{os.path.basename(arm_dir)[5:]}'
        for stem, (epoch, vsel) in STEMS.items():
            npz = os.path.join(arm_dir, 'npz', f'{stem}.npz')
            if not os.path.exists(npz):
                continue
            with np.load(npz) as d:
                if 'q' not in d:
                    continue
                q = d['q'].reshape(nframes, natoms)
            net = q.sum(1).mean()
            const = sum(int(idx[e].sum()) * q[:, idx[e]].mean() for e in els)
            if abs(const - net) > 1e-4 * max(abs(net), 1.0):
                raise ValueError(f'{npz}: per-element means give {const:.6f} but '
                                 f'the frame net charge is {net:.6f}; the atom order '
                                 f'of `q` does not match {valid_xyz}')
            for e in els:
                v = q[:, idx[e]]
                per_frame = v.mean(1)
                rows.append({
                    'arm': arm, 'epoch': epoch, 'ckpt': stem,
                    'val_selected': int(bool(vsel)), 'elem': e,
                    'natoms_elem': int(idx[e].sum()), 'nframes': nframes,
                    'q_mean': float(v.mean()),
                    'q_sd': float(v.std()),
                    'q_sd_of_frame_mean': float(per_frame.std()),
                    'q_min': float(v.min()), 'q_max': float(v.max()),
                    'net_charge_mean_per_frame': float(net),
                })
    cols = ['arm', 'epoch', 'ckpt', 'val_selected', 'elem', 'natoms_elem',
            'nframes', 'q_mean', 'q_sd', 'q_sd_of_frame_mean', 'q_min', 'q_max',
            'net_charge_mean_per_frame']
    if not rows:
        print(f'\nno charge-bearing npz under {runs_root}/cace-*/npz; '
              f'skipped {out_path}')
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            fh.write('\t'.join(
                '' if r[c] is None else f'{r[c]:.8e}' if isinstance(r[c], float)
                else str(r[c]) for c in cols) + '\n')
    arms = sorted({r['arm'] for r in rows})
    print(f'\nwrote {out_path}  ({len(rows)} rows, {len(arms)} arms: {arms}, '
          f'elements {els}, {nframes} frames x {natoms} atoms)')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runs-root', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--out', default=DEFAULT_OUT,
                    help='cace arms only; the sea arms go to --out-sea')
    ap.add_argument('--out-sea', default=DEFAULT_OUT_SEA)
    ap.add_argument('--epochs-out', default=DEFAULT_EPOCHS_OUT)
    ap.add_argument('--epochs-sea-out', default=DEFAULT_EPOCHS_SEA_OUT)
    ap.add_argument('--terms-out', default=DEFAULT_TERMS_OUT)
    ap.add_argument('--elem-out', default=DEFAULT_ELEM_OUT)
    ap.add_argument('--valid-xyz', default=DEFAULT_VALID_XYZ)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.runs_root, 'cace-*', '*.json')))
    rows = []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in STEMS:
            continue
        rows.append(row_from_json(p))
    if not rows:
        raise SystemExit(f'no scored checkpoints under {args.runs_root}')

    cace_rows = [r for r in rows if not r['arm'].startswith('cace-sea-')]
    sea_rows = [r for r in rows if r['arm'].startswith('cace-sea-')]
    for group in (cace_rows, sea_rows):
        group.sort(key=lambda r: (r['arm'], r['epoch'] if r['epoch'] else 10 ** 9))

    print('=== the campaign\'s cace arms')
    headline(cace_rows)
    print('\n=== the se_a-descriptor arms (separate tables, see is_sea_dir)')
    headline(sea_rows)

    write_rows(cace_rows, args.out, 'cace')
    write_rows(sea_rows, args.out_sea, 'sea')
    collect_epochs(args.runs_root, args.epochs_out, sea=False)
    collect_epochs(args.runs_root, args.epochs_sea_out, sea=True)
    collect_terms(args.runs_root, args.terms_out)
    collect_element_charge(args.runs_root, args.valid_xyz, args.elem_out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
