#!/usr/bin/env python
# coding: utf-8
"""Forward-pass a cace checkpoint over the shared valid split and record what it
scores.

``metrics.py`` scores the default checkpoint too (the ``cace`` / ``cace_sr``
rows), but it only prints. This writes the numbers to disk so the learning-curve
notebook can read them without re-running the model, and so the record carries
the identity of what produced it: the sha256 of the checkpoint, which cace
checkout ran it, the frame count, the target scale, and the split-agreement check.

Scored against the same held-out split as every deepmd arm -
``cace_compare_e2e/water/valid/set.000`` - whose identity is re-derived from the
cace loader on each run and compared against before anything is written. If the
two ever disagree the script writes nothing and exits non-zero: a metric against
a mismatched target is worse than no metric.

Two rows come out of one forward pass:
  <tag>      the combined prediction, ``CACE_energy`` / ``CACE_forces``
  <tag>_sr   the same model with the long-range term dropped,
             ``E_tot - w * ewald_potential`` for the checkpoint's mix weight w

``--no-long-range`` is for a checkpoint that has no Ewald head at all (the
author's ``fit-water-timing/fit-water-mp0-sr`` short-range run): it writes one row
and no mechanism block. Note the difference from the ``<tag>_sr`` row above,
which is the same *long-range* model with its Ewald term subtracted and is
therefore not a short-range model - see ``probe_lr_control.py`` for why that
distinction changes the answer by 8x.

The same pass also records the mechanism columns (E_lr and E_sr as a fraction of
the target's spread, their correlation, F_lr over F_tot, and the charge
statistics), so the report's magnitude table is reproducible from the JSON rather
than from an ad-hoc script. The raw E_lr level is code-specific - the author's
kernel runs at ``norm_factor = 1``, historically a ~90x scale - so only the ratio
columns are comparable across codes.

Which cace is imported matters, so it is an argument. A checkpoint pickles its
module classes by name; the author's updated FeatureAdd checkpoint was trained
with the ``torchscript`` branch and raises ``TypeError`` inside main's
``angular.py`` (its stored lxzlyz keys are TorchScript-safe strings). Point
``--cace-root`` at that branch's checkout to score it.

Usage:
    python record_cace_valid.py                    # default: main cace, w=0.01
    python record_cace_valid.py --cace-root /root/app/cace-ts \
        --model .../fit-water-mp0-lr-FeatureAdd/best_model.pth --lr-weight 1.0 \
        --arm-tag cace_lrnew --out cace/cace_lrnew_valid_metrics.json \
        --npz cace/cace_lrnew_valid_pred.npz
    python record_cace_valid.py --cace-root /root/app/cace-ts --no-long-range \
        --model .../fit-water-mp0-sr/best_model.pth --arm-tag cace_timing_sr \
        --out cace/cace_timing_sr_valid_metrics.json \
        --npz cace/cace_timing_sr_valid_pred.npz
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.dirname(HERE)
CACE_DIR = os.path.join(E2E, 'cace')
sys.path.insert(0, HERE)
sys.path.insert(0, CACE_DIR)

from metrics import load_valid, metrics  # noqa: E402 - path set above

DEFAULT_MODEL = os.path.join(CACE_DIR, 'best_model.pth')
DEFAULT_OUT = os.path.join(CACE_DIR, 'cace_valid_metrics.json')
DEFAULT_NPZ = os.path.join(CACE_DIR, 'cace_valid_pred.npz')
# CombinePotential mixes the Ewald term in with this weight; the SR-only arm is
# the combined prediction minus that contribution. FeatureAdd sums at unit
# weight, so an updated-checkpoint run passes 1.0.
LR_MIX_WEIGHT = 0.01


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cace_provenance(cace_root):
    """Which cace checkout a run imported: path, version, and git commit."""
    try:
        import cace
        path = os.path.abspath(cace.__file__)
        version = getattr(cace, '__version__', None)
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        return {'import_error': f'{type(exc).__name__}: {exc}'}
    prov = {'imported_from': path, 'version': version}
    root = cace_root or os.path.dirname(os.path.dirname(path))
    try:
        prov['checkout'] = os.path.abspath(root)
        prov['commit'] = subprocess.run(
            ['git', '-C', root, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True).stdout.strip()
        prov['dirty'] = subprocess.run(
            ['git', '-C', root, 'status', '--porcelain'],
            capture_output=True, text=True, check=True).stdout.strip() != ''
    except Exception:  # noqa: BLE001 - not a git checkout, or git missing
        pass
    return prov


def check_split(cace_root):
    """Refuse to score unless the cace loader's valid split is water/valid.

    The comparison is on positions and cell, which is stronger than comparing
    energies: it pins the frames *and* their order, and it works on cace branches
    whose loader does not carry the energy through batching.
    """
    from inspect_cace_les import split_identity

    pos, cell, z, _E = split_identity(cace_root=cace_root)
    coord, atype, box, E_ref, _F_ref, nat = load_valid()

    if pos.shape != coord.shape or cell.shape != box.reshape(-1, 3, 3).shape:
        print(f'FAIL: cace valid is {pos.shape[0]} frames x {pos.shape[1]} atoms, '
              f'ours is {coord.shape[0]} x {coord.shape[1]}')
        return None, None
    dpos = float(np.abs(pos - coord).max())
    dcell = float(np.abs(cell - box.reshape(-1, 3, 3)).max())
    if dpos > 1e-9 or dcell > 1e-9:
        print(f'FAIL: cace valid geometry differs from water/valid/set.000 '
              f'(max|dpos|={dpos:.3e} A, max|dcell|={dcell:.3e} A)')
        return None, None

    # element numbers as a function of our type index: 0/1 -> H/O in this dataset
    mapping = {int(i): int(z[0][atype[0] == i][0])
               for i in np.unique(atype[0])}
    if not np.array_equal(z, np.vectorize(mapping.get)(atype)):
        print(f'FAIL: cace atomic numbers {sorted(mapping.values())} do not map '
              f'onto our type index {sorted(mapping)}')
        return None, None
    return (dpos, dcell), dict(mapping)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--npz', default=DEFAULT_NPZ,
                    help='per-frame predictions for plotting; "" to skip')
    ap.add_argument('--cace-root', default=None,
                    help='cace checkout to import (default: the installed one)')
    ap.add_argument('--lr-weight', type=float, default=LR_MIX_WEIGHT,
                    help='weight the long-range term enters CACE_energy with')
    ap.add_argument('--arm-tag', default='cace',
                    help='base name of the two rows written (default: cace)')
    ap.add_argument('--no-long-range', action='store_true',
                    help='the checkpoint has no Ewald head (the timing '
                         'fit-water-mp0-sr control): write one row and no '
                         'mechanism block')
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f'model not found: {args.model}')
        return 1

    # Our split, and the target the deepmd arms are scored against.
    coord, _atype, _box, E_ref, F_ref, nat = load_valid()

    # Import from the requested root BEFORE anything imports cace, so the branch
    # choice takes effect (a cached module would keep whichever came first).
    from inspect_cace_les import predict, predict_sr  # noqa: E402 - needs cace on the path

    model = args.model
    tag = sha256(model)[:12]
    print(f'model   {model}\n        sha256[:12] = {tag}')

    # check_split is what first imports cace, from the requested root; provenance
    # reads it back afterwards, so that the report names the branch actually used.
    geom, mapping = check_split(args.cace_root)
    if geom is None:
        return 1

    prov = cace_provenance(args.cace_root)
    print(f'cace    {prov.get("imported_from")}')
    if prov.get('commit'):
        print(f'        commit {prov["commit"][:12]}'
              f'{" (dirty)" if prov.get("dirty") else ""}')

    dpos, dcell = geom
    print(f'split   {len(E_ref)} frames x {nat} atoms, geometry matches '
          f'water/valid (max|dpos|={dpos:.1e} A, max|dcell|={dcell:.1e} A), '
          f'elements {mapping}')

    if args.no_long_range:
        E_tot, E_cace_ref, F_tot = predict_sr(
            cace_root=args.cace_root, model_path=args.model)
        q = E_lr = F_lr = None
        E_sr, F_sr = E_tot, F_tot  # nothing to subtract: no long-range term
    else:
        q, E_lr, E_tot, E_sr, E_cace_ref, F_lr, F_sr, F_tot = predict(
            lr_weight=args.lr_weight, cace_root=args.cace_root,
            model_path=args.model)

    # Secondary check, when the loader carries its own copy of the target.
    if np.isfinite(E_cace_ref).all():
        if len(E_cace_ref) != len(E_ref):
            print(f'FAIL: cace valid has {len(E_cace_ref)} frames, '
                  f'ours has {len(E_ref)}')
            return 1
        if not np.allclose(E_cace_ref, E_ref, atol=1e-5):
            print('FAIL: cace loader targets differ from water/valid/set.000, '
                  'in order')
            print(f'      max|diff| = {np.abs(E_cace_ref - E_ref).max():.3e} eV')
            return 1

    nf = len(E_ref)
    # The loader reports forces as (n_frames * nat, 3); accept that or the
    # already-framed layout, but nothing else.
    f_pred_shape = np.asarray(F_tot).shape
    if f_pred_shape != (nf * nat, 3) and f_pred_shape != (nf, nat, 3):
        print(f'FAIL: force shape {f_pred_shape} is neither '
              f'{(nf * nat, 3)} nor {(nf, nat, 3)}')
        return 1
    reshape = lambda a: np.asarray(a).reshape(nf, nat, 3)

    # denominators, so every error is also readable as a fraction of the target
    e_target_peratom = float(np.sqrt(np.mean((E_ref - E_ref.mean()) ** 2)) / nat)
    f_target_rms = float(np.sqrt(np.mean(F_ref ** 2)))

    # The Task 1 mechanism columns, from this same forward pass: how the learned
    # long-range term behaves relative to the target's own spread. The raw E_lr
    # level is NOT comparable across codes (the author's kernel runs at
    # ``norm_factor = 1``, historically a ~90x scale), so the ratio columns are
    # the ones to read, and they are what the deepmd arms' table compares.
    # A short-range-only checkpoint has no long-range term to describe.
    F_tot = np.asarray(F_tot)
    mech = None
    if not args.no_long_range:
        E_lr = np.asarray(E_lr)
        E_sr = np.asarray(E_sr)
        q = np.asarray(q)
        e_tgt_std = e_target_peratom * nat  # per-frame target spread, eV
        f_lr_rms = float(np.sqrt(np.mean((args.lr_weight * np.asarray(F_lr)) ** 2)))
        qa = q.reshape(nf, nat, -1)
        atype0 = _atype[0]
        mech = {
            'e_lr_std_over_target_std': float(args.lr_weight * E_lr.std() / e_tgt_std),
            'e_sr_std_over_target_std': float(E_sr.std() / e_tgt_std),
            'corr_e_sr_e_lr': float(np.corrcoef(E_sr, E_lr)[0, 1]),
            'f_lr_rms_over_f_tot_rms': f_lr_rms / float(np.sqrt(np.mean(F_tot ** 2))),
            'q_out_channels': int(q.shape[-1]),
            'net_charge_mean_per_frame': float(qa.sum(axis=1).mean()),
            'q_mean_by_element': {
                str(int(mapping[i])): float(qa[:, atype0 == i, :].mean())
                for i in sorted(mapping)},
        }

    # A long-range checkpoint yields two rows (combined and LR-removed); a
    # short-range-only one yields a single row.
    pairs = [(args.arm_tag, E_tot, reshape(F_tot))]
    if not args.no_long_range:
        pairs.append((args.arm_tag + '_sr', E_sr, reshape(F_sr)))
    rows = {}
    for arm, E_pred, F_pred in pairs:
        m = metrics(E_pred, F_pred, E_ref, F_ref, nat)
        m['e_rmse_pct_of_target'] = 100.0 * m['e_rmse'] / e_target_peratom
        m['f_rmse_pct_of_target'] = 100.0 * m['f_rmse'] / f_target_rms
        rows[arm] = m

    if args.no_long_range:
        lr_stats = None
        arm_defs = {args.arm_tag: 'CACE_energy / CACE_forces (short range only; '
                                  'this checkpoint has no long-range head)'}
    else:
        lr_stats = {
            'mean': float(np.mean(E_lr)), 'std': float(np.std(E_lr)),
            'min': float(np.min(E_lr)), 'max': float(np.max(E_lr))}
        arm_defs = {
            args.arm_tag: 'CACE_energy / CACE_forces (combined, short + long range)',
            args.arm_tag + '_sr': f'E_tot - {args.lr_weight:g} * ewald_potential '
                                  '(long range removed)',
        }

    record = {
        'recorded_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'model': os.path.relpath(model, E2E),
        'model_sha256': sha256(model),
        'cace_provenance': prov,
        'lr_mix_weight': None if args.no_long_range else args.lr_weight,
        'valid_split': os.path.relpath(
            os.path.join(E2E, 'water', 'valid', 'set.000'), E2E),
        'n_frames': nf,
        'n_atoms': nat,
        'targets_agree_with_split': True,
        'e_target_peratom_rms_ev': e_target_peratom,
        'f_target_rms_ev_per_a': f_target_rms,
        'lr_energy_ev_per_frame': lr_stats,
        'mechanism': mech,
        'arm_defs': arm_defs,
        'metrics': rows,
    }

    with open(args.out, 'w') as fh:
        json.dump(record, fh, indent=2)
        fh.write('\n')
    print(f'wrote   {args.out}')

    if args.npz:
        if args.no_long_range:
            np.savez_compressed(
                args.npz, E_ref=E_ref, F_ref=F_ref,
                E_cace=np.asarray(E_tot), F_cace=reshape(F_tot))
        else:
            np.savez_compressed(
                args.npz,
                E_ref=E_ref, F_ref=F_ref,
                E_cace=np.asarray(E_tot), F_cace=reshape(F_tot),
                E_cace_sr=np.asarray(E_sr), F_cace_sr=reshape(F_sr),
                q=np.asarray(q), E_lr=np.asarray(E_lr), F_lr=reshape(F_lr),
            )
        print(f'wrote   {args.npz}')

    print()
    hdr = (f'{"arm":<14} {"E/atom RMSE":>12} {"(% tgt)":>9} {"E R2":>8} '
           f'{"F RMSE":>10} {"(% tgt)":>9} {"F R2":>8}')
    print(hdr)
    print('-' * len(hdr))
    for arm in rows:
        m = rows[arm]
        print(f'{arm:<14} {m["e_rmse"]:>12.4e} {m["e_rmse_pct_of_target"]:>8.2f}% '
              f'{m["e_r2"]:>8.5f} {m["f_rmse"]:>10.4e} '
              f'{m["f_rmse_pct_of_target"]:>8.2f}% {m["f_r2"]:>8.5f}')
    print(f'\ntarget scales: per-atom energy spread {e_target_peratom:.4e} eV/atom, '
          f'force RMS {f_target_rms:.4e} eV/A')
    if args.no_long_range:
        print('this checkpoint has no long-range head, so there is no E_lr and '
              'no mechanism block')
        return 0
    print(f'long-range term: ewald_potential mean {np.mean(E_lr):+.4f} eV/frame '
          f'std {np.std(E_lr):.4f}, entering the reported energy at weight '
          f'{args.lr_weight:g} -> {args.lr_weight*np.mean(E_lr):+.4f} +- '
          f'{args.lr_weight*np.std(E_lr):.4f} eV')
    print(f'mechanism: E_lr std/tgt {mech["e_lr_std_over_target_std"]:.3f}, '
          f'E_sr std/tgt {mech["e_sr_std_over_target_std"]:.3f}, '
          f'corr {mech["corr_e_sr_e_lr"]:+.3f}, '
          f'F_lr/F_tot {mech["f_lr_rms_over_f_tot_rms"]:.3f}, '
          f'q channels {mech["q_out_channels"]}, '
          f'net Q/frame {mech["net_charge_mean_per_frame"]:+.3f}, '
          f'q by element {mech["q_mean_by_element"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
