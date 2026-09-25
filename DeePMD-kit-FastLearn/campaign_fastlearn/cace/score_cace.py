#!/usr/bin/env python
# coding: utf-8
"""Score a cace campaign checkpoint over the campaign's own 80-frame valid split.

The first campaign's `record_cace_valid.py` scores against
`cace_compare_e2e/water/valid` and refuses to run unless the cace loader's split
is geometry-identical to it, which is the right guard there. Here the split is
different by construction - the campaign moved both codes onto the FastLearn
frames, and the cace arms were trained with an EXPLICIT `--valid`
(`xyz/valid.xyz`, see each run's `run_settings.json`) instead of the loader's own
`valid_fraction` draw. So this scores against that same explicit split, and takes
the energy/force targets out of each batch rather than from a separate reference
array: with the target travelling alongside the prediction, a split mismatch
cannot silently produce a number. A separate check re-derives the frames and
compares them against the deepmd side's `data/data_3`.

Deliberate differences from `fit_cace.py`: none in the data path (same xyz, same
cutoff, same `data_key={'energy': 'E_total'}`, same atomic_energies, so the target
is the same residual the model was trained on). The reported energy error is
therefore in each code's OWN reference convention - cace's residual carries the
+72.9 eV/frame offset noted in `fit_cace.py`, deepmd absorbs its own -29945 - which
is why the bias/spread split matters when the two codes are laid side by side.

The long-range arm emits `CACE_energy = SR_energy + ewald_potential` at unit
weight (FeatureAdd, not the 0.01 CombinePotential of the shipped scripts), so the
long-range term is `ewald_potential` itself.

The two `sea-` arms (`fit_cace_sea.py`, DeepMD's se_a in place of cace's Cace) are
scored by the same pass, with one difference in the loader: their checkpoints hold
a state dict plus a rebuild recipe rather than cace's whole-module pickle, because a
deepmd network cannot be pickled at all, so they go through
`sea_seam.load_sea_checkpoint` instead of `torch.load`. Nothing downstream changes -
`sea-lr` emits the same `SR_energy`/`ewald_potential`/`q` keys as `lr`, and both sea
arms' `train.log` carries the same phase/epoch lines as the cace arms, so
`--epochs-tsv` parses it unchanged.

Usage:
    python score_cace.py --arm lr --model runs/cace-lr/model-4.pth --tag cace-lr_e500
    python score_cace.py --arm sr --model runs/cace-sr/model-4.pth --tag cace-sr_e500
    python score_cace.py --arm sea-lr --model runs/cace-sea-lr/model-4.pth
    python score_cace.py --arm sea-sr --model runs/cace-sea-sr/model-4.pth
    python score_cace.py --arm sea-lr --epochs-only --log runs/cace-sea-lr/train.log \
        --epochs-tsv runs/cace-sea-lr/epoch_metrics.tsv   # trace, no checkpoint needed

The drivers walk every kept checkpoint of an arm rather than making you type one
line per file, and they are also what writes each arm's `epoch_metrics.tsv`:
    ./sweep_cace.sh     # runs/cace-{sr,lr}/*
    ./sweep_sea.sh      # runs/cace-sea-{sr,lr}/*
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
CAMPAIGN = os.path.dirname(HERE)
COMPARE = os.path.join(os.path.dirname(CAMPAIGN), 'cace_compare_e2e', 'compare')
sys.path.insert(0, COMPARE)

from metrics import metrics  # noqa: E402 - path set above

DEFAULT_CACE_ROOT = '/root/app/cace-ts'
DEFAULT_VALID = os.path.join(CAMPAIGN, 'xyz', 'valid.xyz')
DEFAULT_TRAIN = os.path.join(CAMPAIGN, 'xyz', 'train.xyz')
# the reference pair `fit_cace.py` passes; the composition is constant so only the
# sum is identifiable, and it only sets the target's level
ATOMIC_ENERGIES = {1: -187.6043857100553, 8: -93.80219285502734}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def import_cace(cace_root):
    """Import cace from `cace_root` and assert it is where it came from.

    A checkpoint pickles its module classes by name, so the branch that is
    imported decides what code runs; a cached module from another branch would
    silently score the checkpoint with the wrong classes.
    """
    root = os.path.abspath(cace_root)
    sys.path.insert(0, root)
    import cace
    got = os.path.abspath(cace.__file__)
    if not got.startswith(root + os.sep):
        raise RuntimeError(f'cace imported from {got}, not from {root}')
    return cace


def build_loader(cace_root, train_xyz, valid_xyz, cutoff, batch_size):
    cace = import_cace(cace_root)
    from cace.tasks.load_data import get_dataset_from_xyz, load_data_loader

    cace.tools.setup_logger(level='WARNING')
    torch.set_default_dtype(torch.float32)
    collection = get_dataset_from_xyz(
        train_path=train_xyz, valid_path=valid_xyz, cutoff=cutoff,
        data_key={'energy': 'E_total', 'forces': 'force'},
        atomic_energies=ATOMIC_ENERGIES,
    )
    return cace, load_data_loader(collection, 'valid', batch_size), collection


def check_split_against_deepmd(collection, deepmd_valid):
    """The loader's valid frames must be the deepmd arms' validation frames.

    Compared on geometry (positions and cell), which pins the frames and their
    order and does not depend on either code's energy convention. The energies
    are compared too, but as a LEVEL check with a reported residual: the cace
    target is the residual (total minus the fixed atomic_energies), the deepmd
    npy target is the total, and the gap between them should be the constant
    composition sum every frame.
    """
    frames = list(collection.valid)
    pos = np.array([f.get_positions() for f in frames], dtype=np.float64)
    cell = np.array([f.get_cell()[:] for f in frames], dtype=np.float64)
    try:
        e_cace = np.array([f.get_potential_energy() for f in frames], dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - the loader may not carry energies
        e_cace = None
        out_note = f'{type(exc).__name__}: {exc}'

    d = os.path.join(deepmd_valid, 'set.000')
    coord = np.load(os.path.join(d, 'coord.npy'))
    box = np.load(os.path.join(d, 'box.npy'))
    e_deepmd = np.load(os.path.join(d, 'energy.npy')).reshape(-1)
    nat = coord.shape[1] // 3
    coord = coord.reshape(-1, nat, 3)
    box = box.reshape(-1, 3, 3)

    out = {'nframes_cace': len(pos), 'nframes_deepmd': len(coord), 'natoms': nat}
    if e_cace is None:
        out['energy_from_frame_objects'] = f'unavailable ({out_note})'
    else:
        out['energy_from_frame_objects'] = 'available'
    if pos.shape != coord.shape:
        out['geometry_match'] = False
        out['reason'] = f'frame/atom mismatch {pos.shape} vs {coord.shape}'
        return out
    dpos = float(np.abs(pos - coord).max())
    dcell = float(np.abs(cell - box).max())
    out['max_dpos_a'] = dpos
    out['max_dcell_a'] = dcell
    # Tolerance is 1e-6 A, not exact: the cace side reads the frames back out of
    # `xyz/valid.xyz`, whose decimal text is narrower than float64, while the
    # deepmd side reads its own npy. A few 1e-9 A is that round-trip. A genuinely
    # different frame set would show a lattice-vector or displaced-atom gap
    # (order 1e-2 to 12 A), and a different ordering would scatter across frames -
    # neither is anywhere near this band. The residual is reported either way.
    out['geometry_match'] = bool(dpos <= 1e-6 and dcell <= 1e-6)
    if e_cace is not None:
        # per-frame gap: should be a constant (the reference sum), not a scatter
        gap = e_deepmd - e_cace
        out['energy_gap_mean_ev'] = float(gap.mean())
        out['energy_gap_std_ev'] = float(gap.std())
    return out


def load_model(cace, arm, model_path, device):
    """Load a campaign checkpoint. Returns ``(model, rebuild_spec)``.

    The cace arms wrote whole-module pickles, so a bare ``torch.load`` is right
    for them. The sea arms could not: `cace.tasks.train.TrainingTask.save_model`
    pickles the module, and a deepmd network is not picklable at all (its class is
    built inside the factory), so those files hold a state dict plus a rebuild
    recipe and have to go through `sea_seam.load_sea_checkpoint`.

    `sea_seam` is imported lazily, here rather than at module level, so scoring a
    cace checkpoint does not need deepmd importable at all. The recipe in the file
    is the source of truth for the architecture, which is why the returned arm is
    what the checkpoint says it is, not what the CLI asked for.
    """
    if arm.startswith('sea-'):
        sys.path.insert(0, HERE)
        import sea_seam
        # `build_model` reaches for `cace.modules.atomwise.Atomwise` by attribute,
        # so the subpackage has to be imported even though nothing here names it
        import cace.modules  # noqa: F401
        model, rb = sea_seam.load_sea_checkpoint(cace, model_path, device)
        return model, rb
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    model.to(device)
    return model, None


def forward_arm(model, loader, device, want_lr):
    """Predicted (E, F) over every valid batch, plus the long-range terms.

    The long-range force needs its own pass: cace's modules cache their outputs in
    the batch dict (an idempotence guard), so a second forward on the same dict
    hands back the first pass's tensor, and differentiating it hits a graph the
    model's own Forces module already consumed. So pass 2 swaps each Forces
    module's ``energy_key`` to ``ewald_potential`` and recomputes ``CACE_forces``
    from a fresh batch - the same trick the pre-update design's ``forces_lr``
    module used, and the reason `_long_range_forces` in the first campaign's
    harness exists.
    """
    E, F, Eref, Fref, E_lr, Q = [], [], [], [], [], []
    with torch.enable_grad():
        for batch in loader:
            bd = batch.to(device).to_dict()
            bd['positions'] = bd['positions'].detach().clone().requires_grad_(True)
            pred = model(bd, training=False)
            E.append(pred['CACE_energy'].detach().cpu().numpy().reshape(-1))
            F.append(pred['CACE_forces'].detach().cpu().numpy())
            Eref.append(bd['energy'].detach().cpu().numpy().reshape(-1))
            Fref.append(bd['forces'].detach().cpu().numpy())
            if want_lr:
                E_lr.append(pred['ewald_potential'].detach().cpu().numpy().reshape(-1))
                Q.append(pred['q'].detach().cpu().numpy())

    out = {
        'E': np.concatenate(E), 'F': np.concatenate(F),
        'E_ref': np.concatenate(Eref), 'F_ref': np.concatenate(Fref),
    }

    if want_lr:
        swapped = [(m, m.energy_key) for m in model.modules()
                   if type(m).__name__ == 'Forces' and getattr(m, 'energy_key', None)]
        if not swapped:
            raise RuntimeError('no Forces module to derive long-range forces with')
        F_lr, Eref2 = [], []
        try:
            for m, _old in swapped:
                m.energy_key = 'ewald_potential'
            with torch.enable_grad():
                for batch in loader:
                    bd = batch.to(device).to_dict()
                    bd['positions'] = bd['positions'].detach().clone().requires_grad_(True)
                    pred = model(bd, training=False)
                    F_lr.append(pred['CACE_forces'].detach().cpu().numpy())
                    Eref2.append(bd['energy'].detach().cpu().numpy().reshape(-1))
        finally:
            for m, old in swapped:
                m.energy_key = old
        # the two passes must have visited the same frames in the same order, or
        # F_lr and F_tot describe different frames
        if not np.array_equal(np.concatenate(Eref2), out['E_ref']):
            raise RuntimeError('pass 2 saw a different frame order than pass 1')
        out['E_lr'] = np.concatenate(E_lr)
        out['F_lr'] = np.concatenate(F_lr)
        out['q'] = np.concatenate(Q)

        # Pass 3 prices the self-interaction term the same way the les side prices
        # it: flip the kernel flag and difference the total energy and force. The
        # term is a function of the charges, so for a learned q head its gradient
        # is a real force on the atoms, and that force is exactly what this pass
        # measures. cace's own default is False (and fit_cace.py passes it
        # explicitly), so this is the term both codes trained with.
        ews = [m for m in model.modules() if hasattr(m, 'remove_self_interaction')]
        if not ews:
            raise RuntimeError('no module with remove_self_interaction to flip')
        flags = [bool(m.remove_self_interaction) for m in ews]
        # Read the kernel's own parameters off the loaded module rather than
        # assuming them: which flag the arm trained with is the point of the
        # column, and norm_factor is what turns the energy into sum(q^2).
        out['ewald_params'] = [
            {k: getattr(m, k) for k in
             ('norm_factor', 'sigma', 'dl', 'remove_self_interaction')
             if hasattr(m, k)} for m in ews]
        E3, F3, Eref3 = [], [], []
        try:
            for m in ews:
                m.remove_self_interaction = True
            with torch.enable_grad():
                for batch in loader:
                    bd = batch.to(device).to_dict()
                    bd['positions'] = bd['positions'].detach().clone().requires_grad_(True)
                    pred = model(bd, training=False)
                    E3.append(pred['CACE_energy'].detach().cpu().numpy().reshape(-1))
                    F3.append(pred['CACE_forces'].detach().cpu().numpy())
                    Eref3.append(bd['energy'].detach().cpu().numpy().reshape(-1))
        finally:
            for m, f in zip(ews, flags):
                m.remove_self_interaction = f
        if not np.array_equal(np.concatenate(Eref3), out['E_ref']):
            raise RuntimeError('pass 3 saw a different frame order than pass 1')
        out['E_noself'] = np.concatenate(E3)
        out['F_noself'] = np.concatenate(F3)
    del model
    return out


def decompose(E_pred, E_ref, nat):
    err = (E_pred - E_ref) / nat
    bias = float(err.mean())
    spread = float(err.std())
    rmse = float(np.sqrt((err**2).mean()))
    return {'rmse': rmse, 'bias': bias, 'spread': spread,
            'bias2_over_rmse2': float(bias**2 / rmse**2)}, err


def parse_epochs(log_path):
    """Global epoch, block, energy weight, lr and the two losses, from train.log.

    Each epoch's line is emitted twice (a bare print and a timestamped INFO
    duplicate), so only lines anchored at `Epoch ` are kept. The epoch counter is
    BLOCK-LOCAL (it restarts at each phase block), so the global epoch is the sum
    of the previous blocks' lengths plus the local one.
    """
    import re
    ph = re.compile(r'^phase (\d+) \(E weight ([\d.]+)\) block (\d+)/(\d+), (\d+) epochs')
    ep = re.compile(r'^Epoch (\d+), Train Loss: ([-\d.eE+]+), Val Loss: ([-\d.eE+]+)\s*$')
    st = re.compile(r'^##### Step: (\d+) Learning rate: ([\d.eE+-]+) #####')
    rows, done, cur, cur_lr = [], 0, None, None
    for line in open(log_path, errors='replace'):
        line = line.rstrip('\n')
        m = ph.match(line)
        if m:
            if cur is not None:
                done += cur['block_max']
            cur = {'phase': int(m.group(1)), 'e_weight': float(m.group(2)),
                   'block': int(m.group(3)), 'block_max': 0}
            continue
        m = st.match(line)
        if m:
            cur_lr = float(m.group(2))
            continue
        m = ep.match(line)
        if m and cur is not None:
            n = int(m.group(1))
            cur['block_max'] = max(cur['block_max'], n)
            rows.append((done + n, cur['block'], n, cur['phase'], cur['e_weight'],
                         cur_lr, float(m.group(2)), float(m.group(3))))
    cols = ['global_epoch', 'block', 'block_epoch', 'phase', 'energy_weight',
            'lr', 'train_loss', 'val_loss']
    return cols, rows


def write_epochs_tsv(path, cols, rows):
    with open(path, 'w') as fh:
        fh.write('\t'.join(cols) + '\n')
        for r in rows:
            fh.write('\t'.join('' if v is None else f'{v:g}' if isinstance(v, float)
                               else str(v) for v in r) + '\n')
    print(f'wrote   {path}  ({len(rows)} epochs, cols {cols})')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=['sr', 'lr', 'sea-sr', 'sea-lr'], required=True)
    ap.add_argument('--model', help='checkpoint to score (unless --epochs-only)')
    ap.add_argument('--tag', help='row tag; default: <arm>_<checkpoint stem>')
    ap.add_argument('--cace-root', default=DEFAULT_CACE_ROOT)
    ap.add_argument('--train-xyz', default=DEFAULT_TRAIN)
    ap.add_argument('--valid-xyz', default=DEFAULT_VALID)
    ap.add_argument('--deepmd-valid', default=os.path.join(
        os.path.dirname(CAMPAIGN), 'data', 'data_3'))
    ap.add_argument('--cutoff', type=float, default=5.5)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--out-json')
    ap.add_argument('--npz')
    ap.add_argument('--epochs-tsv')
    ap.add_argument('--log', help='the arm\'s train.log, for --epochs-only')
    ap.add_argument('--epochs-only', action='store_true',
                    help='write --epochs-tsv from the arm log and exit, without '
                         'loading a checkpoint: the training trace is collectable '
                         'while the run is still going')
    args = ap.parse_args()

    # The trace is a pure function of the log, so it is refreshable mid-run - which
    # matters because `collect_sweep.py` gathers it per arm and the pod pulls the
    # logs home every few minutes, long before any checkpoint is scored.
    if args.epochs_only:
        if not (args.log and args.epochs_tsv):
            ap.error('--epochs-only needs --log and --epochs-tsv')
        cols, rows_ep = parse_epochs(args.log)
        write_epochs_tsv(args.epochs_tsv, cols, rows_ep)
        return 0

    if not args.model:
        ap.error('--model is required unless --epochs-only')
    if not os.path.exists(args.model):
        print(f'model not found: {args.model}')
        return 1
    want_lr = args.arm in ('lr', 'sea-lr')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = args.tag or f'cace-{args.arm}_{os.path.splitext(os.path.basename(args.model))[0]}'

    print(f'arm     {args.arm}\nmodel   {args.model}')
    print(f'        sha256[:12] = {sha256(args.model)[:12]}')
    cace, loader, collection = build_loader(
        args.cace_root, args.train_xyz, args.valid_xyz, args.cutoff, args.batch_size)
    print(f'cace    {os.path.abspath(cace.__file__)}')
    print(f'device  {dev}')

    chk = check_split_against_deepmd(collection, args.deepmd_valid)
    print('split   ' + json.dumps(chk))

    model, rb = load_model(cace, args.arm, args.model, dev)
    if rb is not None:
        print(f'sea ckpt  arm {rb["arm"]} seed {rb["seed"]} dim_out {rb["dim_out"]} '
              f'nsel {rb["nsel"]} n_embed_nets {rb["n_embed_nets"]}')
    res = forward_arm(model, loader, dev, want_lr)
    nat = 192
    m = metrics(res['E'], res['F'], res['E_ref'], res['F_ref'], nat)
    dec, err = decompose(res['E'], res['E_ref'], nat)
    rows = {tag: {**m, **dec}}
    print()
    print(f'{tag}: e_rmse={m["e_rmse"]:.6e} eV/atom  f_rmse={m["f_rmse"]:.6e} eV/A  '
          f'e_r2={m["e_r2"]:.5f} f_r2={m["f_r2"]:.5f}')
    print(f'{" " * len(tag)}  bias={dec["bias"]:+.6e} spread={dec["spread"]:.6e} '
          f'bias2/rmse2={dec["bias2_over_rmse2"]:.3f}')

    mech = None
    if want_lr:
        E_lr = res['E_lr']
        f_tot = float(np.sqrt(np.mean(res['F']**2)))
        f_lr = float(np.sqrt(np.mean(res['F_lr']**2)))
        # q is (nf*nat, n_out); a per-frame net charge needs the atom count
        q = res['q'].reshape(-1, nat, res['q'].shape[-1])
        sum_q2 = float((q ** 2).sum(axis=(1, 2)).mean())  # per frame
        pars = res['ewald_params']
        # Pass 3 (flag flipped) prices the self term by differencing, the same way
        # the les side prices it - so this force is measured, not modelled from
        # the sum(q^2) formula. The two should agree, which is the sanity check.
        e_self = float((res['E'] - res['E_noself']).mean())
        f_self = res['F'] - res['F_noself']
        f_self_rms = float(np.sqrt((f_self ** 2).mean()))
        mech = {
            'e_lr_mean_ev_per_frame': float(E_lr.mean()),
            'e_lr_std_ev_per_frame': float(E_lr.std()),
            'f_lr_rms_ev_per_a': f_lr,
            'f_lr_over_f_tot': f_lr / f_tot,
            'e_self_ev_per_frame': e_self,
            'f_self_rms_ev_per_a': f_self_rms,
            'f_self_over_f_tot': f_self_rms / f_tot,
            'q_out_channels': int(res['q'].shape[-1]),
            'q_mean': float(q.mean()),
            'net_charge_mean_per_frame': float(q.sum(axis=(1, 2)).mean()),
            'sum_q2_mean_per_frame': sum_q2,
            'ewald_params': pars,
        }
        if pars and 'norm_factor' in pars[0]:
            # the self-interaction term the kernel keeps when the flag is False:
            # +norm_factor * sum(q^2) / (sigma * (2*pi)^1.5)
            self_e = (pars[0]['norm_factor'] / (pars[0]['sigma'] * (2 * np.pi) ** 1.5)
                      * sum_q2)
            mech['self_term_ev_per_frame'] = float(self_e)
            mech['self_over_e_lr'] = float(self_e / E_lr.mean()) if E_lr.mean() else None
        print(f'mechanism: E_lr {E_lr.mean():+.4f} +- {E_lr.std():.4f} eV/frame, '
              f'F_lr/F_tot {mech["f_lr_over_f_tot"]:.3f}, '
              f'q channels {mech["q_out_channels"]}, '
              f'q mean {mech["q_mean"]:+.4f}, '
              f'net Q/frame {mech["net_charge_mean_per_frame"]:+.4f}, '
              f'sum q^2 {sum_q2:.2f}/frame, ewald {pars}')
        print(f'{" " * 11}self term (flag-flip) {e_self:+.3f} eV/frame, '
              f'F_self/F_tot {mech["f_self_over_f_tot"]:.4f}')
        if 'self_term_ev_per_frame' in mech:
            print(f'{" " * 11}self term (from sum q^2) '
                  f'{mech["self_term_ev_per_frame"]:+.3f} eV/frame '
                  f'= {mech["self_over_e_lr"]:.2f}x E_lr (near-cancelling)')

    if args.out_json:
        rec = {
            'recorded_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
            'arm': args.arm, 'tag': tag, 'model': os.path.relpath(args.model, CAMPAIGN),
            'model_sha256': sha256(args.model),
            'cace_root': args.cace_root,
            'checkpoint_format': 'cace-module-pickle' if rb is None else rb['format'],
            # carried on the score record so it is self-contained: which se_a block
            # and seed the checkpoint was trained with, and from which cace checkout
            'sea_rebuild': rb,
            'valid_xyz': args.valid_xyz, 'valid_xyz_sha256': sha256(args.valid_xyz),
            'cutoff': args.cutoff, 'batch_size': args.batch_size,
            'atomic_energies': ATOMIC_ENERGIES,
            'lr_mix_weight': 1.0 if want_lr else None,
            'n_frames': int(len(res['E'])), 'n_atoms': nat,
            'split_check_vs_deepmd': chk,
            'mechanism': mech,
            'metrics': rows,
        }
        with open(args.out_json, 'w') as fh:
            json.dump(rec, fh, indent=2)
            fh.write('\n')
        print(f'wrote   {args.out_json}')

    if args.npz:
        d = {'E_pred': res['E'], 'E_ref': res['E_ref'], 'err_per_atom': err,
             'F_pred': res['F'], 'F_ref': res['F_ref']}
        if want_lr:
            d.update({'E_lr': res['E_lr'], 'F_lr': res['F_lr'], 'q': res['q'],
                      'E_noself': res['E_noself'], 'F_noself': res['F_noself']})
        np.savez_compressed(args.npz, **d)
        print(f'wrote   {args.npz}')

    if args.epochs_tsv:
        cols, rows_ep = parse_epochs(os.path.join(os.path.dirname(args.model), 'train.log'))
        write_epochs_tsv(args.epochs_tsv, cols, rows_ep)
    return 0


if __name__ == '__main__':
    sys.exit(main())
