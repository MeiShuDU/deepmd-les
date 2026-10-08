#!/usr/bin/env python
# coding: utf-8
"""The author's ``fit-water-timing`` recipe, on the DeePMD-kit-FastLearn data.

This is a re-pointing of the two shipped scripts
(``fit-water-mp0-lr-FeatureAdd/fit_cace_new.py`` and ``fit-water-mp0-sr/fit-cace-nnp.py``),
which are identical except for the head that turns the representation into an
energy. Both are reproduced here behind ``--arm`` so the shared settings cannot
drift apart:

  ``--arm lr``  Cace -> Atomwise('SR_energy') + Atomwise('q') -> EwaldPotential
                -> FeatureAdd('CACE_energy'), i.e. a learned latent charge
                feeding an Ewald sum at unit weight
  ``--arm sr``  Cace -> Atomwise('CACE_energy'), the fitted long-range term's
                control: same representation, same data, no long range

Recipe, unchanged from the water-timing scripts: cutoff 5.5; BesselRBF(6) and
PolynomialCutoff; ``Cace(zs=[1,8], n_atom_basis=3, embed_receiver_nodes=True,
n_radial_basis=12, max_l=3, max_nu=3, num_message_passing=0, type_message_passing=['Bchi'])``;
SR head ``[32,16]``; q head ``[24,12]``, no bias; ``EwaldPotential(dl=2,
sigma=1.0, remove_self_interaction=False, aggregation_mode='sum')``; Adam
lr 1e-2 betas (0.99,0.999); ``StepLR(20, 0.5)``; ``max_grad_norm=10``;
``warmup_steps=5``; ``ema=False``; float32; force weight 1000 throughout; energy
weight 0.1 for five 40-epoch blocks, then 1, 10, 1000 for 100 epochs each
(500 epochs, 80,000 optimizer steps at batch 2 over 320 frames).

Three deliberate differences from the shipped scripts, all forced by the data
move and recorded in the provenance JSON:

1. the split is explicit (``--valid``) rather than ``valid_fraction``, so the
   cace arms train and validate on exactly the frames the deepmd arms do;
2. ``data_key`` maps the energy to ``E_total``. The shipped scripts pass
   ``{'energy': 'energy'}``, and under the ase installed here a bare
   ``energy=`` header lands in a calculator rather than ``atoms.info``, where
   ``AtomicData.from_atoms`` looks - so the shipped script fits forces and no
   energy at all. See ``export_data.py``; ``--guard`` below is the tripwire;
3. the cace checkout is named explicitly (``--cace-root``) because the shipped
   scripts rely on a relative ``../cace/`` that is absent from the data repo,
   so whichever cace is importable gets used silently.

Usage:
    python fit_cace.py --arm lr --out-dir runs/cace_lr
    python fit_cace.py --arm sr --out-dir runs/cace_sr
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys

DEFAULT_CACE_ROOT = '/root/app/cace-ts'
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN = os.path.join(HERE, '..', 'xyz', 'train.xyz')
DEFAULT_VALID = os.path.join(HERE, '..', 'xyz', 'valid.xyz')

CUTOFF = 5.5
TRAIN_BATCH = 2
VALID_BATCH = 4
# The reference energies the shipped scripts pass. The composition of this data
# is constant (64 O + 128 H every frame), so this pair is not identifiable from
# it and only its sum sets the target's level; the residual here is +72.9
# eV/frame, which the fitting bias absorbs exactly as deepmd absorbs -29945.
ATOMIC_ENERGIES = {1: -187.6043857100553, 8: -93.80219285502734}
# energy weight per phase, and the epochs of each phase; 5 copies of the first
ENV_PHASES = [(0.1, 5, 40), (1.0, 1, 100), (10.0, 1, 100), (1000.0, 1, 100)]
FORCE_WEIGHT = 1000.0
# the shipped scripts save best_model.pth into the CWD, so both change into the
# run directory and write relative names exactly as the author does
BEST = 'best_model.pth'
PHASE_CKPT = {0: 'model.pth', 1: 'model-2.pth', 2: 'model-3.pth', 3: 'model-4.pth'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cace_provenance(cace_root):
    import cace
    prov = {'imported_from': os.path.abspath(cace.__file__),
            'version': getattr(cace, '__version__', None),
            'requested_root': os.path.abspath(cace_root)}
    try:
        prov['commit'] = subprocess.run(['git', '-C', cace_root, 'rev-parse', 'HEAD'],
                                        capture_output=True, text=True, check=True).stdout.strip()
        prov['dirty'] = subprocess.run(['git', '-C', cace_root, 'status', '--porcelain'],
                                       capture_output=True, text=True,
                                       check=True).stdout.strip() != ''
    except Exception:  # noqa: BLE001 - provenance is best-effort
        pass
    return prov


def guard_batches(loader, needed, where):
    """Fail loudly if a target the recipe trains on is silently absent.

    This is the tripwire for the ``energy=``-key trap: without it the run would
    spend hours fitting forces only and look like a result.
    """
    batch = next(iter(loader))
    keys = list(batch.keys)
    missing = [k for k in needed if k not in keys]
    if missing:
        raise SystemExit(f'FAIL: the {where} batch carries no {missing}; '
                         f'data_key/format is wrong, refusing to train blind.')
    print(f'  {where}: keys {sorted(keys)}')
    print(f'  {where}: energy {[round(float(x), 4) for x in batch["energy"]]}, '
          f'forces {tuple(batch["forces"].shape)}')
    return batch


def count_trainable_params(model, probe, device):
    """Trainable parameters AFTER the lazily-built heads exist.

    cace's `Atomwise` leaves its output net as `None` and builds it on the first
    forward, because the input width is only known from the representation then.
    A count taken at construction time therefore reads only the representation -
    2,610 of this arm's 41,072 (24,572 of the SR arm's) - and the recorded
    `n_trainable_params` would be a description of the descriptor, not the model.

    The probe forward's random draws are discarded by `fork_rng`, so counting
    cannot move the training recipe; it also surfaces a shape/key error at startup
    rather than an hour in, which is what `--check-only`'s guard is for.
    """
    import torch  # imported in main() by design; by now it is in sys.modules

    devs = [] if device.type != 'cuda' else [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=devs), torch.enable_grad():
        model(probe.to(device).to_dict(), training=True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=['lr', 'sr'], required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--train', default=DEFAULT_TRAIN)
    ap.add_argument('--valid', default=DEFAULT_VALID)
    ap.add_argument('--cace-root', default=DEFAULT_CACE_ROOT)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--check-only', action='store_true',
                    help='load the data, apply the guards, build the model and '
                         'run one forward pass, then exit - a pre-flight check '
                         'that costs seconds instead of a GPU session')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # the shipped scripts write best_model.pth and checkpoint.pt to the CWD, and
    # pass relative paths to save_model; running from the out-dir keeps that
    # idiom and keeps the runs from clobbering each other
    os.chdir(args.out_dir)

    sys.path.insert(0, args.cace_root)
    import numpy as np  # noqa: E402 - after sys.path so the branch wins
    import torch  # noqa: E402
    import cace  # noqa: E402
    from cace.representations import Cace  # noqa: E402
    from cace.modules import PolynomialCutoff, BesselRBF  # noqa: E402
    from cace.models.atomistic import NeuralNetworkPotential  # noqa: E402
    from cace.tasks.train import TrainingTask  # noqa: E402
    from cace.tasks import GetLoss, get_dataset_from_xyz, load_data_loader  # noqa: E402
    from cace.tools import Metrics  # noqa: E402 - the author's script takes it
    # from cace.tools, not cace.tasks, which does not re-export it

    torch.set_default_dtype(torch.float32)
    cace.tools.setup_logger(level='INFO')
    prov = cace_provenance(args.cace_root)
    print(f'cace     {prov["imported_from"]}'
          + (f' @ {prov["commit"][:12]}{" (dirty)" if prov.get("dirty") else ""}'
             if prov.get('commit') else ''))

    print('reading data')
    collection = get_dataset_from_xyz(
        train_path=args.train, valid_path=args.valid, cutoff=CUTOFF,
        data_key={'energy': 'E_total', 'forces': 'force'},
        atomic_energies=ATOMIC_ENERGIES,
    )
    train_loader = load_data_loader(collection=collection, data_type='train',
                                    batch_size=TRAIN_BATCH)
    valid_loader = load_data_loader(collection=collection, data_type='valid',
                                    batch_size=VALID_BATCH)
    steps_per_epoch = len(train_loader)
    total_steps = sum(n * e for _, n, e in ENV_PHASES) * steps_per_epoch
    print(f'train {len(collection.train)} frames -> {steps_per_epoch} steps/epoch; '
          f'valid {len(collection.valid)} frames -> {len(valid_loader)} batches')
    print(f'schedule: {[(w, n * e) for w, n, e in ENV_PHASES]} epochs -> '
          f'{total_steps} optimizer steps')

    device = cace.tools.init_device(args.device)
    print(f'device: {args.device}')
    train_probe = guard_batches(train_loader, ('energy', 'forces'), 'train')
    guard_batches(valid_loader, ('energy', 'forces'), 'valid')

    print('building Cace representation')
    cace_representation = Cace(
        zs=[1, 8], n_atom_basis=3, embed_receiver_nodes=True, cutoff=CUTOFF,
        cutoff_fn=PolynomialCutoff(cutoff=CUTOFF),
        radial_basis=BesselRBF(cutoff=CUTOFF, n_rbf=6, trainable=True),
        n_radial_basis=12, max_l=3, max_nu=3, num_message_passing=0,
        type_message_passing=['Bchi'],
        args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
        device=device, timeit=False,
    ).to(device)

    sr_energy = cace.modules.atomwise.Atomwise(
        n_layers=3, output_key='SR_energy', n_hidden=[32, 16],
        use_batchnorm=False, add_linear_nn=True)
    forces = cace.modules.Forces(energy_key='CACE_energy', forces_key='CACE_forces')

    if args.arm == 'lr':
        q = cace.modules.Atomwise(
            n_layers=3, n_hidden=[24, 12], n_out=1, per_atom_output_key='q',
            output_key='tot_q', residual=False, add_linear_nn=True, bias=False)
        ep = cace.modules.EwaldPotential(
            dl=2, sigma=1.0, feature_key='q', output_key='ewald_potential',
            remove_self_interaction=False, aggregation_mode='sum')
        e_add = cace.modules.FeatureAdd(
            feature_keys=['SR_energy', 'ewald_potential'], output_key='CACE_energy')
        output_modules = [sr_energy, q, ep, e_add, forces]
    else:
        # the SR control: same representation, one head straight to the energy
        sr_energy = cace.modules.atomwise.Atomwise(
            n_layers=3, output_key='CACE_energy', n_hidden=[32, 16],
            use_batchnorm=False, add_linear_nn=True)
        output_modules = [sr_energy, forces]

    cace_nnp = NeuralNetworkPotential(
        representation=cace_representation, output_modules=output_modules).to(device)

    n_params = count_trainable_params(cace_nnp, train_probe, device)
    print(f'trainable parameters: {n_params}')

    if args.check_only:
        # one real forward on the guard batch, so a shape/key problem surfaces
        # here rather than an hour into a run. The model takes a plain dict;
        # Batch.to_dict() is what TrainingTask.train_step feeds it.
        batch = train_probe.to(device)
        pred = cace_nnp(batch.to_dict(), training=True)
        keys = sorted(k for k, v in pred.items() if isinstance(v, torch.Tensor))
        print(f'  forward produced {keys}')
        for key in ('CACE_energy', 'CACE_forces'):
            if key in pred:
                t = pred[key].detach()
                print(f'  {key} {tuple(t.shape)} '
                      f'{"" if t.numel() > 8 else [round(float(x), 4) for x in t.reshape(-1)]}')
        pred['CACE_energy'].sum().backward()
        g = sum(float(p.grad.abs().sum()) for p in cace_nnp.parameters()
                if p.grad is not None)
        print(f'  backward reached the model: sum|grad| = {g:.4e}')
        print('check-only: OK, nothing trained')
        return 0

    force_loss = GetLoss(target_name='forces', predict_name='CACE_forces',
                         loss_fn=torch.nn.MSELoss(), loss_weight=FORCE_WEIGHT)
    e_metric = Metrics(target_name='energy', predict_name='CACE_energy',
                       name='e/atom', per_atom=True)
    f_metric = Metrics(target_name='forces', predict_name='CACE_forces', name='f')

    optimizer_args = {'lr': 1e-2, 'betas': (0.99, 0.999)}
    scheduler_args = {'step_size': 20, 'gamma': 0.5}
    phase_records = []

    # The shipped script's shape, exactly: five 40-epoch blocks at energy weight
    # 0.1, then three 100-epoch blocks at 1, 10, 1000. The first five are
    # separated by *building a new TrainingTask*, which throws away Adam and the
    # StepLR and so restarts both at lr 1e-2; the last three share the fifth
    # task and only swap the loss, so their optimizer state and lr counter carry
    # on from wherever that block left them. Both halves of that are part of the
    # recipe as shipped, so both are reproduced rather than tidied into one
    # four-phase schedule, which would give a different lr trajectory.
    for phase, (e_weight, n_repeat, epochs) in enumerate(ENV_PHASES):
        energy_loss = GetLoss(target_name='energy', predict_name='CACE_energy',
                              loss_fn=torch.nn.MSELoss(), loss_weight=e_weight)
        for rep in range(n_repeat):
            if phase == 0:
                task = TrainingTask(
                    model=cace_nnp, losses=[energy_loss, force_loss],
                    metrics=[e_metric, f_metric], device=device,
                    optimizer_args=optimizer_args,
                    scheduler_cls=torch.optim.lr_scheduler.StepLR,
                    scheduler_args=scheduler_args, max_grad_norm=10,
                    ema=False, ema_start=10, warmup_steps=5)
            else:
                task.update_loss([energy_loss, force_loss])
            print(f'phase {phase} (E weight {e_weight}) block {rep + 1}/{n_repeat}, '
                  f'{epochs} epochs, fresh task: {phase == 0}')
            # defaults kept as shipped: checkpoint.pt every 10 epochs, and
            # best_model.pth overwritten by each fit call, so the one left at the
            # end is the last phase's best-validation checkpoint
            task.fit(train_loader, valid_loader, epochs=epochs, screen_nan=False)
        task.save_model(PHASE_CKPT[phase])
        phase_records.append({'energy_weight': e_weight, 'epochs': n_repeat * epochs,
                              'steps': n_repeat * epochs * steps_per_epoch,
                              'ckpt': PHASE_CKPT[phase]})

    print(f'best checkpoint from the final phase: {BEST}')

    if os.path.exists(BEST):
        try:
            torch.jit.script(torch.load(BEST, map_location=device)).save('best-scripted.pt')
            print('scripted best -> best-scripted.pt')
        except Exception as exc:  # noqa: BLE001 - scripting is best-effort, as shipped
            print(f'scripting the best model failed (as in the shipped script): {exc}')

    record = {
        'recorded_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        'arm': args.arm,
        'cace_provenance': prov,
        'train_xyz': os.path.abspath(args.train),
        'train_sha256': sha256(args.train),
        'valid_xyz': os.path.abspath(args.valid),
        'valid_sha256': sha256(args.valid),
        'n_train_frames': len(collection.train),
        'n_valid_frames': len(collection.valid),
        'batch_size': {'train': TRAIN_BATCH, 'valid': VALID_BATCH},
        'steps_per_epoch': steps_per_epoch,
        'total_steps': total_steps,
        'phases': phase_records,
        'cutoff': CUTOFF,
        'atomic_energies': ATOMIC_ENERGIES,
        'optimizer': optimizer_args,
        'scheduler': {'cls': 'StepLR', **scheduler_args},
        'max_grad_norm': 10,
        'warmup_steps': 5,
        'ema': False,
        'dtype': 'float32',
        'force_weight': FORCE_WEIGHT,
        'n_trainable_params': int(n_params),
        'data_key': {'energy': 'E_total', 'forces': 'force'},
    }
    with open('run_settings.json', 'w') as fh:
        json.dump(record, fh, indent=2)
        fh.write('\n')
    print('wrote run_settings.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
