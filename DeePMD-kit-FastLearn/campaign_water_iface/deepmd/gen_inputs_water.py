"""Generate the four deepmd arms for the water/slab-interface benchmark.

Companion to ``campaign_fastlearn/deepmd/gen_inputs_sea.py``: same idea (one
deepmd / deepmd-les pair per replicate, on the author's own training recipe),
different dataset and different short-range network.

What this pair isolates
-----------------------
``fit-water-interface`` ships cace runs (``fit-interface-mp0`` LR, and
``fit-interface-mp0-sr`` SR) whose short-range net is a small cace ``Cace``
representation plus an ``Atomwise`` head. This campaign keeps the author's
*schedule* knob for knob, but the descriptor and the fitting net are the deepmd
family's own - the ones the user specified - so the deepmd and cace arms differ
in a known, declared way rather than by accident.

What is matched, and to what
----------------------------
data        ``campaign_water_iface/data/{train,valid}``: the author's 500
            frames of 1566 atoms, split 450/50 by cace's own rule
            (``valid_fraction=0.1, seed=1``, ``np.random.default_rng(1).shuffle``
            over frame indices). ``energy.npy`` holds the residual
            ``E - sum_Z ref[Z]`` with the author's references, which is exactly
            what cace's ``AtomicData.from_atoms`` hands its loss.
descriptor  user-specified: ``se_a``, rcut 6.00 (the cace arms train at 5.5, so
            this is the one deliberate reach difference), rcut_smth 0.50,
            ``sel`` read from ``data/sel.txt``, neuron [25, 50, 100],
            axis_neuron 16, resnet_dt false.
SR head     user-specified: ``fitting_net {neuron: [240, 240, 240], activation
            silu, resnet_dt false}``.
optimiser   Adam at lr 1e-2, betas (0.99, 0.999), ``gradient_max_norm: 10``,
            ``warmup_steps: 2250`` on fresh blocks.
            deepmd hardcodes Adam betas (0.9, 0.999) and exposes no knob, so
            that one is looser.
schedule    cace's ``for i in range(5): fit(epochs=40)`` fresh-task loop at
            energy weight 0.01, then ``fit(100)`` at weight 1 and at weight 10
            (mp0-sr, 400 epochs), plus one more ``fit(100)`` at weight 1000 for
            mp0 (500 epochs). Reproduced as chained deepmd runs, one per block,
            each in its own directory.
LR          cace ``StepLR(step_size=20, gamma=0.5)`` on a per-task epoch
            counter. deepmd's ``exp`` schedule is itself a staircase,
            ``lr(s) = start_lr * decay_rate ** ((s - warmup) // decay_steps)``
            (``dpmodel/utils/learning_rate.py:55``), so ``decay_steps`` =
            20 epochs * 450 steps = 9000 with ``decay_rate 0.5`` reproduces
            cace's halving at epoch granularity. Each block then declares the
            ``start_lr`` cace's task counter holds at that block's first epoch:
            1e-2 for each of the five fresh blocks, then 2.5e-3, 7.8125e-5,
            2.44140625e-6 continuing the fifth.
E weight    cace's energy loss is ``w_E * MSE(E_frame)``; deepmd's is
            ``pref_e * MSE(E_frame) / natoms`` (``pt/train/wrapper.py``), so a
            flat ``pref_e = natoms * w_E`` gives the energy:force ratio cace
            trains under. natoms is 1566 here and is read from ``type.raw``,
            not assumed.
virial      ``pref_v = 0``: cace trains on energy and force only.
long range  the ``les_params`` block is cace's ``fit-interface-mp0`` mapping:
            charge head ``Atomwise(n_hidden=[24, 12], n_layers=3,
            add_linear_nn=True)``, ``EwaldPotential(dl=2, sigma=1)``, and
            cace's ``CombinePotential`` mixing weight 0.02 -> ``lr_weight``.
            ``remove_self_interaction: true`` is the author's actual recipe:
            the ``EwaldPotential`` shipped in this datarepo constructs it as
            true (``cace/modules/ewald.py:15`` in the ``/root/app/cace``
            checkout the author's checkpoints match). Unchanged and
            unavoidable: cace's charge head has ``bias=False`` and ``n_out=4``
            (four independent Ewald channels) while les's ``Atomwise`` is a
            single scalar charge with a bias, and the Ewald prefactor differs
            (cace ``norm_factor`` 1.0 vs les's hardcoded value). The charge net
            is free, so these are absorbed by it, but they are not nothing.

Chain semantics - why ``--init-model`` everywhere
-------------------------------------------------
cace builds five *fresh* ``TrainingTask`` objects (each rebuilding the optimiser
and the LR scheduler, so the warmup restarts), then reuses the fifth for the
later 100-epoch fits, so those keep Adam's moments and continue the StepLR
counter 40 -> 340.

deepmd maps that onto ``--init-model`` (finetune path) and ``--restart``:

* ``--init-model`` sets ``restart_training = False``
  (``pt/train/training.py:124``), which forces ``start_step = 0`` (``:452-455``)
  and skips the optimiser restore because the guard requires it
  (``:686-687``: ``if optimizer_state_dict is not None and
  self.restart_training``). So a block begins at local step 0 with a fresh
  Adam - exactly cace's fresh task.
* ``--restart`` restores the optimiser and sets ``start_step`` to the saved
  step, and the loop then runs ``range(self.start_step, self.num_steps)``
  (``:1131``) - ``numb_steps`` becomes an ABSOLUTE target.

``--restart`` therefore restores Adam, but it does *not* reproduce cace's LR on
these continuation blocks: the scheduler lambda is
``warm_up_linear(step + self.start_step, ...)`` (``:690``) fed by a freshly
constructed ``LambdaLR`` counter, so on ``s6`` the argument starts at the
accumulated step 90000, and ``value(90000)`` is ``0.01 * 0.5**10`` ~ 9.8e-6,
not cace's 2.5e-3. cace's StepLR counter is task-local; deepmd's accumulated
step is global, and the two disagree by 2**8 here. Making the arg continuous
would need a per-block ``decay_steps`` that is not cace's 9000.

So every block chains with ``--init-model`` and declares its own ``start_lr``.
The cost, stated plainly: s6/s7/s8 also reset Adam's moments where cace keeps
them. That is the one place this chain is looser than cace, and it is a few
dozen steps of re-adaptation in 225,000.

Layout
------
Each arm/replicate is one directory holding its block directories::

    runs/deepmd_sA/s1/input.yaml   s1/model.ckpt-18000.pt   ...
    runs/deepmd-les_sA/s{1..8}/...
    runs/deepmd_sA/segments.json   (the schedule, for the evaluator)
    runs/deepmd_sA/chain.sh

One directory per block because deepmd's step counter is per-run: it is what
lets each block keep its own lcurve.out, its own les.log and its own
checkpoints, and it is what makes the local checkpoint step convertible to a
global one (``segments.json`` carries each block's ``global_offset``).

Usage:
    python gen_inputs_water.py
    python gen_inputs_water.py --smoke          # 1-epoch blocks, for plumbing only
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
DATA = os.path.join(CAMPAIGN, 'data')

# cace's own recipe, in cace's units, for THIS dataset.
STEPS_PER_EPOCH = 450       # 450 training frames / batch_size 1
DECAY_STEPS = 9000          # 20 epochs: cace's StepLR step_size
DECAY_RATE = 0.5            # cace's StepLR gamma
BASE_LR = 1e-2
FORCE_PREF = 1000.0         # cace's force loss weight
WARMUP = 2250               # cace's warmup_steps=5, which count EPOCHS (5 * 450)
GRAD_MAX_NORM = 10.0        # cace's max_grad_norm
DISP_FREQ = 450             # one validation per epoch, as cace validates
SAVE_FREQ = 9000            # one checkpoint per LR step, so every block end lands
MAX_CKPT_KEEP = 30
VALID_BATCHES = 50          # the whole valid split, one frame per batch
SMOKE_STEPS = 6             # smoke: steps per block, so the chain is minutes not hours
SMOKE_WARMUP = 2
SMOKE_DECAY = 1

# (E weight, number of blocks, epochs per block) - cace's recipe per arm.
# mp0-sr stops after the weight-10 loop; mp0 adds the weight-1000 loop.
PHASES_SR = [(0.01, 5, 40), (1.0, 1, 100), (10.0, 1, 100)]
PHASES_LES = [(0.01, 5, 40), (1.0, 1, 100), (10.0, 1, 100), (1000.0, 1, 100)]

REPLICATES = [
    {'tag': 'A', 'desc_seed': 1, 'fit_seed': 1, 'train_seed': 10},
    {'tag': 'B', 'desc_seed': 2, 'fit_seed': 2, 'train_seed': 20},
]

TRAIN_DIR = 'train'
VALID_DIR = 'valid'

# Shared by both arms; the descriptor and head sizes are the user's.
DESC_NEURON = [25, 50, 100]
FIT_NEURON = [240, 240, 240]
RCUT = 6.00
RCUT_SMTH = 0.50

# cace fit-interface-mp0's own long-range recipe.
LES_COMMON = {
    'sigma': 1.0,
    'dl': 2.0,
    'remove_self_interaction': True,
    'output_scaling_factor': 1.0,
    'lr_weight': 0.02,
}

ARMS = {
    'deepmd': {
        'mtype': 'standard',
        'les': None,
        'phases': PHASES_SR,
        'doc': 'short-range only control, on cace\'s schedule',
    },
    'deepmd-les': {
        'mtype': 'hybrid_ener',
        'les': dict(LES_COMMON, local_charge=True, n_hidden=[24, 12], n_layers=3,
                    add_linear_nn=True),
        'phases': PHASES_LES,
        'doc': 'learned per-atom charge + Ewald, on cace\'s schedule; the charge '
               'head is cace\'s [24, 12]',
    },
}

TEMPLATE = """\
# {arm}, replicate {tag}, block {n}/{nseg} - generated by
# campaign_water_iface/deepmd/gen_inputs_water.py
# {doc}
#
# descriptor / SR head: user-specified for this campaign
# schedule: cace fit-interface {cace_arm} recipe, block {n} = E weight {w_e} for {epochs} epochs
# block LR: start_lr {start_lr} with decay_steps {decay_steps} / decay_rate {decay_rate},
#           i.e. cace's StepLR(20, 0.5) on this task's epoch counter
# chain:   run {chain_note}; --init-model rebuilds the optimiser and resets
#          the step counter, which is what cace does at every fresh block
model:
  type: {mtype}
  type_map: ["O", "H"]
  descriptor:
    type: se_a
    sel: {sel}
    rcut_smth: {rcut_smth}
    rcut: {rcut}
    neuron: {desc_neuron}
    axis_neuron: 16
    resnet_dt: false
    seed: {desc_seed}
  fitting_net:
    neuron: {fit_neuron}
    activation_function: silu
    resnet_dt: false
    seed: {fit_seed}
{les_block}
learning_rate:
  type: exp
  decay_steps: {decay_steps}
  decay_rate: {decay_rate}
  start_lr: {start_lr}
  stop_lr: {stop_lr}

loss:
  type: ener
  start_pref_e: {pref_e}
  limit_pref_e: {pref_e}
  start_pref_f: {pref_f}
  limit_pref_f: {pref_f}
  start_pref_v: 0.0
  limit_pref_v: 0.0

training:
  training_data:
    systems: [{train}]
    batch_size: 1
  validation_data:
    systems: [{valid}]
    batch_size: 1
    numb_btch: {numb_btch}
  numb_steps: {numb_steps}
  warmup_steps: {warmup}
  gradient_max_norm: {grad_max_norm}
  seed: {train_seed}
  disp_file: "lcurve.out"
  disp_freq: {disp_freq}
  save_freq: {save_freq}
  save_ckpt: "model.ckpt"
  max_ckpt_keep: {max_ckpt_keep}
"""

LES_COMMENT = """\
les_params: cace fit-interface-mp0's long-range recipe - charge head [24, 12],
Ewald dl 2 sigma 1, CombinePotential mixing weight 0.02 -> lr_weight, and
remove_self_interaction true as the author's own checkpoints have it. cace's
charge head is bias=False with n_out=4 (four independent Ewald channels); les's
is a single scalar charge with a bias. See the module docstring.
"""


def natoms_of(data_dir, d):
    """Atoms per frame, read from the data rather than remembered.

    It decides `pref_e` (see the module docstring), so a wrong value silently
    weights the energy loss wrongly: `pref_e = natoms * w_E` is only cace's
    ``w_E * MSE(E)`` when natoms is the real atom count, because deepmd divides
    the energy term by the atom count.
    """
    with open(os.path.join(data_dir, d, 'type.raw')) as fh:
        return len(fh.read().split())


def read_sel(data_dir):
    """The se_a `sel` the converter computed for the chosen rcut."""
    with open(os.path.join(data_dir, 'sel.txt')) as fh:
        return [int(x) for x in fh.read().split()]


def cace_lr(epoch_in_task, step_size):
    """cace's StepLR(step_size, 0.5) value at an epoch of the current task."""
    return BASE_LR * DECAY_RATE ** (epoch_in_task // step_size)


def build_segments(step_size, phases, steps_per_epoch, natoms,
                   warmup=WARMUP, decay_steps=None):
    """Expand cace's phase list into the chained deepmd blocks.

    The first phase's blocks are fresh tasks, so each one restarts the epoch
    counter and therefore the LR; later phases continue the last fresh task's
    counter. The counter is what cace's StepLR sees, and the whole point of
    tracking it here is that a block's ``start_lr`` is cace's value at its first
    epoch - not an interpolated approximation of it.

    ``decay_steps`` defaults to ``steps_per_epoch * step_size`` (cace's 20-epoch
    halving period); the smoke mode overrides it so nothing trips deepmd's
    ``decay_steps >= stop_steps`` substitution.
    """
    if decay_steps is None:
        decay_steps = steps_per_epoch * step_size
    segments = []
    task_epoch = 0
    offset = 0
    for phase, (w_e, n_rep, epochs) in enumerate(phases):
        for rep in range(n_rep):
            fresh = phase == 0
            if fresh:
                task_epoch = 0
            num_steps = epochs * steps_per_epoch
            start_lr = cace_lr(task_epoch, step_size)
            segments.append({
                'n': len(segments) + 1,
                'phase': phase,
                'rep': rep,
                'fresh_task': fresh,
                'w_e': w_e,
                'epochs': epochs,
                'num_steps': num_steps,
                'steps_per_epoch': steps_per_epoch,
                'global_offset': offset,
                'task_epoch': task_epoch,
                'start_lr': start_lr,
                'stop_lr': start_lr * DECAY_RATE ** (num_steps // max(1, decay_steps)),
                'decay_steps': decay_steps,
                'warmup': warmup if fresh else 0,
                'natoms': natoms,
                'pref_e': natoms * w_e,
                'pref_f': FORCE_PREF,
            })
            offset += num_steps
            task_epoch += epochs
    return segments


def les_block(les):
    """The les_params mapping, indented under model:, with booleans lowercase."""
    if les is None:
        return ''
    out = [f'  # {line}' for line in LES_COMMENT.split('\n') if line]
    out.append('  les_params:')
    for k, v in les.items():
        if isinstance(v, bool):
            v = str(v).lower()
        elif isinstance(v, list):
            v = '[' + ', '.join(str(x) for x in v) + ']'
        out.append(f'    {k}: {v}')
    out.append('    verbose: true')
    out.append('    log_freq: 1000')
    return '\n'.join(out) + '\n'


def cfg_num(v):
    """A float the way a hand-written config wants it: plain decimal where the value
    is one, shortest round-trip otherwise."""
    return f'{float(v):.12g}'


def yaml_list(xs):
    return '[' + ', '.join(str(x) for x in xs) + ']'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-root', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--replicate-tags', default=''.join(r['tag'] for r in REPLICATES))
    ap.add_argument('--smoke', action='store_true',
                    help='one 1-epoch block per phase, for exercising the chain '
                         'only - NOT the cace schedule')
    args = ap.parse_args()

    sel = read_sel(args.data)
    step_size, steps_per_epoch = 20, STEPS_PER_EPOCH
    disp_freq, save_freq, numb_btch = DISP_FREQ, SAVE_FREQ, VALID_BATCHES
    warmup, decay_steps = WARMUP, None
    if args.smoke:
        # One 1-"epoch" block per phase at SMOKE_STEPS steps, a decay period of
        # one step and a 2-step warmup, so every block is minutes not hours and
        # `decay_steps < numb_steps - warmup` keeps deepmd's `decay_steps >=
        # stop_steps` substitution from firing. Block LR, and so the whole point
        # of the file, is NOT cace's in this mode: it only exercises the chain.
        step_size, steps_per_epoch = 1, SMOKE_STEPS
        warmup, decay_steps = SMOKE_WARMUP, SMOKE_DECAY
        disp_freq, save_freq, numb_btch = 3, SMOKE_STEPS, 2

    wanted_arms = args.arms.split(',')
    unknown = [a for a in wanted_arms if a not in ARMS]
    if unknown:
        raise SystemExit(f'unknown arm(s) {unknown}; known: {list(ARMS)}')
    wanted_reps = [r for r in REPLICATES if r['tag'] in args.replicate_tags]
    if not wanted_reps:
        raise SystemExit(f'no replicate matches {args.replicate_tags!r}')
    natoms = natoms_of(args.data, TRAIN_DIR)
    train = [os.path.join(args.data, TRAIN_DIR)]
    valid = [os.path.join(args.data, VALID_DIR)]

    # systems paths are resolved against the process CWD, which the chain sets to
    # the block directory, so write them relative to a block dir.
    def rel(paths, blockdir):
        return ', '.join(f'"{os.path.relpath(p, blockdir)}"' for p in paths)

    for arm in wanted_arms:
        spec = ARMS[arm]
        phases = spec['phases']
        if args.smoke:
            phases = [(w, n, 1) for w, n, _ in phases]
        segments = build_segments(step_size, phases, steps_per_epoch, natoms,
                                  warmup=warmup, decay_steps=decay_steps)
        for rep in wanted_reps:
            tag = f'{arm}_s{rep["tag"]}'
            armdir = os.path.join(args.out_root, tag)
            os.makedirs(armdir, exist_ok=True)
            for seg in segments:
                blockdir = os.path.join(armdir, f's{seg["n"]}')
                os.makedirs(blockdir, exist_ok=True)
                body = TEMPLATE.format(
                    arm=arm, tag=rep['tag'], doc=spec['doc'],
                    cace_arm='mp0' if spec['les'] else 'mp0-sr',
                    n=seg['n'], nseg=len(segments), prev=seg['n'] - 1,
                    chain_note=('first block, trained from scratch' if seg['n'] == 1
                                else f'after block {seg["n"] - 1}'),
                    mtype=spec['mtype'],
                    desc_seed=rep['desc_seed'], fit_seed=rep['fit_seed'],
                    train_seed=rep['train_seed'],
                    sel=yaml_list(sel),
                    rcut_smth=f'{RCUT_SMTH:.2f}', rcut=f'{RCUT:.2f}',
                    desc_neuron=yaml_list(DESC_NEURON),
                    fit_neuron=yaml_list(FIT_NEURON),
                    les_block=les_block(spec['les']),
                    w_e=seg['w_e'], epochs=seg['epochs'],
                    numb_steps=seg['num_steps'],
                    start_lr=cfg_num(seg['start_lr']),
                    stop_lr=cfg_num(seg['stop_lr']),
                    decay_steps=seg['decay_steps'],
                    decay_rate=DECAY_RATE,
                    pref_e=cfg_num(seg['pref_e']),
                    pref_f=cfg_num(seg['pref_f']),
                    warmup=seg['warmup'],
                    grad_max_norm=GRAD_MAX_NORM,
                    disp_freq=disp_freq, save_freq=save_freq,
                    numb_btch=numb_btch,
                    max_ckpt_keep=MAX_CKPT_KEEP,
                    train=rel(train, blockdir), valid=rel(valid, blockdir),
                )
                with open(os.path.join(blockdir, 'input.yaml'), 'w') as fh:
                    fh.write(body)

            record = {
                'arm': arm,
                'replicate': rep['tag'],
                'smoke': bool(args.smoke),
                'chain': {
                    'flag': '--init-model',
                    'step_counter': 'reset to 0 per block',
                    'optimizer': 'fresh Adam per block',
                    'why': 'restart keeps Adam but makes the LR argument the '
                           'accumulated global step, which is 0.01*0.5**10 on s6 '
                           'instead of cace\'s 2.5e-3; init-model with a per-block '
                           'start_lr reproduces cace\'s task-local StepLR. '
                           'See the module docstring.',
                },
                'cace_recipe': {
                    'source': 'fit-water-interface/fit-interface-mp0[-sr]/fit-cace-nnp.py',
                    'steps_per_epoch': STEPS_PER_EPOCH,
                    'phases': phases,
                    'steplr': {'step_size_epochs': 20, 'gamma': DECAY_RATE},
                    'optimizer': {'type': 'Adam', 'lr': BASE_LR,
                                  'betas': [0.99, 0.999],
                                  'note': 'deepmd Adam betas are (0.9, 0.999); '
                                          'not settable'},
                    'warmup_steps': WARMUP,
                    'warmup_note': 'cace warmup_steps=5 counts EPOCHS '
                                   '(5 * 450 = 2250 optimizer steps)',
                    'max_grad_norm': GRAD_MAX_NORM,
                    'force_weight': FORCE_PREF,
                    'energy_weight_to_pref_e': f'natoms={natoms} x w_E, undoing '
                                               'deepmd\'s per-atom energy loss',
                    'cace_cutoff': 5.5,
                    'deepmd_rcut': RCUT,
                    'seeds': {k: rep[k] for k in ('desc_seed', 'fit_seed',
                                                  'train_seed')},
                },
                'segments': segments,
                'total_steps': sum(s['num_steps'] for s in segments),
            }
            with open(os.path.join(armdir, 'segments.json'), 'w') as fh:
                json.dump(record, fh, indent=2)

            with open(os.path.join(armdir, 'chain.sh'), 'w') as fh:
                fh.write('#!/bin/bash\n'
                         '# generated by gen_inputs_water.py; use run_water_chain.py '
                         'to run it with timing\n'
                         '# the WSL2 host deadlocks under the default caching '
                         'allocator; harmless elsewhere\n'
                         'set -euo pipefail\n'
                         'cd "$(dirname "$0")"\n'
                         'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"\n')
                for i, seg in enumerate(segments):
                    n = seg['n']
                    init = (f' --init-model ../s{n - 1}/model.ckpt-'
                            f'{segments[i - 1]["num_steps"]}.pt'
                            if n > 1 else '')
                    fh.write(f'(cd s{n} && dp --pt train input.yaml{init})\n')
            print(f'wrote {os.path.relpath(armdir, HERE)}: '
                  f'{len(segments)} blocks, '
                  f'{record["total_steps"]} steps total')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
