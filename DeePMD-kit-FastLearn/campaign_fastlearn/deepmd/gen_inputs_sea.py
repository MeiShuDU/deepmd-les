"""Generate the two cace-schedule deepmd arms - a deepmd / deepmd-les pair whose
short-range network and whole training strategy are cace's own, so that a
comparison against cace-sea-sr / cace-sea-lr prices the LES *implementation*
rather than the short-range net or the learning-rate strategy.

That is the confound FIG 9 measured: the deepmd family's SR net is a
[240,240,240] resnet head trained at lr 1e-3 with a ramped prefactor, cace's is a
[32,16] silu head trained at 1e-2 with stepped per-phase weights. This pair
removes both, so a remaining gap is the LES code and nothing else.

What is matched, and to what
----------------------------
descriptor  ``sea_seam.SE_A``: rcut 5.5, rcut_smth 0.50, sel [39, 73],
            neuron [25, 50, 100], axis_neuron 16, resnet_dt false, seed 1,
            type_one_side false (deepmd's default). Identical to the campaign's
            other deepmd arms, so the descriptor is not a new variable here.
SR head     cace ``Atomwise(n_in=1600, n_layers=3, n_hidden=[32, 16],
            activation silu, use_batchnorm=False, add_linear_nn=True)`` ->
            ``fitting_net {neuron: [32, 16], activation_function: silu,
            resnet_dt: false}``. The parallel ``Dense(1600 -> 1)`` that
            ``add_linear_nn`` adds has no analogue in deepmd's fitting net and is
            dropped: 51,777 parameters against cace's 53,378.
optimiser   Adam at lr 1e-2, ``warmup_steps: 5``, ``gradient_max_norm: 10``.
            cace also sets betas (0.99, 0.999); deepmd hardcodes (0.9, 0.999) and
            exposes no knob, so that one is looser.
schedule    cace ``ENV_PHASES`` = [(0.1, 5, 40), (1.0, 1, 100), (10.0, 1, 100),
            (1000.0, 1, 100)]: five fresh 40-epoch tasks at E weight 0.1, each one
            rebuilding the optimiser and the LR scheduler, then phases 1-3 sharing
            the fifth task. Reproduced as eight chained deepmd runs, one per block,
            each in its own directory.
LR          cace ``StepLR(step_size=20, gamma=0.5)`` on a per-task epoch counter.
            deepmd's ``exp`` schedule is itself a staircase,
            ``lr(s) = start_lr * decay_rate ** ((s - warmup) // decay_steps)``, so
            ``decay_steps = 20 epochs * 160 steps = 3200`` with
            ``decay_rate = 0.5`` reproduces cace's halving exactly at epoch
            granularity. Each block then declares the ``start_lr`` cace's counter
            holds at that block's first epoch: 1e-2 for each of the five phase-0
            tasks, then 2.5e-3, 7.8125e-5, 2.44140625e-6 continuing the fifth.
            Chain with ``--init-model``, not ``--restart``: init-model resets the
            step counter (so each block's schedule really starts at local step 0)
            and rebuilds the optimiser, which is exactly what cace does at each of
            its five fresh phase-0 tasks. It is also what cace does NOT do at the
            two phase boundaries after that, where it keeps the optimiser; those
            Adam moment resets are the one place this chain is looser than cace,
            and they cost a few dozen steps of re-adaptation in 80,000.
E weight    cace's energy loss is ``w_E * MSE(E_frame)``; deepmd's is
            ``pref_e * MSE(E_frame) / natoms`` (``pt/loss/ener.py``), so a flat
            ``pref_e = natoms * w_E`` gives the energy:force ratio cace trains under.
            natoms is 192 per frame in this data and is read from ``type.raw``, not
            assumed: at 96 the energy weight would be half of cace's and the run
            would not be cace's strategy. Writing cace's ``w_E`` verbatim would leave
            deepmd's energy 192x weaker relative to force.
virial      ``pref_v = 0``: cace trains on energy and force only.
long range  the ``les_params`` block is the cace `fit-water-timing` mapping the
            campaign already uses (sigma 1.0, dl 2.0, remove_self_interaction
            false, output_scaling_factor 1.0, lr_weight 1.0), with two knobs the
            earlier arms left at their defaults and this pair has to set because
            they are part of cace's own model:
              n_hidden [24, 12], n_layers 3, add_linear_nn true  = cace's charge
                head, which the earlier arms ran at les's [32, 16] default;
              initial_guess omitted  = cace's charge head starts from its own
                random init, so anchoring it at the SPC/E table would not be the
                same model.
            Unchanged and unavoidable: cace's charge head has bias=False while the
            les Atomwise has a bias, and the Ewald kernel's constant prefactor
            (cace norm_factor 1.0 against les's hardcoded 90.4756) is not reachable
            from les_params. Both are absorbed by the free charge net, and both are
            documented in gen_inputs.py.

Layout
------
Each arm/replicate is one directory holding eight block directories::

    runs/deepmd-cace-sched_sA/s1/input.yaml   s1/model.ckpt-*.pt   ...
    runs/deepmd-cace-sched_sA/s2/input.yaml   ...
    ...
    runs/deepmd-cace-sched_sA/segments.json   (the schedule, for the evaluator)
    runs/deepmd-cace-sched_sA/chain.sh

One directory per block because deepmd's step counter is per-run: it is what lets
each block keep its own lcurve.out, its own les.log and its own checkpoints, and
it is what makes the local checkpoint step convertible to a global one
(``segments.json`` carries each block's ``global_offset``).

Usage:
    python gen_inputs_sea.py
    python gen_inputs_sea.py --smoke          # 1-epoch blocks, for plumbing only
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
DATA = os.path.join(CAMPAIGN, '..', 'data')

# cace's own recipe, in cace's units.
STEPS_PER_EPOCH = 160       # 320 training frames / batch_size 2
DECAY_STEPS = 3200          # 20 epochs: cace's StepLR step_size
DECAY_RATE = 0.5            # cace's StepLR gamma
BASE_LR = 1e-2
FORCE_PREF = 1000.0         # cace's FORCE_WEIGHT
WARMUP = 5                  # cace's warmup_steps, applied per fresh task
GRAD_MAX_NORM = 10.0        # cace's max_grad_norm
DISP_FREQ = 1000
SAVE_FREQ = 3200            # one checkpoint per LR step, so every block end lands
MAX_CKPT_KEEP = 30

# (E weight, number of blocks, epochs per block) - cace's ENV_PHASES verbatim.
PHASES = [(0.1, 5, 40), (1.0, 1, 100), (10.0, 1, 100), (1000.0, 1, 100)]

REPLICATES = [
    {'tag': 'A', 'desc_seed': 1, 'fit_seed': 1, 'train_seed': 10},
    {'tag': 'B', 'desc_seed': 2, 'fit_seed': 2, 'train_seed': 20},
]

TRAIN_DIRS = ['data_0', 'data_1', 'data_2']
VALID_DIR = 'data_3'

# Shared by both arms; see the module docstring for the cace provenance.
LES_COMMON = {
    'sigma': 1.0,
    'dl': 2.0,
    'remove_self_interaction': False,
    'output_scaling_factor': 1.0,
    'lr_weight': 1.0,
}

ARMS = {
    'deepmd-cace-sched': {
        'mtype': 'standard',
        'les': None,
        'doc': 'short-range only control, on cace\'s SR net and cace\'s schedule',
    },
    'deepmd-les-cace-sched': {
        'mtype': 'hybrid_ener',
        'les': dict(LES_COMMON, local_charge=True, n_hidden=[24, 12], n_layers=3,
                    add_linear_nn=True),
        'doc': 'learned per-atom charge + Ewald, on cace\'s SR net and schedule; '
               'the charge head is cace\'s [24, 12] and starts unanchored, as '
               'cace\'s does',
    },
}

TEMPLATE = """\
# {arm}, replicate {tag}, block {n}/{nseg} - generated by
# campaign_fastlearn/deepmd/gen_inputs_sea.py
# {doc}
#
# SR net:  cace Atomwise [32, 16] silu, without its parallel linear skip
# schedule: cace ENV_PHASES, block {n} = E weight {w_e} for {epochs} epochs
# block LR: start_lr {start_lr} with decay_steps {decay_steps} / decay_rate {decay_rate},
#           i.e. cace's StepLR(20, 0.5) on this task's epoch counter
# chain:   run after block {prev}; --init-model rebuilds the optimiser, which is
#          what cace does at every phase-0 block
model:
  type: {mtype}
  type_map: ["O", "H"]
  descriptor:
    type: se_a
    sel: [39, 73]
    rcut_smth: 0.50
    rcut: 5.50
    neuron: [25, 50, 100]
    axis_neuron: 16
    resnet_dt: false
    seed: {desc_seed}
  fitting_net:
    neuron: [32, 16]
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
    batch_size: 2
  validation_data:
    systems: [{valid}]
    batch_size: 4
    numb_btch: 20
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
LES block: cace fit-water-timing, knob for knob, plus the two cace knobs the
earlier deepmd arms left at les defaults (n_hidden/n_layers = the [24,12] charge
head; no initial_guess, matching cace's unanchored init). The Ewald prefactor
still differs (cace 1.0 vs les 90.4756) and is absorbed by the free charge net;
see gen_inputs.py.
"""


def natoms_of(data_dir, dirs):
    """Atoms per frame, read from the data rather than remembered.

    It decides `pref_e` (see the module docstring), so a wrong value silently weights
    the energy loss wrongly: `pref_e = natoms * w_E` is only cace's ``w_E * MSE(E)``
    when natoms is the real atom count, because deepmd divides the energy term by
    ``atype.shape[-1]`` (`pt/train/wrapper.py:192`). It is 192 here, not the 96 an
    earlier note claimed.
    """
    counts = set()
    for d in dirs:
        with open(os.path.join(data_dir, d, 'type.raw')) as fh:
            counts.add(len(fh.read().split()))
    if len(counts) != 1:
        raise SystemExit(f'systems disagree on the atom count: {sorted(counts)}')
    return counts.pop()


def cace_lr(epoch_in_task, step_size):
    """cace's StepLR(step_size, 0.5) value at an epoch of the current task."""
    return BASE_LR * DECAY_RATE ** (epoch_in_task // step_size)


def build_segments(step_size, phases, steps_per_epoch, natoms):
    """Expand cace's phase list into the chained deepmd blocks.

    Phase 0's blocks are fresh tasks, so each one restarts the epoch counter and
    therefore the LR; phases 1-3 continue the last phase-0 task's counter. The
    counter is what cace's StepLR sees.
    """
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
                'stop_lr': start_lr * DECAY_RATE ** (num_steps // max(1, steps_per_epoch * step_size)),
                'decay_steps': steps_per_epoch * step_size,
                'warmup': WARMUP if fresh else 0,
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
    out.append('    log_freq: 100')
    return '\n'.join(out) + '\n'


def cfg_num(v):
    """A float the way a hand-written config wants it: plain decimal where the value is
    one (96 * 0.1 is 9.600000000000001 in float), shortest round-trip otherwise."""
    s = f'{float(v):.12g}'
    return s


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

    phases, step_size, steps_per_epoch = PHASES, 20, STEPS_PER_EPOCH
    disp_freq, save_freq = DISP_FREQ, SAVE_FREQ
    if args.smoke:
        # Three 1-epoch blocks per phase and a decay period of one epoch, so
        # `decay_steps` stays below `numb_steps - warmup` and nothing trips
        # deepmd's `decay_steps >= stop_steps` substitution. Block LR, and so the
        # whole point of the file, is NOT cace's in this mode.
        phases = [(w, n, 3) for w, n, _ in PHASES]
        step_size, steps_per_epoch = 1, 160
        disp_freq, save_freq = 10, 160

    wanted_arms = args.arms.split(',')
    unknown = [a for a in wanted_arms if a not in ARMS]
    if unknown:
        raise SystemExit(f'unknown arm(s) {unknown}; known: {list(ARMS)}')
    wanted_reps = [r for r in REPLICATES if r['tag'] in args.replicate_tags]
    if not wanted_reps:
        raise SystemExit(f'no replicate matches {args.replicate_tags!r}')
    natoms = natoms_of(args.data, TRAIN_DIRS)

    segments = build_segments(step_size, phases, steps_per_epoch, natoms)
    train = [os.path.join(args.data, d) for d in TRAIN_DIRS]
    valid = [os.path.join(args.data, VALID_DIR)]

    # systems paths are resolved against the process CWD, which the chain sets to
    # the block directory, so write them relative to a block dir.
    def rel(paths, blockdir):
        return ', '.join(f'"{os.path.relpath(p, blockdir)}"' for p in paths)

    for arm in wanted_arms:
        spec = ARMS[arm]
        for rep in wanted_reps:
            tag = f'{arm}_s{rep["tag"]}'
            armdir = os.path.join(args.out_root, tag)
            os.makedirs(armdir, exist_ok=True)
            for seg in segments:
                blockdir = os.path.join(armdir, f's{seg["n"]}')
                os.makedirs(blockdir, exist_ok=True)
                body = TEMPLATE.format(
                    arm=arm, tag=rep['tag'], doc=spec['doc'],
                    n=seg['n'], nseg=len(segments), prev=seg['n'] - 1,
                    mtype=spec['mtype'],
                    desc_seed=rep['desc_seed'], fit_seed=rep['fit_seed'],
                    train_seed=rep['train_seed'],
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
                    max_ckpt_keep=MAX_CKPT_KEEP,
                    train=rel(train, blockdir), valid=rel(valid, blockdir),
                )
                with open(os.path.join(blockdir, 'input.yaml'), 'w') as fh:
                    fh.write(body)

            record = {
                'arm': arm,
                'replicate': rep['tag'],
                'smoke': bool(args.smoke),
                'cace_recipe': {
                    'steps_per_epoch': STEPS_PER_EPOCH,
                    'phases': PHASES,
                    'steplr': {'step_size_epochs': 20, 'gamma': DECAY_RATE},
                    'optimizer': {'type': 'Adam', 'lr': BASE_LR,
                                  'betas': [0.99, 0.999],
                                  'note': 'deepmd Adam betas are (0.9, 0.999); '
                                          'not settable'},
                    'warmup_steps': WARMUP, 'max_grad_norm': GRAD_MAX_NORM,
                    'force_weight': FORCE_PREF,
                    'energy_weight_to_pref_e': f'natoms={natoms} x w_E, undoing '
                                               'deepmd\'s per-atom energy loss',
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
                         '# generated by gen_inputs_sea.py; use run_sea_chain.py '
                         'to run it with timing\n'
                         'set -euo pipefail\n'
                         'cd "$(dirname "$0")"\n')
                for seg in segments:
                    n = seg['n']
                    init = (f' --init-model ../s{n - 1}/model.ckpt.pt'
                            if n > 1 else '')
                    fh.write(f'(cd s{n} && dp --pt train input.yaml{init})\n')
            print(f'wrote {os.path.relpath(armdir, HERE)}: '
                  f'{len(segments)} blocks, '
                  f'{record["total_steps"]} steps total')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
