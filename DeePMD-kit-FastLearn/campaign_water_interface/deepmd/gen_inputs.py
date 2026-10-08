"""Generate the two deepmd arms for the water-interface campaign - a deepmd /
deepmd-les pair that keeps cace's *training strategy* while using deepmd's own
descriptor and fitting net.

What is being compared
---------------------
Four arms, two replicates each, eight pod runs:

    cace-lr   (fit-interface-mp0)      cace's own net, cace's schedule, LR on
    cace-sr   (fit-interface-mp0-sr)   cace's own net, cace's schedule, SR only
    deepmd-les  <- this file          deepmd se_a + fitted charge/Ewald, cace's schedule
    deepmd      <- this file          deepmd se_a, SR only, cace's schedule

`deepmd-les-cace-sched` <- this file is a fifth, added later: the same
hybrid_ener long range as `deepmd-les`, but on the *sibling* arm's own model
rather than on deepmd's [240, 240, 240] fitting net. It exists because the pair
above cannot say whether `deepmd-les`'s distance from `sea-lr_s*` is the LES
implementation or the SR net it was bolted onto: `deepmd-les` runs deepmd's own
48k-parameter fitter where `sea-lr_s*` runs cace's [32, 16] one, and two things
differ at once. This arm takes `sea-lr_s*`'s descriptor geometry (se_a, rcut 5.5,
sel [43, 84]) and its cace SR net, so it differs from `sea-lr_s*` in the long-range
implementation and in the two deviations no config key can reach (`add_linear_nn`
above, and the charge head's `bias`/`n_out`), and in nothing else.
`campaign_fastlearn`'s `deepmd-les-cace-sched` is the same idea against its own
sibling pair.

So the pair this file writes is NOT campaign_fastlearn's `deepmd-cace-sched`
pair: there the short-range net was deliberately rebuilt as cace's own
Atomwise([32,16], silu) to isolate the LES *implementation*. Here the descriptor
is se_a (rcut 6.00, rcut_smth 0.50, sel [43, 84], neuron [25,50,100],
axis_neuron 16) and the fitting net is [240,240,240] as specified, i.e. deepmd's
own architecture held constant across the two deepmd arms, so the deepmd pair
prices the long-range term inside deepmd while the cace pair prices it inside
cace, and the four together separate "which code" from "which schedule".

What is matched, and to what
----------------------------
schedule    cace `fit-interface-mp0`: five fresh 40-epoch tasks at E weight 0.01,
            each rebuilding the optimiser and the LR scheduler, then phases
            sharing the fifth task: 100 epochs at w_E 1, 100 at 10, 100 at 1000.
            cace `fit-interface-mp0-sr` is identical but stops after w_E 10 (the
            mp0-sr script's fourth block carried the same w_E 1000 as mp0's - the
            arm is the *short-range* one, so its schedule is the 400-epoch subset).
            Reproduced as chained deepmd runs, one per block, each in its own
            directory: 8 blocks / 225,000 steps for the LES arm,
            7 blocks / 180,000 steps for the SR arm.
LR          cace `StepLR(step_size=20, gamma=0.5)` driven by a per-task epoch
            counter, `lr = 1e-2 * 0.5 ** (epoch_in_task // 20)`. deepmd's `exp`
            schedule is the same staircase - `pt/utils/learning_rate.py`:
            `lr(s) = start_lr * decay_rate ** (s // decay_steps)` - so
            `decay_steps = 20 epochs * 450 steps = 9000` with `decay_rate = 0.5`
            reproduces the halving exactly, at step granularity, on a fresh task.
            Each block declares the `start_lr` cace's counter holds at that
            block's first epoch: 1e-2 for each of the five phase-0 tasks, then
            2.5e-3, 7.8125e-5 and 2.44140625e-6 continuing the fifth.
            Chain with `--init-model`, not `--restart`: init-model resets the step
            counter (so the block's schedule really starts at local step 0) and
            rebuilds the optimiser, which is exactly what cace does at each of its
            five fresh phase-0 tasks. It is also what cace does NOT do at the two
            phase boundaries after that, where it keeps the optimiser; those Adam
            moment resets are the one place this chain is looser than cace.
LR phase    deepmd's scheduler origin is the end of its warmup,
            `warm_up_linear` returns `lr_exp.value(step - warmup_steps) / start_lr`,
            while cace's StepLR counts from the task's own epoch 0. A fresh block
            therefore halves 5 epochs (one warmup) later than cace's does; the
            continued blocks, which carry `warmup_steps: 0`, are exact. The
            deviation is 5 of 40 epochs in phase 0 only.
optimiser   Adam at lr 1e-2, `gradient_max_norm: 10`, cace's `warmup_steps: 5`.
            cace's warmup counts EPOCHS (`global_step` advances once per epoch),
            so it is 5 * 450 = 2250 deepmd steps, not 5. cace also sets betas
            (0.99, 0.999); deepmd hardcodes (0.9, 0.999) with no knob, so that
            one is looser. `opt_type` is not reachable either (Adam is hardcoded
            default in `pt/train/training.py:159`).
E weight    cace's energy loss is `w_E * MSE(E_frame)` on the COMBINED energy:
            `CombinePotential.forward` overwrites `CACE_energy` with
            `E_sr + 0.02 * E_ewald` before the loss sees it, and likewise for the
            forces, so cace supervises the total and the deepmd pair does too.
            deepmd's loss is `pref_e * MSE(E_frame) / natoms`
            (`pt/loss/ener.py`), so a flat `pref_e = natoms * w_E` gives the
            energy:force ratio cace trains under. natoms is 1566 per frame in this
            data and is read from `type.raw`, not assumed: writing cace's `w_E`
            verbatim would leave deepmd's energy 1566x weaker relative to force.
force       cace's `FORCE_WEIGHT = 1000`, on the combined force; `pref_f = 1000`
            and `pref_v = 0` (cace trains no virial).
validation  cace validates once per epoch over the whole valid split with
            `batch_size: 1`. `disp_freq: 450` (one epoch) with
            `numb_btch: 50, batch_size: 1` covers the same 50 frames.
split       cace `valid_fraction=0.1, seed=1`, reproduced upstream by
            `prep_water_interface.py`, which calls cace's own
            `random_train_valid_split`; the ledger is `split_manifest.txt`.
long range  the `les_params` block is cace fit-interface-mp0's Ewald mapping:
            `sigma 1, dl 2`, `remove_self_interaction` cace's default true,
            `output_scaling_factor` 1.0 (les would otherwise scale the charge
            head by 0.1), `n_hidden [24, 12]` with `n_layers 3` and
            `add_linear_nn` for cace's charge head, and `lr_weight 0.02` for
            cace's pot2 weight. `initial_guess` is omitted: cace's charge head
            starts from its own random init, so anchoring it at the SPC/E table
            would not be the same model.
            Unchanged and unavoidable: cace's charge head has `bias=False` while
            les's Atomwise has a bias; cace emits a 4-channel charge
            (`n_out=4`, summed over channels in Ewald) and les has no `n_out`
            key; and the Ewald kernel's constant prefactor (cace
            `norm_factor=1.0` against les's hardcoded 90.4756) is not reachable
            from les_params. All three are absorbed by the free charge net.

Layout
------
Each arm/replicate is one directory holding its block directories::

    runs/deepmd-les_sA/s1/input.yaml   s1/model.ckpt-*.pt   ...
    runs/deepmd-les_sA/segments.json   (the schedule, for the evaluator)
    runs/deepmd-les_sA/chain.sh

One directory per block because deepmd's step counter is per-run: it is what lets
each block keep its own lcurve.out, its own les.log and its own checkpoints, and
it is what makes the local checkpoint step convertible to a global one
(`segments.json` carries each block's `global_offset`).

Usage:
    python gen_inputs.py
    python gen_inputs.py --smoke          # tiny blocks, for plumbing only
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
DATA = os.path.join(CAMPAIGN, 'data', 'water-interface')

# cace's own recipe, in cace's units.
STEPS_PER_EPOCH = 450      # 450 training frames / batch_size 1
DECAY_STEPS = 9000         # 20 epochs: cace's StepLR step_size
DECAY_RATE = 0.5           # cace's StepLR gamma
BASE_LR = 1e-2
FORCE_PREF = 1000.0        # cace's force loss weight
WARMUP = 2250              # cace's warmup_steps 5 * 450 steps
GRAD_MAX_NORM = 10.0       # cace's max_grad_norm
DISP_FREQ = 450            # one epoch, cace's validation cadence
# deepmd's default is 10 frames, which it concatenates and hands to the
# neighbour list in ONE call. At 1566 atoms and rcut 6.0 each frame's
# (nloc, nall, 3) displacement tensor is 0.49 GiB (nall 14094, 9 periodic
# images), so 10 frames is ~10 GiB and both local smoke runs died in
# `pt/utils/nlist.py:117` - the LES arm with SIGSEGV, the SR arm with
# `CUDA driver error: out of memory`. 2 frames is ~2.3 GiB peak and keeps the
# normalisation sample above a single frame; the frames are drawn per block, so
# the descriptor's mean/std differ slightly between blocks, which is a declared
# deviation from a fixed normalisation.
DATA_STAT_NBATCH = 2
SAVE_FREQ = 4500           # 10 epochs; must divide every block's length
MAX_CKPT_KEEP = 40

# (E weight, number of blocks, epochs per block), cace verbatim.
PHASES_LR = [(0.01, 5, 40), (1.0, 1, 100), (10.0, 1, 100), (1000.0, 1, 100)]
PHASES_SR = [(0.01, 5, 40), (1.0, 1, 100), (10.0, 1, 100)]

REPLICATES = [
    {'tag': 'A', 'desc_seed': 1, 'fit_seed': 1, 'train_seed': 10},
    {'tag': 'B', 'desc_seed': 2, 'fit_seed': 2, 'train_seed': 20},
]

TRAIN_DIRS = ['train']
VALID_DIR = 'valid'

DESCRIPTOR = {
    'type': 'se_a',
    'sel': [43, 84],        # measured at rcut 6.0 by prep_water_interface.py
    'rcut_smth': 0.50,
    'rcut': 6.00,
    'neuron': [25, 50, 100],
    'axis_neuron': 16,
    'resnet_dt': False,
}
FITTING = {
    'neuron': [240, 240, 240],
}

# The sibling recipe: what `sea-lr_s*` itself runs, so that a deepmd arm built on
# it differs from `sea-lr_s*` in *which code implements the long-range term* and
# in nothing else that the config can express.
#
# Descriptor: sea-lr_sA/input.json is se_a at rcut 5.5, rcut_smth 0.5,
# sel [43, 84], neuron [25, 50, 100], axis_neuron 16, resnet_dt false,
# type_one_side false (deepmd's default, so not written). The `sel` is the
# rcut-6.0 measurement, so at 5.5 it is a loose-but-valid cap - which is what
# sea-lr_s* itself declares, and copying it is the point.
# Fitting: cace's Atomwise([32, 16], silu). cace's `build_mlp` ignores
# `n_layers` whenever `n_hidden` is a list (`n_neurons = [n_in] + n_hidden +
# [n_out]`, cace/modules/blocks.py:54-58), so `n_layers 3` with
# `n_hidden [32, 16]` is a two-hidden-layer MLP and deepmd's `neuron: [32, 16]`
# is the same net. This is the mapping campaign_fastlearn's
# `deepmd-les-cace-sched` uses to put cace's SR net inside deepmd. The two arms of
# this pair therefore share the SR net that the earlier `deepmd-les_s*` arm did
# not: it ran deepmd's own [240, 240, 240].
# Not reachable: cace's `add_linear_nn` skip, a parallel `Dense(n_in, n_out)`
# added to the MLP output. deepmd's `ener` fitting has no equivalent key, so the
# deepmd arm is cace's net minus that one linear term - the same declared
# deviation campaign_fastlearn's arm carries.
SIBLING_DESCRIPTOR = dict(DESCRIPTOR, rcut=5.50)
SIBLING_FITTING = {
    'neuron': [32, 16],
    'activation_function': 'silu',
    'resnet_dt': False,
}

# cace fit-interface-mp0's long-range block; see the module docstring.
LES_COMMON = {
    'local_charge': True,
    'n_hidden': [24, 12],
    'n_layers': 3,
    'add_linear_nn': True,
    'sigma': 1.0,
    'dl': 2.0,
    'remove_self_interaction': True,
    'output_scaling_factor': 1.0,
    'lr_weight': 0.02,
}

# sea-lr_s* declares `remove_self_interaction: false` in its `long_range.ewald`
# block, so the sibling arm takes false too and the pair agrees knob for knob.
# The self term shifts the optimal charge magnitude, so it is not a detail to
# leave differing between two arms that are compared to each other; the earlier
# `deepmd-les_s*` used true and is the reason this is written out here.
LES_SIBLING = dict(LES_COMMON, remove_self_interaction=False)

LES_COMMENT_SIBLING = """\
LES block: sea-lr_s*'s own long range, knob for knob - sigma 1, dl 2,
self-interaction KEPT (`remove_self_interaction: false`, which is what
sea-lr_sA/input.json declares under long_range.ewald, and not the les default the
`deepmd-les_s*` arm used), charge head [24, 12] with no initial_guess, lr_weight
0.02 for sea-lr_s*'s `long_range.weight`. cace's 4-channel charge, its bias-free
head and the Ewald constant prefactor (cace 1.0 vs les 90.4756) have no
les_params key and are absorbed by the free charge net; see the module docstring.
"""

ARMS = {
    'deepmd': {
        'mtype': 'standard',
        'les': None,
        'phases': PHASES_SR,
        'doc': "short-range control on cace's schedule (mirrors fit-interface-mp0-sr)",
    },
    'deepmd-les': {
        'mtype': 'hybrid_ener',
        'les': LES_COMMON,
        'phases': PHASES_LR,
        'doc': "se_a + learned charge/Ewald on cace's schedule "
               "(mirrors fit-interface-mp0)",
    },
    'deepmd-les-cace-sched': {
        'mtype': 'hybrid_ener',
        'les': LES_SIBLING,
        'les_comment': LES_COMMENT_SIBLING,
        'phases': PHASES_LR,
        'descriptor': SIBLING_DESCRIPTOR,
        'fitting': SIBLING_FITTING,
        'doc': "sea-lr_s*'s own descriptor and cace SR net (Atomwise [32, 16] "
               "silu, without its parallel linear skip), with deepmd's "
               "hybrid_ener long range: the same model, and the same schedule, "
               "as the sibling arm sea-lr_s*",
    },
}

TEMPLATE = """\
# {arm}, replicate {tag}, block {n}/{nseg} - generated by
# campaign_water_interface/deepmd/gen_inputs.py
# {doc}
#
# schedule: cace PHASES, block {n} = E weight {w_e} for {epochs} epochs
# block LR: start_lr {start_lr} with decay_steps {decay_steps} / decay_rate {decay_rate},
#           i.e. cace's StepLR(20, 0.5) on this task's epoch counter
# chain:   run after block {prev}; --init-model rebuilds the optimiser, which is
#          what cace does at every phase-0 block
#
# precision: float32, and it is NOT set here. deepmd's interface precision comes
#          from the DP_INTERFACE_PREC environment variable, read once at
#          `import deepmd.env` (deepmd/env.py:34-50); there is no input.yaml key
#          for it, and the model-level `precision` key is a different thing (it
#          would pin the embedding-net parameters while leaving the interface and
#          the LES/Ewald path in fp64). Run these configs through run_chain.py,
#          which sets DP_INTERFACE_PREC=low for the child `dp`; a bare
#          `dp --pt train` runs them in deepmd's default float64 instead, which
#          is roughly 1/64 the fp32 rate on this GPU class and would silently
#          break the cost comparison against cace.
model:
  type: {mtype}
  type_map: ["O", "H"]
  data_stat_nbatch: {data_stat_nbatch}
  descriptor:
    type: {desc_type}
    sel: {desc_sel}
    rcut_smth: {desc_rcut_smth}
    rcut: {desc_rcut}
    neuron: {desc_neuron}
    axis_neuron: {desc_axis_neuron}
    resnet_dt: {desc_resnet_dt}
    seed: {desc_seed}
  fitting_net:
    neuron: {fit_neuron}
    seed: {fit_seed}
{fit_extra}{les_block}
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
LES block: cace fit-interface-mp0, knob for knob - sigma 1, dl 2, self-interaction
removed (cace's default), charge head [24, 12] with no initial_guess, lr_weight
0.02 for cace's pot2 weight. cace's 4-channel charge, its bias-free head and the
Ewald constant prefactor (cace 1.0 vs les 90.4756) have no les_params key and are
absorbed by the free charge net; see the module docstring.
"""


def natoms_of(data_dir, dirs):
    """Atoms per frame, read from the data rather than remembered.

    It decides `pref_e` (see the module docstring), so a wrong value silently
    weights the energy loss wrongly: `pref_e = natoms * w_E` is only cace's
    ``w_E * MSE(E)`` when natoms is the real atom count, because deepmd divides
    the energy term by ``atype.shape[-1]`` (`pt/train/wrapper.py`).
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


def build_segments(step_size, phases, steps_per_epoch, natoms, warmup):
    """Expand cace's phase list into the chained deepmd blocks.

    Phase 0's blocks are fresh tasks, so each one restarts the epoch counter and
    therefore the LR; the later phases continue the last phase-0 task's counter,
    which is what cace's StepLR sees.

    ``warmup`` is a parameter rather than the module constant because the smoke
    mode has to shorten it: `warmup_steps` must stay below `numb_steps`, or
    deepmd's `stop_steps = num_steps - warmup_steps` goes negative and the
    `decay_steps >= stop_steps` guard substitutes a nonsense period.
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
                'warmup': warmup if fresh else 0,
                'natoms': natoms,
                'pref_e': natoms * w_e,
                'pref_f': FORCE_PREF,
            })
            offset += num_steps
            task_epoch += epochs
    return segments


def fit_extra(fitting):
    """The fitting_net keys beyond neuron/seed, indented; empty when there are none.

    Only the sibling arm carries them: deepmd's own fitter takes its activation
    and resnet_dt from defaults, while cace's Atomwise needs silu spelled out.
    """
    out = []
    for key in ('activation_function', 'resnet_dt'):
        if key in fitting:
            v = fitting[key]
            out.append(f'    {key}: {str(v).lower() if isinstance(v, bool) else v}')
    return ''.join(line + '\n' for line in out)


def les_block(les, comment=LES_COMMENT):
    """The les_params mapping, indented under model:, with booleans lowercase."""
    if les is None:
        return ''
    out = [f'  # {line}' for line in comment.split('\n') if line]
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
    """A float the way a hand-written config wants it: shortest round-trip."""
    return f'{float(v):.12g}'


def cfg_list(vals):
    return '[' + ', '.join(cfg_num(v) for v in vals) + ']'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-root', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--replicate-tags', default=''.join(r['tag'] for r in REPLICATES))
    ap.add_argument('--smoke', action='store_true',
                    help='tiny blocks, for exercising the chain only - NOT the '
                         'cace schedule')
    args = ap.parse_args()

    step_size, steps_per_epoch = 20, STEPS_PER_EPOCH
    disp_freq, save_freq, numb_btch = DISP_FREQ, SAVE_FREQ, 50
    warmup = WARMUP
    if args.smoke:
        # Three 1-epoch blocks per phase and a decay period of one epoch, so
        # `decay_steps` stays below `num_steps - warmup_steps` and nothing trips
        # deepmd's `decay_steps >= stop_steps` substitution, which would silently
        # swap in a default period. Block LR, and so the whole point of the file,
        # is NOT cace's in this mode.
        phases_override = 2
        step_size, steps_per_epoch = 1, 50
        warmup = 5
        disp_freq, save_freq, numb_btch = 50, 50, 2

    wanted_arms = args.arms.split(',')
    unknown = [a for a in wanted_arms if a not in ARMS]
    if unknown:
        raise SystemExit(f'unknown arm(s) {unknown}; known: {list(ARMS)}')
    wanted_reps = [r for r in REPLICATES if r['tag'] in args.replicate_tags]
    if not wanted_reps:
        raise SystemExit(f'no replicate matches {args.replicate_tags!r}')
    natoms = natoms_of(args.data, TRAIN_DIRS)

    train = [os.path.join(args.data, d) for d in TRAIN_DIRS]
    valid = [os.path.join(args.data, VALID_DIR)]

    # systems paths are resolved against the process CWD, which the chain sets to
    # the block directory, so write them relative to a block dir.
    def rel(paths, blockdir):
        return ', '.join(f'"{os.path.relpath(p, blockdir)}"' for p in paths)

    for arm in wanted_arms:
        spec = ARMS[arm]
        # Per-arm overrides: absent means the deepmd arms' own recipe, so the two
        # arms that already have trained checkpoints regenerate byte for byte.
        descriptor = spec.get('descriptor', DESCRIPTOR)
        fitting = spec.get('fitting', FITTING)
        phases = ([(w, n, phases_override) for w, n, _ in spec['phases']]
                  if args.smoke else spec['phases'])
        segments = build_segments(step_size, phases, steps_per_epoch, natoms,
                                  warmup)
        for rep in wanted_reps:
            tag = f'{arm}_s{rep["tag"]}'
            armdir = os.path.join(args.out_root, tag)
            os.makedirs(armdir, exist_ok=True)
            for seg in segments:
                # every block must end on a save_freq boundary or the chain's
                # --init-model source (model.ckpt-{num_steps}.pt) is never written
                if seg['num_steps'] % save_freq:
                    raise SystemExit(
                        f"{tag} block {seg['n']}: num_steps {seg['num_steps']} is "
                        f"not a multiple of save_freq {save_freq}")
                blockdir = os.path.join(armdir, f's{seg["n"]}')
                os.makedirs(blockdir, exist_ok=True)
                body = TEMPLATE.format(
                    arm=arm, tag=rep['tag'], doc=spec['doc'],
                    n=seg['n'], nseg=len(segments), prev=seg['n'] - 1,
                    mtype=spec['mtype'],
                    data_stat_nbatch=DATA_STAT_NBATCH,
                    desc_type=descriptor['type'],
                    desc_sel=cfg_list(descriptor['sel']),
                    desc_rcut_smth=cfg_num(descriptor['rcut_smth']),
                    desc_rcut=cfg_num(descriptor['rcut']),
                    desc_neuron=cfg_list(descriptor['neuron']),
                    desc_axis_neuron=descriptor['axis_neuron'],
                    desc_resnet_dt=str(descriptor['resnet_dt']).lower(),
                    desc_seed=rep['desc_seed'],
                    fit_neuron=cfg_list(fitting['neuron']),
                    fit_seed=rep['fit_seed'],
                    fit_extra=fit_extra(fitting),
                    train_seed=rep['train_seed'],
                    les_block=les_block(
                        spec['les'], spec.get('les_comment', LES_COMMENT)),
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
                'cace_recipe': {
                    # the arm's own phase list says which cace script it mirrors:
                    # PHASES_LR is mp0's, PHASES_SR is mp0-sr's 400-epoch subset
                    'source': 'cace-lr-fit-datarepo/.../fit-water-interface/'
                              + ('fit-interface-mp0'
                                 if spec['phases'] is PHASES_LR
                                 else 'fit-interface-mp0-sr')
                              + '/fit-cace-nnp.py',
                    'steps_per_epoch': STEPS_PER_EPOCH,
                    'phases': spec['phases'],
                    'steplr': {'step_size_epochs': 20, 'gamma': DECAY_RATE},
                    'optimizer': {'type': 'Adam', 'lr': BASE_LR,
                                  'betas': [0.99, 0.999],
                                  'note': 'deepmd Adam betas are (0.9, 0.999); '
                                          'not settable'},
                    'warmup_steps': WARMUP,
                    'warmup_note': "cace's warmup_steps 5 counts epochs "
                                   "(global_step advances once per epoch), so it "
                                   "is 2250 deepmd steps",
                    'max_grad_norm': GRAD_MAX_NORM,
                    'force_weight': FORCE_PREF,
                    'energy_weight_to_pref_e': f'natoms={natoms} x w_E, undoing '
                                               "deepmd's per-atom energy loss",
                    'split': {'valid_fraction': 0.1, 'seed': 1,
                              'source': 'prep_water_interface.py -> '
                                        'split_manifest.txt'},
                    'seeds': {k: rep[k] for k in ('desc_seed', 'fit_seed',
                                                  'train_seed')},
                },
                # the arm's own model, not the module default: an arm that
                # overrides the descriptor or the fitting net must not have its
                # segments.json describe the other one
                'model': {'descriptor': spec.get('descriptor', DESCRIPTOR),
                          'fitting_net': spec.get('fitting', FITTING),
                          'type': spec['mtype'], 'les_params': spec['les']},
                'segments': segments,
                'total_steps': sum(s['num_steps'] for s in segments),
            }
            with open(os.path.join(armdir, 'segments.json'), 'w') as fh:
                json.dump(record, fh, indent=2)

            with open(os.path.join(armdir, 'chain.sh'), 'w') as fh:
                fh.write('#!/bin/bash\n'
                         '# generated by gen_inputs.py; use run_chain.py to run '
                         'it with timing\n'
                         'set -euo pipefail\n'
                         'cd "$(dirname "$0")"\n')
                for seg in segments:
                    n = seg['n']
                    init = (f' --init-model ../s{n - 1}/model.ckpt-'
                            f'{segments[n - 2]["num_steps"]}.pt' if n > 1 else '')
                    fh.write(f'(cd s{n} && dp --pt train input.yaml{init})\n')
            print(f'wrote {os.path.relpath(armdir, HERE)}: '
                  f'{len(segments)} blocks, '
                  f'{record["total_steps"]} steps total')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())