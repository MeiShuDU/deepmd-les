"""Generate the 8 deepmd configs for the FastLearn cace-vs-deepmd campaign.

Two axes, crossed:

  arm        what the long-range term is
  ---------  ---------------------------------------------------------------
  deepmd                       none (``type: standard``) - the SR control
  deepmd-les                   learned per-atom charge + Ewald
  deepmd-les-claim-neutral     same, charges projected onto sum(q) = 0
  deepmd-les-freeze-charge     SPC/E charges held fixed, framework is a
                               classical Ewald (no charge NN at all)

  replicate  descriptor/fitting seed and training seed
  ---------  ---------------------------------------------------------------
  A          descriptor 1 / fitting 1, training seed 10
  B          descriptor 2 / fitting 2, training seed 20

which is 4 x 2 = 8 runs, all sharing one SR network setup and one step count,
so the only thing separating them is the long-range treatment.

Short-range network - the local 10,000-step extended run
(``01.train/rerun/gen_extended.py``), with ``rcut`` moved 6.00 -> 5.50 as
asked. Its ``neuron``/``axis_neuron``/``resnet_dt``/``sel`` conventions are
kept; ``sel`` is the one number that has to follow ``rcut``, see below.
Per-arm comments in the generated files record where each block comes from.

Long-range block - the cace ``fit-water-timing`` scripts, knob for knob. cace
and LES use the same Ewald convention (real-space sum excludes the self term,
k-space adds it back unless it is asked to remove it), so the values transfer
literally:

  cace                                 -> les_params
  ------------------------------------    --------------------------------
  EwaldPotential(dl=2, sigma=1.0,         dl: 2.0
    remove_self_interaction=False)        sigma: 1.0
                                          remove_self_interaction: false
  Atomwise(n_out=1, n_hidden=[24,12],     n_hidden: [24, 12]
    add_linear_nn=True, bias=False)
  E_LR added to E_SR at weight 1          lr_weight: 1.0
  q used raw, no prefactor                output_scaling_factor: 1.0
                                          (LES defaults to 0.1, so this has to
                                           be written out to match cace)

Two knobs a faithful cace copy cannot carry over, recorded so nobody reads a
difference into them later: cace's charge head has ``bias=False`` while the LES
Atomwise has a bias, and cace applies no total-charge constraint at all, which
is exactly why ``claim_neutral`` is a separate arm rather than part of the
match.

A third, and the one that actually changes numbers: the two Ewald kernels
differ by a constant prefactor.

    cace/modules/ewald.py   norm_factor = 1.0
    les/src/les/module/ewald.py   norm_factor = 90.4756

Both comments call it 1/(2 eps_0), and 1/(2 * 5.52635e-3) = 90.47, so cace's
1.0 keeps the energy in units where the "charge" is sqrt(4 pi eps_0) e times the
physical charge, i.e. about 9.49x too big - which is the whole reason the
recorded cace q sheet reads like O -4.97..+4.78 instead of a plausible +-0.5 e.
It is a unit convention, not physics, and it is not reachable from
``les_params``: ``Les._parse_arguments`` reads sigma / dl /
remove_self_interaction / use_epsilon_r_scaling and nothing else, so the kernel
is constructed with the 90.4756 default. The deepmd arms therefore run in the
physical convention, in which ``initial_guess`` = SPC/E (-0.8476, +0.4238 e)
is a sensible starting point rather than a negligible one, and in which the
freeze-charge arm's fixed charge gives a physically scaled E_LR (~11 eV/frame
here) instead of the ~0.12 eV/frame the cace units would produce - i.e. a real
long-range term, which is the point of having that arm.

For the two local-charge arms this factor is absorbed: E_LR is bilinear in q and
the charge net is free, so any q_c maps to q_c / sqrt(90.4756), and lr_weight
1.0 remains the faithful reading of cace adding E_LR to E_SR at weight 1. Only
the freeze-charge arm's scale is fixed by the convention, and there the physical
one is the one worth testing.

`sel` at rcut 5.5: the measured maximum over the 320 training frames is
[31, 59] neighbours of type O and H respectively (reproduced with deepmd's own
NeighborStatOP, whose ``auto`` rule would round that up to [32, 60]). The value
written here is [39, 73], the same ~1.25x headroom the 100k campaign's
rcut-5.5 config uses, so the SR network has the capacity its siblings had and
the descriptor never truncates.

Batch schedule, revised from the extended run as asked: the extended run's
``batch_size: auto`` resolves to one frame per system per step regardless of
its name (``pt/utils/dataloader.py`` sets rule 32 with a ceiling, and 32 // 192
= 0 -> 1), and the cace arms use batch 2 over the same 320 frames. Setting
``batch_size: 2`` here puts both families at 160 optimizer steps per epoch and
80,000 steps = 500 epochs, which is the point of the move to this data.

Data: these arms read the native npy systems under
``DeePMD-kit-FastLearn/data``, unchanged from the extended and 100k runs. The
cace arms read ``campaign_fastlearn/xyz``, which ``export_data.py`` writes
from those same frames.

Cost knobs, the only deliberate deviations: ``disp_freq`` 100 -> 1000 and
``save_freq`` 1000 -> 10000, to bound the log for a run 8x longer than the
extended one. ``max_ckpt_keep: 5`` is the extended run's own value, kept: a
checkpoint is 26 MB, so the whole campaign's retention is under 1 GB, and five
late checkpoints (steps 40000-80000) is what the "evaluate the whole split and
average late checkpoints" rule needs. ``numb_steps`` 80000, per instruction.

Usage:
    python gen_inputs.py                     # writes runs/<arm>_s<A|B>/input.yaml
    python gen_inputs.py --numb-steps 20 --out-root /tmp/smoke
"""
import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
# the deepmd arms read the native npy data; the cace arms read
# campaign_fastlearn/xyz, the same frames exported to extended XYZ
DATA = os.path.join(CAMPAIGN, '..', 'data')

LR = {'decay_steps': 5000, 'start_lr': '0.001', 'stop_lr': '3.51e-8'}

# The SPC/E charges, ordered by type index: type_map is ["O", "H"].
SPCE = [-0.8476, 0.4238]

# Shared by every LES arm; see the module docstring for the cace provenance.
LES_COMMON = {
    'sigma': 1.0,
    'dl': 2.0,
    'remove_self_interaction': False,
    'output_scaling_factor': 1.0,
    'lr_weight': 1.0,
}

ARMS = {
    'deepmd': {
        'mtype': 'standard',
        'les': None,
        'doc': 'short-range only control: identical SR network, no long-range term',
    },
    'deepmd-les': {
        'mtype': 'hybrid_ener',
        'les': dict(LES_COMMON, local_charge=True, initial_guess=SPCE),
        'doc': 'learned per-atom charge; initial_guess starts it at SPC/E and the '
               'net learns the deviation',
    },
    'deepmd-les-claim-neutral': {
        'mtype': 'hybrid_ener',
        'les': dict(LES_COMMON, local_charge=True, initial_guess=SPCE,
                    claim_neutral=True),
        'doc': 'as deepmd-les, with the total charge projected to zero each frame '
               '(cace applies no such constraint)',
    },
    'deepmd-les-init0-neutral': {
        'mtype': 'hybrid_ener',
        # initial_guess is an ADDITIVE per-type baseline, so [0, 0] starts the
        # charge net at exactly q = 0 rather than at SPC/E. Against
        # deepmd-les-claim-neutral this differs in ONE knob - where the charge
        # channel is anchored - because the neutrality projection is the same in
        # both. That is what isolates the anchor from the neutrality constraint as
        # a cause of the charge-scale gap against cace-lr, whose q head has no
        # anchor either.
        'les': dict(LES_COMMON, local_charge=True, initial_guess=[0.0, 0.0],
                    claim_neutral=True),
        'doc': 'as deepmd-les-claim-neutral, but the charge net starts at zero '
               'instead of the SPC/E table: the learned |q| owes nothing to the '
               'initialisation',
    },
    'deepmd-les-freeze-charge': {
        'mtype': 'hybrid_ener',
        # freeze_charge fixes the charges, so it is mutually exclusive with
        # initial_guess (argcheck in les.py rejects the pair), and there is no
        # charge NN for n_hidden / output_scaling_factor to act on.
        'les': {'freeze_charge': SPCE, 'sigma': 1.0, 'dl': 2.0,
                'remove_self_interaction': False, 'lr_weight': 1.0},
        'doc': 'SPC/E charges held fixed: the long-range term is a classical '
               'Ewald with no learnable charge and no initial_guess',
    },
    # The two rsi controls. Each is its parent arm with ONE knob flipped -
    # remove_self_interaction False -> True - so a diff against the parent shows
    # a single changed line and any metric difference is the flag.
    #
    # What the flag does, and why it only matters for a LEARNED charge: les sums
    # k-space and subtracts Sum(q^2)/(sigma (2pi)^1.5) * norm_factor only when
    # this is true, so with False that self term stays in E_LR. It is
    # +5.7446 eV * Sum(q^2) per frame. For a per-type fixed table Sum(q^2) is a
    # constant (SPC/E: 69.0), so the term is a constant offset the SR bias eats
    # and its force is exactly zero. For a learned charge Sum(q^2) is a function
    # of geometry (measured ~116 on the rsi=False siblings, i.e. |q| ~1.30x
    # SPC/E), so the term is a spurious quadratic regulariser pulling |q| toward
    # zero AND its gradient is a fake force in the very LR channel under test -
    # measured at 1.01-1.31x the arms' total force. That is the confound these
    # two arms price.
    'deepmd-les-rsi': {
        'mtype': 'hybrid_ener',
        'les': dict(LES_COMMON, local_charge=True, initial_guess=SPCE,
                    remove_self_interaction=True),
        'doc': 'as deepmd-les, but the Ewald self term is REMOVED (the physical '
               'convention) instead of retained: prices the flag for a learned charge',
    },
    'deepmd-les-claim-neutral-rsi': {
        'mtype': 'hybrid_ener',
        'les': dict(LES_COMMON, local_charge=True, initial_guess=SPCE,
                    claim_neutral=True, remove_self_interaction=True),
        'doc': 'as deepmd-les-claim-neutral, but the Ewald self term is REMOVED '
               '(the physical convention) instead of retained',
    },
}

REPLICATES = [
    {'tag': 'A', 'desc_seed': 1, 'fit_seed': 1, 'train_seed': 10},
    {'tag': 'B', 'desc_seed': 2, 'fit_seed': 2, 'train_seed': 20},
]

TRAIN_DIRS = ['data_0', 'data_1', 'data_2']
VALID_DIR = 'data_3'

TEMPLATE = """\
# {arm}, replicate {tag} - generated by campaign_fastlearn/deepmd/gen_inputs.py
# {doc}
#
# SR network: 01.train/rerun/gen_extended.py (rcut 6.00 -> 5.50, sel follows rcut)
# LR block:   cace-lr-fit-datarepo/.../fit-water-timing/{source}
# steps:      {numb_steps} = {epochs} epochs of {steps_per_epoch} steps
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
    neuron: [240, 240, 240]
    resnet_dt: true
    seed: {fit_seed}
{les_block}
learning_rate:
  type: exp
  decay_steps: {decay_steps}
  start_lr: {start_lr}
  stop_lr: {stop_lr}

loss:
  type: ener
  start_pref_e: 0.02
  limit_pref_e: 1.0
  start_pref_f: 1000.0
  limit_pref_f: 1.0
  start_pref_v: 0.0
  limit_pref_v: 0.0

training:
  training_data:
    systems: [{train}]
    batch_size: 2
  validation_data:
    systems: [{valid}]
    batch_size: 1
    numb_btch: 3
  numb_steps: {numb_steps}
  seed: {train_seed}
  disp_file: "lcurve.out"
  disp_freq: {disp_freq}
  save_freq: {save_freq}
  save_ckpt: "model.ckpt"
  max_ckpt_keep: 5
"""


def les_block(les, comment):
    """The les_params mapping, indented under model:, with booleans lowercase."""
    if les is None:
        return ""
    out = [f'  # {line}' for line in comment.split('\n')]
    out.append("  les_params:")
    for k, v in les.items():
        if isinstance(v, bool):
            v = str(v).lower()
        elif isinstance(v, list):
            v = '[' + ', '.join(str(x) for x in v) + ']'
        out.append(f'    {k}: {v}')
    # harness-only knobs, same for every LES arm
    out.append("    verbose: true")
    out.append("    log_freq: 100")
    return "\n".join(out) + "\n"


def rel(paths, rundir):
    """Data systems as paths relative to a run dir, so the tree is relocatable.

    Each run executes with its own directory as the CWD (that is what keeps
    lcurve.out and the LES package's hardcoded les.log from colliding), so the
    configs point up at the shared data by a route that survives an rsync to
    the pod.
    """
    return ', '.join(f'"{os.path.relpath(p, rundir)}"' for p in paths)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-root', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--numb-steps', type=int, default=80000)
    ap.add_argument('--disp-freq', type=int, default=1000)
    ap.add_argument('--save-freq', type=int, default=10000)
    ap.add_argument('--steps-per-epoch', type=int, default=160,
                    help='320 training frames / batch_size 2, used only for the '
                         'comment that names the equivalent epoch count')
    ap.add_argument('--arms', default=','.join(ARMS),
                    help='comma-separated subset of the arms to write')
    args = ap.parse_args()

    train = [os.path.join(args.data, d) for d in TRAIN_DIRS]
    valid = [os.path.join(args.data, VALID_DIR)]

    wanted = args.arms.split(',')
    unknown = [a for a in wanted if a not in ARMS]
    if unknown:
        raise SystemExit(f'unknown arm(s) {unknown}; known: {list(ARMS)}')

    for arm in wanted:
        spec = ARMS[arm]
        for rep in REPLICATES:
            tag = f'{arm}_s{rep["tag"]}'
            rundir = os.path.join(args.out_root, tag)
            os.makedirs(rundir, exist_ok=True)
            body = TEMPLATE.format(
                arm=arm, tag=rep['tag'], doc=spec['doc'], mtype=spec['mtype'],
                source=('fit_cace_new.py' if spec['les'] else 'n/a (no long range)'),
                desc_seed=rep['desc_seed'], fit_seed=rep['fit_seed'],
                train_seed=rep['train_seed'],
                les_block=les_block(
                    spec['les'],
                    'LES block: cace fit-water-timing, knob for knob, except the\n'
                    'Ewald prefactor: cace uses norm_factor 1.0 (charges scaled by\n'
                    'sqrt(1/(2 eps_0)) ~ 9.49) and les hardcodes the physical\n'
                    '90.4756. Not reachable from les_params; see gen_inputs.py.'),
                numb_steps=args.numb_steps,
                epochs=args.numb_steps // args.steps_per_epoch,
                steps_per_epoch=args.steps_per_epoch,
                disp_freq=args.disp_freq, save_freq=args.save_freq,
                train=rel(train, rundir), valid=rel(valid, rundir),
                **LR,
            )
            path = os.path.join(rundir, 'input.yaml')
            with open(path, 'w') as fh:
                fh.write(body)
            print('wrote', os.path.relpath(path, HERE))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
