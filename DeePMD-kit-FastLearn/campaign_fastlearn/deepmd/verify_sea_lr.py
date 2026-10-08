"""Check that the generated chained deepmd runs reproduce cace's LR schedule.

The generator claims that eight chained deepmd ``exp`` schedules equal cace's
``StepLR(step_size=20, gamma=0.5)`` on a per-task epoch counter, warmup included.
That claim is arithmetic, so it is checked arithmetically rather than by eyeballing
a training curve: this script rebuilds the LR the trainer itself would build
(``LearningRateExp`` over the block's own yaml) and compares it, step by step over
every step of every block of every arm, against a from-scratch reimplementation of
cace's schedule.

Two ways the claim could fail and both are asserted against:

* deepmd silently swaps in a default ``decay_steps`` when
  ``decay_steps >= numb_steps - warmup_steps`` (``LearningRateExp.__init__``), which
  would replace the halving-every-20-epochs period with a much shorter one. The
  first check is that each block's own ``decay_steps`` survived construction.
* the block-entry ``start_lr`` must be the value cace's counter holds at that
  block's first epoch, or the chained segments do not concatenate into cace's run.

The only permitted disagreements are the ones the generator documents: a one-step
lag in the 5-step warmup ramp, and a 5-step lag at each 20-epoch halving inside the
five warmup blocks (``warmup_steps`` shifts the exponent in
``lr(s) = start_lr * decay_rate ** ((s - warmup) // decay_steps)``). Every other
step of every block must agree exactly. The script prints those windows and asserts
they are the only ones.

Usage:
    python verify_sea_lr.py [--root runs]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_inputs_sea import (BASE_LR, DATA, DECAY_RATE, FORCE_PREF, REPLICATES,
                            TRAIN_DIRS, natoms_of)
from deepmd.common import j_loader
from deepmd.dpmodel.utils.learning_rate import LearningRateExp


def cace_lr_steps(seg):
    """cace's per-step LR over one block, from cace's own recipe.

    cace holds one `StepLR(step_size, 0.5)` per TASK and steps it once per epoch,
    so the LR is flat inside an epoch and the counter it floors is the task's own
    epoch count. A phase-0 block is a fresh task (counter restarts, warmup applies);
    a later block continues the fifth phase-0 task's counter.
    """
    steps_per_epoch = seg['steps_per_epoch']
    step_size = 20
    warmup = seg['warmup']
    out = np.empty(seg['num_steps'])
    for s in range(seg['num_steps']):
        epoch = s // steps_per_epoch
        counter = epoch if seg['fresh_task'] else seg['task_epoch'] + epoch
        base = BASE_LR * DECAY_RATE ** (counter // step_size)
        # cace's train loop: `lr_scale = min(1, (global_step + 1) / warmup_steps)`
        if warmup and s < warmup:
            base *= min(1.0, (s + 1) / warmup)
        out[s] = base
    return out


def deepmd_lr_steps(jdata, warmup):
    """The per-step LR deepmd's trainer would use, from the block's own yaml.

    `OptimizerPytorch` builds `lr = start_lr * warm_up_linear(step + start_step, w)`
    with `start_step = 0` under `--init-model`, and `warm_up_linear(s, w)` is `s / w`
    below `w` and `lr_exp.value(s - w) / start_lr` above it.
    """
    params = dict(jdata['learning_rate'])
    params.pop('type', None)
    numb_steps = jdata['training']['numb_steps']
    # exactly what `Trainer.get_lr` does before constructing the schedule
    params['stop_steps'] = numb_steps - warmup
    lr = LearningRateExp(**params)
    asked = jdata['learning_rate']['decay_steps']
    out = np.empty(numb_steps)
    for s in range(numb_steps):
        if warmup and s < warmup:
            out[s] = lr.start_lr * (s / warmup)
        else:
            out[s] = lr.value(s - warmup)
    return lr, out, asked


def blocks(root, arm, tag):
    armdir = os.path.join(root, f'{arm}_s{tag}')
    with open(os.path.join(armdir, 'segments.json')) as fh:
        record = json.load(fh)
    for seg in record['segments']:
        yield seg, os.path.join(armdir, f's{seg["n"]}', 'input.yaml')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'runs'))
    ap.add_argument('--arms', default='deepmd-cace-sched,deepmd-les-cace-sched')
    args = ap.parse_args()

    n_bad = 0
    n_shifted = 0
    # read from the data, not from segments.json, so the E:F weight is checked against
    # the atom count deepmd itself will divide by (`pt/train/wrapper.py:192`)
    natoms_data = natoms_of(os.path.join(DATA), TRAIN_DIRS)
    print(f'atoms per frame, from {TRAIN_DIRS[0]}/type.raw: {natoms_data}')
    for arm in args.arms.split(','):
        for rep in REPLICATES:
            tag = rep['tag']
            print(f'\n=== {arm}_s{tag} ===')
            counter_at_end = 0
            for seg, path in blocks(args.root, arm, tag):
                jdata = j_loader(path)
                numb_steps = jdata['training']['numb_steps']
                warmup = jdata['training']['warmup_steps']
                lr, got, asked = deepmd_lr_steps(jdata, warmup)
                want = cace_lr_steps(seg)

                # (1) the substitution trap did not fire: the period is the one asked for
                if lr.decay_steps != asked:
                    n_bad += 1
                    print(f'  s{seg["n"]}: *** decay_steps substituted '
                          f'{asked} -> {lr.decay_steps}')
                # (2) the block starts where cace's counter says it should
                if abs(lr.start_lr - seg['start_lr']) > 0:
                    n_bad += 1
                    print(f'  s{seg["n"]}: *** start_lr {lr.start_lr} != '
                          f'cace {seg["start_lr"]}')
                # (3) the loss weights and the SR net are the cace ones. The config
                # writes `pref_e` rounded (`9.6`, not 96*0.1's 9.600000000000001), so
                # this is a tolerance match and not an identity one.
                loss = jdata['loss']
                fit = jdata['model']['fitting_net']
                ok_loss = (np.isclose(loss['start_pref_e'], seg['pref_e'], rtol=1e-12)
                           and loss['start_pref_f'] == FORCE_PREF
                           and loss['start_pref_v'] == 0.0
                           and loss['start_pref_e'] == loss['limit_pref_e']
                           and loss['start_pref_f'] == loss['limit_pref_f'])
                ok_fit = (fit['neuron'] == [32, 16]
                          and fit['activation_function'] == 'silu')
                ok_natoms = (seg['natoms'] == natoms_data
                             and abs(seg['pref_e'] - natoms_data * seg['w_e']) < 1e-9
                             and np.isclose(loss['start_pref_e'],
                                            natoms_data * seg['w_e'], rtol=1e-12))
                if not (ok_loss and ok_fit and ok_natoms):
                    n_bad += 1
                    print(f'  s{seg["n"]}: *** loss/fitting not cace '
                          f'(loss {ok_loss}, fit {ok_fit}, natoms {ok_natoms})')

                # (4) the LR trajectory. Compare in ratio space: a relative error of
                # zero is what "the same schedule" means here.
                denom = np.where(want > 0, want, 1.0)
                rel = np.where(want > 0, np.abs(got - want) / denom, np.abs(got))
                bad = np.flatnonzero(rel > 1e-12)
                # the permitted windows: warmup ramp steps, and the 5 steps after each
                # halving boundary inside a warmup block (the `- warmup` in the exponent)
                boundaries = [seg['decay_steps'] * k
                              for k in range(1, seg['epochs'] // 20 + 1)]
                permitted = set(range(warmup)) if warmup else set()
                for b in boundaries:
                    permitted |= set(range(b, min(b + warmup, numb_steps))) if warmup else set()
                unexpected = [int(s) for s in bad if int(s) not in permitted]
                if unexpected:
                    n_bad += 1
                    i = unexpected[0]
                    print(f'  s{seg["n"]}: *** {len(unexpected)} unexpected step(s), '
                          f'first at {i}: got {got[i]:.10g} want {want[i]:.10g}')
                n_shifted += len([s for s in bad])
                # (5) concatenation: cace keeps one task across the phase boundaries, so
                # the shared blocks must continue the fifth task's counter with no gap or
                # overlap. A jump in the LR itself is not a defect - cace's StepLR halves
                # exactly at epoch 40, which is block 6's first epoch - so what is checked
                # here is the counter, not the value.
                counter = 0 if seg['fresh_task'] else counter_at_end
                if seg['task_epoch'] != counter:
                    n_bad += 1
                    print(f'  s{seg["n"]}: *** task counter {seg["task_epoch"]} != '
                          f'expected {counter}')
                counter_at_end = seg['task_epoch'] + seg['epochs']

                uniq = np.unique(np.round(want, 12))
                print(f'  s{seg["n"]:>2d} phase {seg["phase"]} w_E {seg["w_e"]:>6g}  '
                      f'{numb_steps:>5d} steps  start {lr.start_lr:<12.8g} '
                      f'stop {lr.min_lr:<12.8g}  decay_steps {lr.decay_steps}  '
                      f'warmup {warmup}  distinct LR {len(uniq)}  '
                      f'pref_e {seg["pref_e"]:g} = {seg["natoms"]}x{seg["w_e"]:g}  '
                      f'mismatched steps {len(bad)}')

    print(f'\n{n_shifted} step(s) differ across all blocks, all inside the documented '
          f'warmup/boundary windows' if not n_bad else '')
    if n_bad:
        print(f'\nFAIL: {n_bad} problem(s)')
        return 1
    print('OK: every block\'s LR is cace\'s StepLR(20, 0.5) on cace\'s task counter,')
    print('    except the documented one-step warmup lag and 5-step boundary lag')
    print('    inside the five warmup blocks; decay_steps was never substituted.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
