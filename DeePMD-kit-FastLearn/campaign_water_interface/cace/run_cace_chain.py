"""Run the cace arms of the water-interface campaign, and time them.

This is the cace counterpart of ``../deepmd/run_chain.py`` and writes the same
``timing.json``, so the two families' wall clocks land in one table. Read that
file's docstring first: the *shape* of the output is deliberately identical, and
the differences below are all consequences of cace's script being a single
process rather than a chain of ``dp --pt train`` calls.

What differs from the deepmd runner, and why
--------------------------------------------
blocks    cace's ``fit-cace-nnp.py`` runs all of its training loops in one
          process: five fresh 40-epoch ``TrainingTask``s at E weight 0.01, then
          three (mp0) or two (mp0-sr) 100-epoch loops sharing the last task at
          E weights 1, 10, 1000. A "block" is therefore one of those loops, and
          the script itself times each one and appends it to ``blocks.json`` in
          its working directory. This runner does not re-time them: a block's
          ``seconds`` is measured around the ``task.fit`` call by
          ``perf_counter`` inside the process, which is strictly more accurate
          than timing the whole process and dividing.
resume    cace has none worth the name. ``TrainingTask.fit`` writes
          ``checkpoint.pt`` every 10 epochs (``cace/tasks/train.py:261-262``),
          but ``checkpoint()`` (``:284``) records no ``global_step``, and the
          epoch counter is what the ``StepLR`` schedule and ``warmup_steps``
          run on - so even the restore path that exists (``load_state_dict``,
          ``:293``) could not put the schedule back where it left off. The
          published arm scripts never call it in any case, so a crashed arm
          restarts from epoch 0. That is the single most important cost fact
          for the pod plan, and it is why ``--blocks`` here bounds a run from
          *inside* (via ``CACE_STOP_AFTER``) rather than resuming a prefix of
          it: a bounded run is a cost probe and its ``blocks.json`` says
          ``"finished": false``.
cpu       cace's ``init_device('cuda')`` asserts CUDA exists
          (``cace/tools/torch_tools.py:45``), so it cannot fall back on its own;
          the arm scripts read ``CACE_DEVICE`` instead. ``--cpu`` sets it.
seed      ``TrainingTask`` takes no seed (``cace/tasks/train.py:17-52``) and
          cace builds its net from torch's global RNG, so the replicate is the
          single positional argument the arm scripts take, which they feed to
          ``torch.manual_seed`` before the first draw. A and B are 1 and 2.

Usage:
    python run_cace_chain.py --cpu --smoke --blocks 1     # cheapest plumbing check
    python run_cace_chain.py --arm cace-lr --rep A
    python run_cace_chain.py                              # both arms x both reps
    python run_cace_chain.py --dry-run
"""
import argparse
import datetime
import json
import os
import shlex
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(HERE, 'runs')
DATA = os.path.join(HERE, 'data', 'slab-fps-n-500.xyz')

# The number of training loops inside each arm's script, read off the source:
# fit-interface-mp0 has `for i in range(5)` plus three later fit calls, and
# fit-interface-mp0-sr has the same five plus two.
ARM_BLOCKS = {'cace-lr': 8, 'cace-sr': 7}
# The last file each script writes, as the postcondition.
ARM_FINAL = {'cace-lr': 'water-model-4.pth', 'cace-sr': 'water-model-3.pth'}
REPLICA_SEEDS = {'A': 1, 'B': 2}


def stamp():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def parse_blocks(spec, nblocks):
    """'1-8' / '1,2,5' / '3-' -> the cace block numbers to run."""
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            lo = int(lo) if lo else 1
            hi = int(hi) if hi else nblocks
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    unknown = [n for n in out if not 1 <= n <= nblocks]
    if unknown:
        raise SystemExit(f'block(s) {unknown} outside 1..{nblocks}')
    return sorted(set(out))


def read_blocks_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def run_arm(root, arm, tag, wanted, env, dry, python):
    """One arm, one replicate: launch the script, then turn its output into timing.json."""
    armdir = os.path.join(root, f'{arm}_s{tag}')
    script = os.path.join(HERE, arm, 'fit-cace-nnp.py')
    nblocks = ARM_BLOCKS[arm]
    seed = REPLICA_SEEDS[tag]
    blocks_json = os.path.join(armdir, 'blocks.json')
    out = os.path.join(armdir, 'timing.json')

    timing = {
        'arm': arm, 'replicate': tag, 'host': socket.gethostname(),
        'device': 'cpu' if env.get('CACE_DEVICE') == 'cpu' else 'gpu',
        'smoke': bool(env.get('_SMOKE')),
        'started': stamp(), 'ended': None, 'blocks': [], 'total_seconds': None,
        'total_steps': None, 's_per_batch': None, 'aborted_at': None,
        # extra, beyond run_chain.py's shape, so the two are still comparable
        'scheduled_blocks': nblocks, 'replicate_seed': seed,
        'cace_resume': False, 'restarted_from_partial': False,
    }

    # --- preflight -----------------------------------------------------------
    missing = [p for p in (script, DATA) if not os.path.exists(p)]
    if missing:
        print(f'{arm}_s{tag}: REFUSING - missing {missing}', flush=True)
        timing['aborted_at'] = 0
        return timing
    os.makedirs(armdir, exist_ok=True)

    prior = read_blocks_json(blocks_json)
    if prior is not None and prior.get('finished'):
        done = prior.get('blocks') or []
        print(f'{arm}_s{tag}: already complete ({len(done)} blocks, '
              f'"finished": true); skipping', flush=True)
        timing['blocks'] = adapt(done, 0)
        timing['max_gpu_mb'] = prior.get('max_gpu_mb')
        timing['steps_per_epoch'] = prior.get('steps_per_epoch')
        timing['restarted_from_partial'] = False
        finish(timing, out, dry)
        return timing
    if prior is not None and prior.get('blocks'):
        # cace cannot resume, so this is a restart from epoch 0, not a continuation
        print(f'{arm}_s{tag}: WARNING - a previous attempt died after '
              f'{len(prior["blocks"])} block(s) and cace cannot resume; this run '
              f'starts again from epoch 0', flush=True)
        timing['restarted_from_partial'] = True

    # --- the command ---------------------------------------------------------
    argv = [python, os.path.join('..', '..', arm, 'fit-cace-nnp.py'), str(seed)]
    env = dict(env)
    # CACE_STOP_AFTER bounds the run from *inside*: fit-cace-nnp.py's _fit raises
    # SystemExit(0) as soon as len(_BLOCKS) reaches it. It must therefore not be
    # set for a full schedule. `_fit` runs its stop check after appending the
    # block and after _dump(), so a bound of nblocks fires at the same place the
    # script's own ending does - and the SystemExit pre-empts the two statements
    # that follow the last _fit call: `task.save_model(ARM_FINAL)` and
    # `_dump(finished=True)`. The arm then trains its whole budget, reports
    # finished: false, and throws the model away; cace cannot resume, so that is
    # the entire arm. 0 disables the stop and lets the script end naturally.
    env['CACE_STOP_AFTER'] = '0' if wanted[-1] >= nblocks else str(wanted[-1])
    if env.get('_SMOKE'):
        argv.append('--smoke')
    print(f'{arm}_s{tag}: {nblocks} blocks scheduled, running blocks 1-{wanted[-1]}'
          + (f' (seed {seed})' if not dry else ''), flush=True)
    if wanted != list(range(1, wanted[-1] + 1)):
        print(f'  note: {wanted} is not a prefix, but cace has no resume, so the run '
              f'still goes 1-{wanted[-1]} and only 1-{wanted[-1]} is reported',
              flush=True)
    print(f'  cd {os.path.relpath(armdir, HERE)} && '
          f'{" ".join(shlex.quote(a) for a in argv)}', flush=True)
    if dry:
        timing['aborted_at'] = None
        finish(timing, out, dry)
        return timing

    t0 = stamp()
    log_path = os.path.join(armdir, 'train.log')
    # Stream straight into the log rather than capturing and writing it at the end:
    # an arm takes hours and a block takes tens of minutes, so the log has to grow
    # while the run is alive for it to be any use on the pod. Nothing is parsed out
    # of it here - the per-block timings come from blocks.json.
    with open(log_path, 'w') as log:
        proc = subprocess.run(argv, cwd=armdir, env=env, stdout=log,
                              stderr=subprocess.STDOUT, text=True)
    with open(log_path) as log:
        tail = log.read().strip().splitlines()[-30:]
    print('\n'.join('      | ' + ln for ln in tail), flush=True)

    # --- postconditions ------------------------------------------------------
    record = read_blocks_json(blocks_json)
    if record is None:
        print(f'{arm}_s{tag}: cace wrote no blocks.json; see {log_path}', flush=True)
        timing['aborted_at'] = 0
        finish(timing, out, dry)
        return timing
    timing['blocks'] = adapt(record.get('blocks') or [], proc.returncode)
    timing['max_gpu_mb'] = record.get('max_gpu_mb')
    timing['steps_per_epoch'] = record.get('steps_per_epoch')
    timing['started'] = t0
    if proc.returncode != 0:
        print(f'{arm}_s{tag}: cace exited {proc.returncode}; see {log_path}',
              flush=True)
        timing['aborted_at'] = len(timing['blocks']) + 1
        finish(timing, out, dry)
        return timing
    bounded = len(timing['blocks']) < nblocks
    if bounded:
        print(f'{arm}_s{tag}: stopped after {len(timing["blocks"])}/{nblocks} '
              f'blocks as asked - a cost probe, NOT a trained model', flush=True)
    else:
        # Every scheduled block ran, so ARM_FINAL is the postcondition. It is
        # checked before the "finished" flag on purpose: a run stopped between the
        # last _fit and the script's own save_model reports finished: false *and*
        # leaves no model, and the model is the part that matters. Checking the
        # flag first would report that arm as a vague flag mismatch and never
        # notice the missing weights.
        final = os.path.join(armdir, ARM_FINAL[arm])
        if not os.path.exists(final):
            print(f'{arm}_s{tag}: all {nblocks} blocks ran but {ARM_FINAL[arm]} is '
                  f'missing - the schedule never reached its final save',
                  flush=True)
            timing['aborted_at'] = nblocks
            finish(timing, out, dry)
            return timing
        if not record.get('finished'):
            print(f'{arm}_s{tag}: {ARM_FINAL[arm]} is present but blocks.json says '
                  f'"finished": false', flush=True)
    finish(timing, out, dry)
    if timing['total_seconds']:
        print(f'  -> {timing["total_seconds"] / 60:.1f} min, '
              f'{timing["total_steps"]} batches, '
              f'{timing["s_per_batch"] * 1000:.1f} ms/batch', flush=True)
    return timing


def adapt(blocks, returncode):
    """The script's blocks.json entries, in run_chain.py's per-block shape.

    ``seconds``/``s_per_batch``/``num_steps`` come from the in-process timing;
    ``label``/``epochs`` are cace's own block identity and are carried through
    because a cace block is an epoch count, not a step count. ``started``/
    ``ended``/``reported_s_per_batch`` have no cace counterpart and stay null.
    """
    out = []
    for i, b in enumerate(blocks, 1):
        rec = {
            'n': i, 'num_steps': b.get('num_steps'), 'seconds': b.get('seconds'),
            's_per_batch': b.get('s_per_batch'), 'reported_s_per_batch': None,
            'started': None, 'ended': None, 'returncode': returncode, 'ok': True,
            'label': b.get('label'), 'epochs': b.get('epochs'),
        }
        out.append(rec)
    return out


def finish(timing, out, dry):
    done = [b for b in timing['blocks'] if b.get('seconds')]
    if done:
        timing['total_seconds'] = round(sum(b['seconds'] for b in done), 2)
        timing['total_steps'] = sum(b['num_steps'] for b in done if b['num_steps'])
        timing['s_per_batch'] = round(timing['total_seconds'] / timing['total_steps'], 6)
    timing['ended'] = stamp()
    if not dry:
        with open(out, 'w') as fh:
            json.dump(timing, fh, indent=2)


def summarize(timings):
    """The cace pair side by side, and the deepmd arms if they have been timed."""
    rows = [t for t in timings if t and t.get('s_per_batch')]
    if not rows:
        return
    print('\n' + '=' * 88)
    print(f'{"chain":20s} {"blocks":>6s} {"batches":>8s} {"minutes":>8s} '
          f'{"ms/batch":>9s} {"peak GPU MB":>11s}')
    for t in rows:
        print(f'{t["arm"] + "_s" + t["replicate"]:20s} '
              f'{len(t["blocks"]):6d} {t["total_steps"]:8d} '
              f'{t["total_seconds"] / 60:8.1f} {t["s_per_batch"] * 1000:9.1f} '
              f'{str(t.get("max_gpu_mb")):>11s}')
    sr = next((t for t in rows if t['arm'] == 'cace-sr'), None)
    lr = next((t for t in rows if t['arm'] == 'cace-lr'), None)
    if sr and lr:
        print(f'\nthe long-range term costs {lr["s_per_batch"] / sr["s_per_batch"]:.2f}x '
              f'the short-range arm ({lr["s_per_batch"] * 1000:.1f} vs '
              f'{sr["s_per_batch"] * 1000:.1f} ms/batch)')
    other = []
    for arm in ('deepmd', 'deepmd-les'):
        for tag in ('A', 'B'):
            p = os.path.join(os.path.dirname(HERE), 'deepmd', 'runs',
                             f'{arm}_s{tag}', 'timing.json')
            rec = read_blocks_json(p)
            if rec and rec.get('s_per_batch'):
                other.append(rec)
    if other:
        print('\nthe deepmd arms on the same data and schedule '
              '(../deepmd/runs/*/timing.json):')
        for t in other:
            print(f'{t["arm"] + "_s" + t["replicate"]:20s} '
                  f'{len(t.get("blocks", [])):6d} {t.get("total_steps"):8d} '
                  f'{(t.get("total_seconds") or 0) / 60:8.1f} '
                  f'{t["s_per_batch"] * 1000:9.1f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--arm', action='append',
                    help='arm to run; repeatable; default both, SR first')
    ap.add_argument('--rep', action='append', help='replicate tag A/B; default both')
    ap.add_argument('--blocks', default='1-',
                    help='which of the script\'s training loops to run, e.g. 1 or '
                         '1-3; a prefix only, since cace has no resume')
    ap.add_argument('--python', default=sys.executable,
                    help='interpreter for the arm script; defaults to this one, '
                         'which avoids the pod\'s PATH problem')
    ap.add_argument('--cpu', action='store_true',
                    help='pin to CPU (CUDA_VISIBLE_DEVICES="" and CACE_DEVICE=cpu)')
    ap.add_argument('--smoke', action='store_true',
                    help='one epoch per block instead of 40/100 - plumbing only, '
                         'NOT the cace schedule; flagged in timing.json')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    env = dict(os.environ)
    if args.cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
        env['CACE_DEVICE'] = 'cpu'
    else:
        env.setdefault('CACE_DEVICE', 'cuda')
        # WSL2: the default caching allocator deadlocks on this host.
        env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env['_SMOKE'] = '1' if args.smoke else ''

    arms = args.arm or ['cace-sr', 'cace-lr']
    reps = args.rep or ['A', 'B']
    timings = []
    # replicate-major, matching run_chain.py, so each (SR, LR) pair completes
    # together and a run cut short leaves whole pairs rather than half of two
    for tag in reps:
        for arm in arms:
            if arm not in ARM_BLOCKS:
                raise SystemExit(f'unknown arm {arm}; known: {list(ARM_BLOCKS)}')
            if tag not in REPLICA_SEEDS:
                raise SystemExit(f'unknown replicate {tag}; known: '
                                 f'{list(REPLICA_SEEDS)}')
            wanted = parse_blocks(args.blocks, ARM_BLOCKS[arm])
            print(f'\n### {arm}_s{tag}: blocks {wanted} of {ARM_BLOCKS[arm]} on '
                  f'{socket.gethostname()} [{env["CACE_DEVICE"]}]', flush=True)
            timings.append(run_arm(args.root, arm, tag, wanted, env, args.dry_run,
                                   args.python))
    summarize(timings)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
