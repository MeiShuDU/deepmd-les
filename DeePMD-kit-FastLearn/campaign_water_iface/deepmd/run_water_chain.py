"""Run the generated water-interface chains, one block at a time, and time them.

Each arm is a chain of `dp --pt train` runs (``gen_inputs_water.py`` writes the
configs and ``chain.sh``), each block started from the previous block's final
checkpoint with ``--init-model``. That flag is what makes the block a *fresh
task* in cace's sense: it reloads the weights but rebuilds the optimiser and
restarts the step counter at 0, so the block's own LR schedule and the epoch
counter begin where the config says they do. Why not ``--restart`` is argued in
``gen_inputs_water.py``'s docstring and recorded in ``segments.json``.

This runner exists rather than a shell ``for`` loop for three reasons the user
asked for and one they did not:

* timing. Every block's wall clock, and the per-batch figure derived from it, is
  written into ``timing.json`` in the arm directory as soon as the block ends, so
  a chain that dies mid-way still has the timing for the blocks that ran.
  deepmd's own ``average training time: X s/batch`` line is parsed as a
  cross-check.
* sequencing. Blocks run one at a time, in order, never concurrently, and the
  runner refuses to start a block whose ``--init-model`` checkpoint is missing -
  a chained schedule is worthless if a block silently restarts from scratch.
* a postcondition. After each block the expected final checkpoint
  (``model.ckpt-{num_steps}.pt``, ``pt/train/training.py:1150``) must exist, or
  the chain stops there instead of feeding a half-trained model to the next
  block.

The fourth reason: the arms are meant to be compared under the same wall clock,
so the timing is a result of this run and not a by-product.

Usage:
    python run_water_chain.py --blocks 1-2 --cpu            # smoke, both arms
    python run_water_chain.py --arm deepmd-les --rep A
    python run_water_chain.py                               # all four chains, sequentially
    python run_water_chain.py --dry-run
"""
import argparse
import datetime
import json
import os
import re
import shlex
import socket
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(HERE, 'runs')
ARMS = ['deepmd', 'deepmd-les']
REPS = ['A', 'B']
REPORTED = re.compile(r'average training time:\s*([0-9.]+)\s*s/batch')


def stamp():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def parse_blocks(spec, nseg):
    """'1-8' / '1,2,5,6' / '3-' / 'all' -> the block numbers to run."""
    if spec in (None, '', 'all'):
        return list(range(1, nseg + 1))
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-', 1)
            lo = int(lo) if lo else 1
            hi = int(hi) if hi else nseg
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    unknown = [n for n in out if not 1 <= n <= nseg]
    if unknown:
        raise SystemExit(f'block(s) {unknown} outside 1..{nseg}')
    return out


def load_arm(root, arm, tag):
    armdir = os.path.join(root, f'{arm}_s{tag}')
    with open(os.path.join(armdir, 'segments.json')) as fh:
        return armdir, json.load(fh)


def prev_steps(record, n):
    """numb_steps of the previous block, which is the name of its final ckpt."""
    return {s['n']: s for s in record['segments']}[n - 1]['num_steps']


def train_command(record, n, blockdir, dp):
    """The `dp --pt train` argv for one block, and its --init-model source."""
    argv = [dp, '--pt', 'train', 'input.yaml']
    init = None
    if n > 1:
        init = os.path.join('..', f's{n - 1}', f'model.ckpt-{prev_steps(record, n)}.pt')
        argv += ['--init-model', init]
    return argv, init


def run_block(armdir, record, seg, env, dry, timing, dp):
    n = seg['n']
    blockdir = os.path.join(armdir, f's{n}')
    argv, init = train_command(record, n, blockdir, dp)
    final = os.path.join(blockdir, f'model.ckpt-{seg["num_steps"]}.pt')
    # a dry run is a plan, so the checkpoint a later block needs has not been written
    # yet and must not be reported as missing
    if init is not None and not dry and not os.path.exists(os.path.join(blockdir, init)):
        print(f'  s{n}: REFUSING - --init-model {init} is missing', flush=True)
        return None
    if os.path.exists(final):
        print(f'  s{n}: already complete ({os.path.basename(final)} exists), skipping',
              flush=True)
        done = next((b for b in timing['blocks'] if b['n'] == n), None)
        if done is not None:
            return done
        return {'n': n, 'num_steps': seg['num_steps'], 'seconds': None,
                's_per_batch': None, 'reported_s_per_batch': None,
                'started': None, 'ended': None, 'returncode': 0, 'skipped': True}
    print(f'  s{n}: {" ".join(shlex.quote(a) for a in argv)}'
          + (f'   (in {blockdir})' if dry else ''), flush=True)
    if dry:
        return None
    t0, started = time.perf_counter(), stamp()
    log_path = os.path.join(blockdir, 'train.log')
    with open(log_path, 'w') as log:
        proc = subprocess.run(argv, cwd=blockdir, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        log.write(proc.stdout)
        tail = proc.stdout.strip().splitlines()[-25:]
        print('\n'.join('      | ' + ln for ln in tail), flush=True)
    seconds = time.perf_counter() - t0
    rec = {
        'n': n, 'num_steps': seg['num_steps'], 'seconds': round(seconds, 2),
        's_per_batch': round(seconds / seg['num_steps'], 6),
        'reported_s_per_batch': None, 'started': started, 'ended': stamp(),
        'returncode': proc.returncode,
    }
    m = REPORTED.search(proc.stdout)
    if m:
        rec['reported_s_per_batch'] = float(m.group(1))
    if proc.returncode != 0:
        print(f'  s{n}: dp exited {proc.returncode}; see {log_path}', flush=True)
        rec['ok'] = False
        return rec
    if not os.path.exists(final):
        print(f'  s{n}: dp succeeded but {os.path.basename(final)} is missing',
              flush=True)
        rec['ok'] = False
        return rec
    rec['ok'] = True
    print(f'  s{n}: {seconds:.1f} s for {seg["num_steps"]} steps '
          f'({rec["s_per_batch"] * 1000:.1f} ms/batch)', flush=True)
    return rec


def run_chain(root, arm, tag, blocks, env, dry, dp):
    armdir, record = load_arm(root, arm, tag)
    if record.get('smoke'):
        print(f'{arm}_s{tag}: SMOKE configs (not the cace schedule)', flush=True)
    timing = {
        'arm': arm, 'replicate': tag, 'host': socket.gethostname(),
        'device': 'cpu' if env.get('CUDA_VISIBLE_DEVICES') == '' else 'gpu',
        'smoke': bool(record.get('smoke')),
        'started': stamp(), 'ended': None, 'blocks': [], 'total_seconds': None,
        'total_steps': None, 's_per_batch': None, 'aborted_at': None,
    }
    out = os.path.join(armdir, 'timing.json')
    segs = {s['n']: s for s in record['segments']}
    for n in blocks:
        rec = run_block(armdir, record, segs[n], env, dry, timing, dp)
        if rec is None and not dry:
            timing['aborted_at'] = n
            break
        if rec is not None:
            timing['blocks'].append(rec)
            timing['ended'] = stamp()
            with open(out, 'w') as fh:
                json.dump(timing, fh, indent=2)
        if rec is not None and rec.get('returncode') not in (0, None):
            timing['aborted_at'] = n
            break
        if rec is not None and rec.get('ok') is False:
            timing['aborted_at'] = n
            break
    done = [b for b in timing['blocks'] if b.get('seconds')]
    if done:
        timing['total_seconds'] = round(sum(b['seconds'] for b in done), 2)
        timing['total_steps'] = sum(b['num_steps'] for b in done)
        timing['s_per_batch'] = round(timing['total_seconds'] / timing['total_steps'], 6)
    timing['ended'] = stamp()
    if not dry:
        with open(out, 'w') as fh:
            json.dump(timing, fh, indent=2)
    if timing['total_seconds']:
        print(f'  -> {timing["total_seconds"] / 60:.1f} min, '
              f'{timing["total_steps"]} steps, '
              f'{timing["s_per_batch"] * 1000:.1f} ms/batch', flush=True)
    return timing


def summarize(timings):
    """A cross-arm table, so the cost of the pair is one glance and not four files."""
    rows = [t for t in timings if t]
    if len(rows) < 2:
        return
    print('\n' + '=' * 78)
    print(f'{"chain":22s} {"steps":>7s} {"minutes":>8s} {"ms/batch":>9s} '
          f'{"deepmd ms/batch":>15s}')
    for t in rows:
        rep = [b['reported_s_per_batch'] for b in t['blocks']
               if b.get('reported_s_per_batch')]
        # a dry run has no numbers to print; the table still shows the plan
        steps = 'n/a' if t['total_steps'] is None else f'{t["total_steps"]:7d}'
        mins = 'n/a' if t['total_seconds'] is None else f'{t["total_seconds"] / 60:8.1f}'
        ms = 'n/a' if t['s_per_batch'] is None else f'{t["s_per_batch"] * 1000:9.1f}'
        dms = (f'{sorted(rep)[len(rep) // 2] * 1000:15.1f}' if rep else 'n/a'.rjust(15))
        print(f'{t["arm"] + "_s" + t["replicate"]:22s} {steps:>7s} {mins:>8s} '
              f'{ms:>9s} {dms:>15s}')
    sr = next((t for t in rows if 'les' not in t['arm']), None)
    lr = next((t for t in rows if 'les' in t['arm']), None)
    if sr and lr and sr['s_per_batch'] and lr['s_per_batch']:
        print(f'\nthe long-range term costs '
              f'{lr["s_per_batch"] / sr["s_per_batch"]:.2f}x the short-range run '
              f'({lr["total_seconds"] / 60:.1f} vs {sr["total_seconds"] / 60:.1f} min '
              f'for {sr["total_steps"]} vs {lr["total_steps"]} steps)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--arm', action='append',
                    help='arm to run; repeatable; default both, SR first')
    ap.add_argument('--rep', action='append', help='replicate tag; default both')
    ap.add_argument('--blocks', default=None,
                    help="block range, e.g. '1-2'; default every block of the arm")
    ap.add_argument('--dp', default='dp',
                    help='the dp entry point; on the pod the venv is not on a '
                         'non-interactive PATH, so pass /root/venv310/bin/dp')
    ap.add_argument('--cpu', action='store_true',
                    help='pin to CPU (CUDA_VISIBLE_DEVICES=""); for the local smoke')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    env = dict(os.environ)
    if args.cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        # WSL2: the default caching allocator deadlocks on this host.
        env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    arms = args.arm or ARMS
    reps = args.rep or REPS
    timings = []
    # replicate-major, so each (SR, LES) pair completes together and a run that is
    # cut short still leaves whole pairs to compare rather than half of two
    for tag in reps:
        for arm in arms:
            _, record = load_arm(args.root, arm, tag)
            blocks = parse_blocks(args.blocks, len(record['segments']))
            print(f'\n### {arm}_s{tag}: blocks {blocks} '
                  f'({sum(s["num_steps"] for s in record["segments"] if s["n"] in blocks)} '
                  f'steps) on {socket.gethostname()} [{env.get("CUDA_VISIBLE_DEVICES", "gpu")}]',
                  flush=True)
            timings.append(run_chain(args.root, arm, tag, blocks, env, args.dry_run,
                                     args.dp))
    summarize(timings)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())