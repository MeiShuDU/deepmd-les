# The cace arms of the water-interface campaign

The two arms here are cace's own published scripts for `fit-water-interface`, run unchanged except for the edits listed below, so that their wall clock can be compared with the deepmd arms in `../deepmd/` on the same data, the same schedule and the same GPU.

| arm | source | what it is | blocks | epochs |
| --- | --- | --- | --- | --- |
| `cace-lr` | `fit-interface-mp0/fit-cace-nnp.py` | CACE + learned charge + Ewald (`CombinePotential([sr, lr], [1.0, 0.02])`) | 8 | 500 |
| `cace-sr` | `fit-interface-mp0-sr/fit-cace-nnp.py` | the same CACE, short range only | 7 | 400 |

`fit-interface-mp1` and `fit-interface-mp1-sr` are not mirrored: they set `num_message_passing=1`, which the `se_a` descriptor the user specified has no analogue for.

## Layout

```
cace-lr/fit-cace-nnp.py      the arm, adapted from the datarepo (see below)
cace-sr/fit-cace-nnp.py
data/slab-fps-n-500.xyz      the 500-frame benchmark, staged (md5 87ab8e3fb4d03320233c84ce6cd8980f)
runs/cace-lr_s{A,B}/         one directory per arm-replicate; the arm script's CWD
run_cace_chain.py            runs an arm and writes timing.json
```

Everything the arm script reads and writes is relative to its CWD, which is the run directory.
That is what keeps the datarepo's own `fit-interface-*/` directories and their published `model-{1,2,3}.pth` untouched.

## Why the datarepo alone cannot run

Two things are dangling in `cace-lr-fit-datarepo`, both reported to the user and neither fixable inside the datarepo:

* `fit-cace-nnp.py` line 5 does `sys.path.append('../cace/')` and line 26 reads `../waterslab-datasets/slab-fps-n-500.xyz`.
  Neither sibling directory was ever published, so the `cace` package comes from the installed environment and the xyz is staged at `data/`.
  The `sys.path.append` line is left as written: from a run directory it resolves to a path that does not exist and is silently a no-op, and removing it would be a gratuitous diff.
* `fit-interface-mp0-sr/fit-cace-nnp.py` line 27 is the literal `valid_fraction=VVVVV`, which is not valid Python, so that arm cannot run as shipped.

## The edits, hunk by hunk

`diff -u` against the datarepo sources gives nine hunks for `cace-lr` and eight for `cace-sr` (+75/-6 lines in both, against 237 and 224 source lines), all of them one of the following.
Nothing else in either script is touched: the net, the optimiser, the scheduler, the loss weights, the 40/100-epoch block structure and the 500/400-epoch total are the authors'.

1. **Provenance header** (lines 3-17).
   Records which source the file came from and how to invoke it, since the file is now one of eight campaign arms and its own directory no longer implies its provenance.
2. **`import json`, `import os`, `import time`** after `import logging`.
   Needed by the helpers below.
3. **The seed/smoke/stop/timing helper block**, inserted after `torch.set_default_dtype(torch.float32)` and therefore before every random draw.
   `SEED` comes from the one positional argument and is passed to `torch.manual_seed`; cace's `TrainingTask` takes no seed (`cace/tasks/train.py:17-52`) and builds its net from torch's global RNG, so this is the only place a replicate can be pinned.
   `_STOP_AFTER` reads `CACE_STOP_AFTER`, which is how `run_cace_chain.py --blocks` bounds a process that has no resume.
   `PH()` turns each block's 40 or 100 epochs into 1 under `--smoke`, and `_dump()` rewrites `blocks.json` after every block so a run that dies still has its timings.
4. **`train_path='../waterslab-datasets/...'` -> `'../../data/slab-fps-n-500.xyz'`** (line 26 of the source).
   The staged copy, byte-identical to the datarepo's.
5. **`valid_fraction=VVVVV` -> `valid_fraction=0.1`** (line 27, `cace-sr` only).
   The authors' own intent, unrecoverable from the file otherwise; `0.1` is what `fit-interface-mp0` and the md arms use, and it is the split the deepmd arms were given.
6. **`use_device = 'cuda'` -> `os.environ.get('CACE_DEVICE', 'cuda')`** (line 45 of the source).
   cace's `init_device('cuda')` asserts CUDA exists (`cace/tools/torch_tools.py:45`), so with `CUDA_VISIBLE_DEVICES=""` the arm dies rather than falling back to CPU; the campaign's local smoke needs CPU.
   The default is unchanged, so on the pod the arm behaves exactly as published.
7. **The phase-0 `task.fit(...)` inside `for i in range(5)` -> `_fit('phase0.%d' % i, task, 40)`**.
8. **The `task.fit(..., epochs=100, ...)` calls -> `_fit('w_e=1' | 'w_e=10' | 'w_e=1000', task, 100)`** (two in `cace-sr`, three in `cace-lr`).
   `_fit` calls the identical `task.fit` with the identical arguments and only wraps it in a timer, so the training is bit-identical; it exists because cace's per-block wall clock is the campaign's primary measurement.
9. **`_dump(finished=True)`** after the final `trainable_params` log line, so `blocks.json` says whether the 500/400-epoch schedule actually completed.

The one thing the campaign cannot reproduce is cace's checkpointing, because cace's checkpointing is not resumable.
`TrainingTask.fit` writes `checkpoint.pt` every 10 epochs (`cace/tasks/train.py:261-262`), but `checkpoint()` (`:284`) records no `global_step`, and the epoch counter is exactly what the `StepLR` schedule and `warmup_steps` run on, so the restore path that does exist (`load_state_dict`, `:293`) could not put the schedule back where it left off.
The published arm scripts never call it in any case, so a crashed arm restarts from epoch 0 and the pod plan has to budget for that.
Since cace also cannot be stopped and resumed, a partial run is a cost probe and never a trained model: `--blocks` bounds it from inside and the resulting `blocks.json` says `"finished": false`.

## Running

```bash
python run_cace_chain.py --cpu --smoke --blocks 1     # cheapest plumbing check
python run_cace_chain.py --arm cace-lr --rep A        # one arm, one replicate
python run_cace_chain.py                              # both arms x both replicates
python run_cace_chain.py --dry-run                    # print the commands only
```

`--smoke` is not the cace schedule and says so in `timing.json`; it exists so the plumbing can be exercised in minutes.
`--blocks` takes a prefix only, because cace cannot resume: a bounded run is a cost probe whose `blocks.json` has `"finished": false`.
A completed arm is skipped on a second invocation, and a partial one is restarted from scratch with a warning.
`--python` defaults to the interpreter running the runner, which sidesteps the pod's `PATH` problem for a non-interactive shell.
