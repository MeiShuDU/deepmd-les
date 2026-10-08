# HPC handoff: deepmd-versioned LES against cace-versioned LES

Audience: a Claude Code session on the HPC with this repository checked out and no memory of the earlier work.
Read this end to end before starting a run.
Sections 2 and 6 are the ones that prevent wasted GPU hours; section 1 says what is actually still open.

Every claim below was verified on the originating host unless it says otherwise.
For the model-comparison numbers, `LES_PERFORMANCE.md` is authoritative and this file only summarises it.

## 1. The goal, and the one thing still open

The LES (Latent Ewald Summation) library was integrated into a custom build of DeepMD-kit 3.1.2 (PyTorch backend) as a new model type `hybrid_ener`, which adds learned-charge long-range electrostatics to the ordinary short-range DeepMD model.

The question being answered is whether that integration is as good as the original cace-versioned LES, and the answer must come from actual end-to-end training in BOTH codes on the SAME dataset.
A design argument is not accepted as an answer.

Raw LES kernel agreement between the two implementations already PASSED (`tests/hybrid_ener/raw_les_vs_cace.py`: the Ewald energies of the two implementations agree, no cace dependencies beyond a single module).
So the kernel is not the open question.

What remains is the training comparison:

- **#21, the cace side.** The deepmd side is validated end to end but only to 2000 steps (section 5); the cace side has only been smoke-tested, and the reference to compare against is the author's published `best_model.pth`.
- **#22, the comparison.** Per-atom energy and force RMSE / MAE / R2 for both codes on the shared validation split, plus LES diagnostics (latent-charge statistics, E_lr share of the total energy, net-charge conservation) and charge consistency between the two codes.

## 2. What has to move, and what git will not give you

Three trees, one of which is not in this repository at all:

| tree | what it is | how to move it |
|---|---|---|
| this repo (`deepmd-les/`) | modified deepmd-kit 3.1.2 + les + the FastLearn runs | `git clone`, or rsync the directory |
| `/root/app/cace` | the cace package, its own git repo (HEAD `3f1a665`) | clone, then `pip install -e .` |
| `DeePMD-kit-FastLearn/cace_compare_e2e/` | the shared dataset, the author's cace reference, the deepmd cace-side configs | **rsync this one explicitly** |

`cace_compare_e2e/` is **untracked** (70 MB); a `git clone` will silently leave it behind, and it is the whole point of the exercise.
It contains `cace/water.xyz` (35 MB, the cace author's 64-H2O benchmark: 1593 frames of 192 atoms), `cace/best_model.pth` (the author's trained reference), `prep_water.py` (which regenerates the deepmd-side `water/{train,valid}` systems from `water.xyz`), and the two deepmd configs `deepmd/input.json` and `deepmd/input_w001.json`.

Training checkpoints are deliberately not in git: 72 files totalling 1.5 GB under `01.train/rerun/extended/`, held out by `01.train/rerun/.gitignore`.
The `*.tsv` metric files **are** tracked (`sweep_valid.tsv`, `sweep_valid_d7500.tsv`, `lrphase_*.tsv`, `full_eval*.tsv`, `bias_frames*.tsv`), so the notebook analysis is reproducible from a clone with no retraining; only new training needs checkpoints.

## 3. Environment

On the originating host: conda env `py39`, python 3.9.23.
The HPC pod (`cpod-1vg1a4d7wa0n`) was set up on python 3.10.12 in `/root/venv310`, and the procedure below is the one **actually verified there** (not the `.[torch]` form, which builds the C++ ops this backend does not need).

```bash
# deepmd-kit: both backends off, so the CMake/C++ op build is skipped entirely.
# The originating host also has ENABLE_CUSTOMIZED_OP=False (deepmd.pt is pure
# Python and falls back cleanly without deepmd_op_pt), so this matches it.
cd deepmd-kit-3.1.2
DP_ENABLE_TENSORFLOW=0 DP_VARIANT=cpu pip install -e .

cd ../les
pip install -e . --no-deps     # else it pulls deepmd-kit[torch] from PyPI over this checkout

cd /root/app/cace
pip install -e . --no-deps     # CACE declares numpy<2 / ase<=3.22.1, which would downgrade
                               # numpy and break deepmd; the originating host also violates
                               # those pins and that is the combination that produced the results
```

Runtime deps to install first: `numpy scipy pyyaml 'dargs>=0.4.7' h5py wcmatch packaging ml_dtypes mendeleev array-api-compat ase`, plus `matscipy` for CACE.
The pip dependency-resolver warning about `cace requires numpy<2` is expected and safe to ignore.

This is **not stock deepmd-kit**.
`hybrid_ener` exists only in `deepmd-kit-3.1.2/` in this repo, and both packages are installed editable so source edits take effect immediately.
Installing a released deepmd-kit from PyPI would silently drop the model type.
`hpc_setup.sh` in this directory automates the whole block.

Notes that cost time to learn:

- `dp --pt train <cfg>` and `dp --pt test <ckpt>` need the `--pt` flag (PyTorch backend).
- The `les` package writes its log to a **hardcoded `les.log` in the current working directory**, so never run two LES trainings from one directory. The harness gives every run its own subdirectory for exactly this reason; keep that property.
- `dp --pt test` cannot load an unfrozen hybrid `.pt` at all, see section 6.

## 4. Absolute paths

The pod was set up with the trees at the **same absolute paths** as the originating host:

- repo: `/root/app/deepmd-les/deepmd-les`
- cace: `/root/app/cace`

This was deliberate: it means every hardcoded absolute path in the scripts still resolves, so the transferred code is byte-identical with zero edits and there is no transfer-induced divergence to explain away.
`grep -rn /root/` finds the paths below; they need no rewriting **as long as the trees stay at those paths**.

| file | line | what | note |
|---|---|---|---|
| `01.train/rerun/gen_extended.py` | 44 | `DATA = "/root/app/.../DeePMD-kit-FastLearn/data"` | resolves as-is |
| `01.train/rerun/check_frozen.py` | 19 | `data_3` path | resolves as-is |
| `01.train/rerun/check_virial.py` | 37 | `data_3` path | resolves as-is |
| `01.train/rerun/check_serialize.py` | 27 | `data_3` path | resolves as-is |
| `01.train/rerun/sweep_valid.sh` | 15 | `source /root/miniconda3/etc/profile.d/conda.sh` | **must change on the pod**: it uses `/root/venv310`, not conda |
| `tests/hybrid_ener/raw_les_vs_cace.py` | 36, 39, 46 | the cace `ewald.py` source and `benchmark/liquid-64.xyz` | resolves as-is (verified running on the pod) |

If you ever relocate a tree, rewrite those paths; otherwise leave them.

`hybridles_model.py`, `hybridles.py` and the `les/` tree have no absolute paths and relocate cleanly.
The cace-side configs use relative paths (`"systems": ["../water/train"]`) and are run from `DeePMD-kit-FastLearn/cace_compare_e2e/deepmd/`; `fit-cace-nnp.py` likewise reads `../water.xyz` and is run from `cace_compare_e2e/cace/`.

## 5. The remaining work, concretely

### #21, cace side

Two options, and the choice is the user's:

- **(a)** run the author's published reference, `cace_compare_e2e/cace/fit-cace-nnp.py`, and compare deepmd against that. This is the honest external baseline (43,052 params, CombinePotential weight 0.01, learned charges only).
- **(b)** rebuild `train_cace.py` against `water.xyz` so both sides share hyperparameters. **`train_cace.py` is STALE**: it points at a different 320-frame dataset, reads a different energy-key convention, and implements neither the author's combine weight nor the learned-charge-only design. Rebuild it or drop it; do not run it as-is.

Two earlier findings drive the design, do not re-derive them:

- **The fixed-charge baseline is dead.** A fixed `O -1.0` / `H +0.5` baseline applied at weight 1.0 gives `E_lr = -100,885` to `-102,608` eV/frame (`|E_lr/E_ref|` about 7300), because intramolecular O-H pairs are not excluded from the Ewald sum. Both sides therefore use **learned charges only**.
- **`output_scaling_factor` defaults to 0.1 in `les`, and Ewald energy is quadratic in `q`, so deepmd's native `E_lr` weight is already 0.01** - exactly the cace benchmark's `CombinePotential` weight. Confirmed by measurement: `E_lr(osc 0.1) / E_lr(osc 1.0) = 0.010000`. The current deepmd configs set `output_scaling_factor: 1.0` so `q` is raw and comparable to cace's `q`, and control the `E_lr` weight independently through `lr_weight`: `input.json` is 1.0 (full strength) and `input_w001.json` is 0.01 (cace-matched). `lr_weight` is read in `HybridLESModel.__init__` and applied as `E_lr_total * lr_weight` in `forward` before the autograd forces, so `F_LR` and the virial scale with it automatically.

The two runs on disk in `cace_compare_e2e/deepmd/run_lrw1` and `run_lrw001` are `numb_steps: 2000` feasibility runs, not the comparison.
**The comparison length is an open decision.** Pick it deliberately and use the same `numb_steps`, the same LR schedule and the same batch size on both sides; otherwise the comparison inherits the schedule confound documented in `LES_PERFORMANCE.md`, which is the single most expensive lesson from the earlier work.

### #22, the comparison

Metrics: per-atom energy and force RMSE / MAE / R2, on the shared validation split, with the same metric definition in both codes.
LES diagnostics: latent-charge statistics, `E_lr` share of total energy, net-charge conservation, and `q` consistency between the two implementations.
For the deepmd side, use the full-set evaluator (section 6), never the logged curve.

## 6. Measurement rules, which are not optional

These were each learned from a wrong result, and the wrong result is on the record in `LES_PERFORMANCE.md`.

**1. The `rmse_*_val` columns in `lcurve*.out` are not full-set values.**
`validation_data` uses `batch_size: 1` / `numb_btch: 3`, so every logged point is 3 randomly drawn frames out of 80. Measured against the same checkpoint's exact value, a 3-frame draw is a median 2% to 12% low but spans 2x to 5x from the 5th to the 95th percentile.
Never rank arms on a logged point.

**2. Consecutive logged points come from different checkpoints.**
A window of the logged curve therefore cannot be compared against one checkpoint's exact value; that comparison looks like a contradiction and is not one.

**3. Evaluate the whole split per checkpoint, and average several late checkpoints.**
`01.train/rerun/eval_full.py` does this, driven by `sweep_valid.sh`. On the full set the model's own energy RMSE still swings by 1.1x to 2.3x between adjacent late checkpoints in EVERY arm including the LES-free ordinary one, while force varies under 1%. A single checkpoint is not a converged value.

**4. The energy swing is the energy zero drifting, not physics.**
Split `rmse_e^2 = bias^2 + spread^2` (bias is the mean per-frame signed error, spread its std). Across late checkpoints the spread is flat to under 1% while the bias moves about 5x and can flip sign even in the ordinary arm. Forces are blind to a per-frame energy constant (`F = -dE/dr`), which is exactly why `rmse_e` jags while `rmse_f` stays smooth. An energy ranking read off `rmse_e` is reading a training convention, not the physics; the force ranking is unaffected. `diag_bias.py <split> <run>` regenerates that table.

**5. Never quote the LES force advantage without naming `decay_steps`.**
`LearningRateExp` is piecewise constant, not a ramp: `decay_steps` is the length of the full-rate phase, so with `start_lr 1e-3 / stop_lr 3.51e-8 / numb_steps 10000` the cliff sits at step 5000 (`1e-3` then `5.93e-6`) or 7500 (`1e-3` then `4.56e-7`).
The `decay_steps 7500` experiment is a **convergence probe, not an accuracy comparison**: it buys 2500 more full-rate steps but ends the run at an effective standstill, so the model freezes rather than converges, and its metrics mostly measure the freeze. No accuracy conclusion follows from it in either direction, and neither schedule is one anyone would ship.

**6. Prove provenance before comparing two run sets.**
The `source commit:` line in the run logs is stale: it comes from a `_version.py` frozen at install time and is never regenerated, so every run on this host prints the same value regardless of what the source actually was. Instead, check that the initial LES weight norms printed at forward 1 match, and that the first `lcurve` rows are bit-identical while the learning rate is shared.

**7. Use `no_jit=True` for hybrid checkpoints, or freeze first.**
`dp --pt test` calls `torch.jit.script` on the reconstructed model and fails on an unfrozen hybrid `.pt`: `element_numbers` is registered `persistent=False` in `hybridles.py:64-74`, TorchScript discards that flag, so the scripted model expects a key the checkpoint does not have (`Missing key(s): model.Default.atomic_model.element_numbers`). Use `no_jit=True` (what `eval_full.py` does) or `dp freeze` first and load the `.pth`.
The docstring in `eval_full.py` gives a stale reason for this (f-strings in the forward path); scripting works now, the real cause is that key.

## 7. HPC-specific changes

### The pod as it actually stands (verified 2026-09-19)

`ssh hpc` reaches `cpod-1vg1a4d7wa0n.podtcp.compshare.cn:24329` as root.

| what | value |
|---|---|
| python | `/root/venv310/bin/python`, 3.10.12 (not conda) |
| repo | `/root/app/deepmd-les/deepmd-les` |
| cace | `/root/app/cace` |
| GPU | RTX 4090, 24564 MiB, driver 610.57.04 |
| torch | 2.8.0+cu128, `cuda.is_available() True` |
| numpy | 2.2.6 (the originating host has 2.0.2 - harmless, see below) |
| `ENABLE_CUSTOMIZED_OP` | `False`, same as the originating host |

The numpy difference is provably harmless: with a fixed seed the smoke build prints `E_pred=-62.629262` on **both** hosts, identical to the last digit, so the whole descriptor -> LES -> Ewald -> `E_total` forward and the backward agree across python 3.9/numpy 2.0.2 and python 3.10/numpy 2.2.6.

`hpc_setup.sh` in this directory rebuilds the venv from scratch if needed.

### Operating notes

- **`run_extended.sh` is strictly sequential** because the originating host had a single 8 GB GPU. On the HPC, run one arm per GPU with `CUDA_VISIBLE_DEVICES`; this is safe because each run already owns its own subdirectory, so checkpoints, `lcurve` and `les.log` never collide.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a WSL2 workaround.** The CUDA caching allocator otherwise hoarded its high-water mark (the neighbour list grows over training), climbing to 7.9 GB of 8.19 GB and deadlocking in `dxgkrnl`. With it the footprint stays flat at about 5.4 GB. Keep it if the HPC node shows similar behaviour; on a normal node it is untested but harmless.
- **`sweep_valid.sh` runs one `(run, step)` per process**, because loading several models into one CUDA context tripped an allocator assert (`c10 CUDACachingAllocator "!handles_.at(i)"`) on the originating host, and it deliberately does not set `expandable_segments` (implicated in that assert). If the HPC does not reproduce this, batching is an optimisation, not a correctness requirement.
- The `nvidia-smi` sampler loop in `run_extended.sh` exists purely for the WSL2 stall investigation and can be dropped.
- Both shell scripts use `set -u` and no `set -e`, so a failed arm is reported rather than aborting the batch. **Grep the log for `!!! FAILED`** after every batch; a silent partial batch is the easiest way to compare an incomplete set.
- `sweep_valid.sh` activates conda through a hardcoded path (section 4); change it to `/root/venv310` on the pod.
- `dp --pt train` does **not** accept `--numb_steps`; to shorten a run, edit `numb_steps` in the JSON. There is no CLI override for it.
- **A stopped pod can only start when a 4090 is free, and the platform is shared.** On 2026-09-19 all 4090 stock in `cn-bj2` was taken (see `./hpc_compshare.py gpu`, which labels each zone), so start failed with `226604 This GPU type is currently out of resources`. `ensure --retry M` re-issues the start once a minute for M minutes; a start is idempotent, so this is safe to leave running.

### Reaching the pod from WSL2

The originating client is WSL2 with `networkingMode=mirrored`.
That mirrors the Windows adapter **including its 9000-byte jumbo MTU**, which the virtual interface cannot actually carry, and the ICMP that would report the failure is blocked, so oversized segments are silently dropped.
Symptom: `ssh` hangs forever at `expecting SSH2_MSG_KEX_ECDH_REPLY` (the default `sntrup761x25519-sha512` key exchange puts a ~1.2 KB share in one packet), and `rsync` stalls mid-transfer.
Fix: `sudo ip link set dev eth1 mtu 1400` before connecting. Verify with a bulk transfer such as `ssh hpc 'head -c 5000000 /dev/zero | base64' | wc -c`, which returns nothing without the fix and ~6.75 MB with it.
The fix does **not** survive a WSL restart, so it is applied at boot by `/etc/systemd/system/wsl-fix-mtu.service`, which runs `/usr/local/sbin/wsl-fix-mtu.sh --wait 30`.
The script only touches interfaces that are `up` **and** carry a jumbo MTU (above 1500), so ordinary 1500-byte interfaces and the mirrored `loopback0` are left alone.
If you see this symptom, do not "fix" it by forcing a smaller `KexAlgorithms` alone - that gets the handshake through but data transfer still stalls.

### Provisioning the pod through the API

`hpc_compshare.py` in this directory drives the compshare (UCloud) API directly, so the pod can be listed, started, stopped and logged into without the web console.
It uses `ucloud-sdk-python3`, kept in its own venv (`/root/venv-compshare`) so it cannot perturb the deepmd/cace environment; the shebang points there, so run it as `./hpc_compshare.py`.

Config is split so that no secret is ever committed:

| file | contents |
|---|---|
| `~/.compshare/credentials.json` (mode 600) | the API key pair from https://console.compshare.cn/uaccount/api_manage |
| `compshare_spec.json` (in this directory) | region, zone, image, GPU type, CPU, memory, disk, ssh alias |

The spec holds the live instance's real values, read back from the API rather than guessed: `Region cn-bj2`, `Zone cn-bj2-03`, image `compshareImage-1t1vuerl80np` (`cuda132_torch2130_py312`), 1x4090, CPU 14, 65536 MB, a 50 GB boot disk and `ChargeType Postpay`.
Note the pod's `Name` is `host`, which is what the console defaulted to; `UHostId` pins it, so every command with no `<ref>` targets it regardless.

The usual sequence is a single command:

    ./hpc_compshare.py ensure --retry 30

which starts the pod if it is stopped, waits until it is reachable, installs `~/.ssh/id_rsa.pub`, and rewrites the `Host hpc` block in `~/.ssh/config` from the live host and port.
The create call has no SSH-key field (only `LoginMode` + `Password`), so a password is used exactly once to install the key - and because the API hands that password back base64-encoded, nothing needs to be stored for a pod created this way.
Other commands: `list`, `show`, `gpu`, `price`, `images`, `types`, `ssh-config`, `wait`, `start`, `stop`, `create --yes`, `terminate --yes`.

`gpu` is the one to reach for when a start fails, because the platform is shared and a card can be temporarily gone:

    Exclusive:
      5001 (cn-bj2-03)                 nothing free
      10027 (cn-wlcb-01)               {'A100': 2, 'A800': 5, 'H20': 2, 'V100S': 25}

Two API quirks are worked around here and should not be mistaken for bugs in the script:

- `describe_comp_share_instance` returns an empty `SshLoginCommand` while the pod is stopped, and orders its `IPSet` with the `Private` entry (`cpod-...`, the pod id, not routable) before the `Bgp` one. `ssh_endpoint` therefore prefers the `Bgp` host and falls back to `ssh_port` in the spec.
- The generated schema for `DescribeCompShareGpuInventory` declares `GpuInventoryByZone`, but the live API returns `GpuInventory`; the SDK's `loads` silently drops the undeclared field, so that one call reads the raw payload (`raw_call`).

Rewriting `~/.ssh/config` from live state was verified to reproduce the hand-written block **byte for byte** and to be idempotent.

## 8. Already verified, do not redo

### Verified on the pod (2026-09-19)

- **The whole `tests/hybrid_ener/` harness (6 scripts) passes on the pod**, identically to the originating host. Every deterministic quantity is bit-identical across hosts: `raw_les_vs_cace.py` prints the same Ewald energies to the last digit (`E_A(les) = E_B(cace) = -3.032161185222e+00` for TIP3P, `-1.136641125546e-01` for random q) and the same unit ratio `90.475600`; `check_virial.py` prints the same `LR contribution = 7.135644e-03`; `check_lr.py` the same `descriptor-response contribution max = 2.729e-02`. Only the harness's unseeded random-init quantities differ, which is expected.
- **`smoke_build.py` prints `E_pred=-62.629262` on both hosts**, identical to the last digit once the seed is fixed. That is the full end-to-end equivalence proof: same descriptor, same LES latent charges, same Ewald sum, same `E_total`, and the same backward (40 params with grad).
- **`dp --pt train` runs on the pod's 4090** and writes a checkpoint, with the `[HybridLES]` SR/LR decomposition and `[LES-grad]` per-parameter gradient logs printing normally.
- **A 100-step training run is bit-identical across hosts.** Same config (`cace_compare_e2e/deepmd/input.json`, `numb_steps` 100) on both: step 100 gives `trn rmse_e = 9.69e-02, rmse_f = 1.80e+00, lr = 4.31e-08` and `val rmse_e = 1.10e-01, rmse_f = 1.91e+00` on **both** hosts, to every printed digit. This is the strongest equivalence evidence available short of a full run.
- **Throughput: the pod is 6.3x faster than the originating host.** Measured on the identical 100-step config: pod **0.1404 s/batch**, originating host **0.8823 s/batch**. The batch is tiny (4 frames x 192 atoms = 768 atoms), so the pod's 4090 is far from saturated; the win is real but modest, not the 20x a 4090-vs-8GB-GPU comparison would suggest.
  Planning numbers on the pod, per arm: 2,000 steps about 5 min; 10,000 about 23 min; 50,000 about 2 h; 358,500 (the cace reference's effective length) about 14 h.
- **`dp --pt freeze` works on the pod and the frozen model matches eager**: `max|dE| = 0` exactly, `max|dF| = 1.1e-16`, `max|dV| = 6.7e-15`.
- **CACE runs on the pod**: `inspect_cace_les.py` loads the author's `best_model.pth` and evaluates the 159-frame valid split.
- **`hpc_setup.sh` completes**: torch 2.8.0+cu128 (CUDA 12.8, `is_available() True`), deepmd-kit / les / cace all editable, `ENABLE_CUSTOMIZED_OP=False`.

To re-run the equivalence check after any environment change: run the same short config on both hosts and diff the logged `rmse` columns; they should match to every printed digit.

### Verified on the originating host, do not redo

- **Three correctness bugs in `deepmd/pt/model/model/hybridles_model.py`** were fixed before any of the results meant anything, because pre-fix the model was energy-force inconsistent: `coord.requires_grad_(True)` must be set before the descriptor recompute; the long-range force must be `autograd.grad(E_lr_total.sum(), coord)` over the full `coord` rather than the per-frame slice, or the descriptor-response term is silently dropped; and `E_lr` must be reshaped to `[nframes, 1]` before it is added to `energy_redu`, or the sum broadcasts to an `[nframes, nframes]` outer sum, invisible in training (batch size auto is one frame) and fatal in evaluation at batch greater than one. The fix is value-preserving: the regenerated sweep reproduced the pre-fix sweep bit-exactly.
- **`serialize()`/`deserialize()`** carries `les_params` and all 8 LES tensors, round-trip bit-exact (`check_serialize.py`).
- **The virial includes the Ewald part.** The full virial matches finite differences of all nine strain components on a sheared, non-cubic cell to 1.4e-05, where a short-range-only virial is off by 1.5e+02 (`check_virial.py`).
- **`torch.jit.script` and the real `dp --pt freeze` CLI** both work and agree with the eager model (`check_frozen.py`).
- **The verification harness `tests/hybrid_ener/`** (8 scripts) runs on a synthetic H2O system with a randomly initialised model and no checkpoint, exercising the real `get_model() -> HybridLESModel -> HybridLESAtomicModel` path. All pass. This is the canonical copy (section 9).
- **The model comparison itself** is written up in `LES_PERFORMANCE.md` and reproduced by the executed notebook `01.train/rerun/LES_analysis.ipynb`.

## 9. Known open issues, reported and deliberately not fixed

- The `element_numbers` / TorchScript gap in section 6, rule 7. Fixing it means not registering that buffer `persistent=False`, which is a one-line change with a scripted-loading consequence worth checking.
- **24 `.pyc` files are tracked** (committed before `*.pyc` was ignored), so they will show as dirty forever unless someone runs `git rm --cached` on them.
- **`01.train/rerun/` holds older duplicates** of the `tests/hybrid_ener/` check scripts (all six differ, by 80 to 192 lines each). `tests/hybrid_ener/` is the canonical set; the `rerun/` copies are historical.
- `eval_full.py`'s docstring gives the stale `no_jit=True` rationale.
- `CLAUDE.md` claims `check_frozen.py` invokes the `dp freeze` CLI; it does not.
- A stray vim swap file, `01.train/rerun/extended/run_hybrid_fixed_sA/.input.yaml.swp`, is untracked and harmless.

## 10. Where the truth lives

| path | what it is |
|---|---|
| `LES_PERFORMANCE.md` | the model comparison and its caveats; authoritative for every number |
| `01.train/rerun/LES_analysis.ipynb` | the executed analysis, with its tracked `*.tsv` inputs |
| `CLAUDE.md` (both) | architecture of the `hybrid_ener` integration, registration points, conventions |
| `tests/hybrid_ener/` | the canonical verification harness |
| `cace_compare_e2e/` | the shared dataset, the cace reference, the cace-side configs |

Conventions to keep: comments and docstrings in the modified deepmd files are written in **Chinese**, matching that tree.
Files under `deepmd-kit-3.1.2/` and `les/` are edited in place and take effect immediately because both installs are editable.

One thing that does **not** travel with the repo: the originating host kept an auto-memory directory at `/root/.claude/projects/-root-app-deepmd-les/memory/`, holding the open initiatives, the user's framing rules and the host-specific gotchas. If you want that context on the HPC, copy that directory to the same path there; otherwise this file is the substitute.
