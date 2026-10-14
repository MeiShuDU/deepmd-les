# DeepMD-CACE

`deepmd_cace` is a lightweight training integration: it uses DeepMD-kit
PyTorch's existing `DescrptSeA` descriptor and CACE's existing graph data,
atomwise heads, force differentiation, losses, and `TrainingTask`.
It does not modify or fork either dependency.

The package lives in `desc_bridging/` at the repository root, which is why the
commands below set `PYTHONPATH=desc_bridging` and run from that root.
It is under development; the repository README states its status and the
contributions of the upstream projects it builds on.

## Requirements

Use an environment where the DeepMD-kit PyTorch backend, CACE, ASE, NumPy, and
PyTorch are already importable.
For CPU-only runs on machines with a visible GPU, set
`CUDA_VISIBLE_DEVICES=""` before Python starts because DeepMD's descriptor
statistics use its process-global device.

## Dataset

Each system directory follows the DeepMD layout:

```text
dataset/train/
  type.raw
  type_map.raw
  set.000/{coord,box,energy,force}.npy
dataset/valid/
  type.raw
  type_map.raw
  set.000/{coord,box,energy,force}.npy
```

`coord.npy` and `force.npy` may be flattened per frame (`nframes x natoms*3`)
or shaped (`nframes x natoms x 3`). `box.npy` contains nine cell values per
frame; an all-zero cell denotes a non-periodic frame. `type.raw` stores integer
type ids and `type_map.raw` must exactly match `model.type_map` in `input.json`.
Energy and force values are consumed as given; no atomic reference energy is
subtracted unless `atomic_energies` is defined in `input.json`. That mapping
uses atomic numbers as keys and is subtracted by CACE when constructing each
sample, matching the campaign fitting scripts.

To split one labeled DeepMD system deterministically:

```bash
PYTHONPATH=desc_bridging python -m deepmd_cace.split_dataset path/to/system \
  --output path/to/dataset --valid-fraction 0.1 --seed 1
```

The split command merges the source's `set.*` frames, shuffles indices with
NumPy's seeded generator, and writes separate `train/` and `valid/` systems.

## Configure and run

Start with [`input.json`](input.json). The outer `model`, `loss`, and
`training` sections follow the familiar DeepMD-kit JSON organization.
`model.type_map` and `model.descriptor` configure `DescrptSeA`; `fitting_net`
configures CACE's SR `Atomwise` head. `model.arm` selects `sr` or `lr` and must
agree with `long_range.enabled`; LR adds CACE's latent-charge `Atomwise` head
and Ewald energy to the SR energy.
Set `long_range.charge_eq_latent.enabled` to replace the charge head's proposal
by an electrostatically equalized set before the Ewald sum; the style is
described in the model reference below.
For the `CombinePotential` LR layout, set `long_range.share_descriptor` to
`true` to compute se_a once and share its node features across the SR and LR
heads; the LR force is then differentiated once from the combined energy.
`training.blocks` supports an energy-weight schedule. Each block accepts
`energy_weight`, `repeat`, `epochs`, `fresh_task`, and an optional `checkpoint`
filename. A fresh task retains model weights but resets optimizer, scheduler,
warmup, and task counters; a continuing task only replaces the loss weights.
The included configuration selects the script's SR arm and reproduces
`fit_cace_sea.py`'s 500-epoch schedule and optimizer settings:
five fresh 40-epoch blocks at energy weight `0.1`, then 100 epochs each at
`1`, `10`, and `1000`, with the last three blocks sharing optimizer state.
Omit `blocks` to use a single `training.epochs` block and `loss.energy_weight`.

```bash
PYTHONPATH=desc_bridging python -m deepmd_cace desc_bridging/input.json --check-only
PYTHONPATH=desc_bridging python -m deepmd_cace desc_bridging/input.json
```

`--check-only` loads both splits, computes DeepMD descriptor statistics, runs a
forward pass including force differentiation, and confirms that gradients reach
the descriptor without starting training. Descriptor statistics are computed
in frame chunks to bound neighbor-list memory; `training.descriptor_stats_batch_size`
defaults to `1`. Optimizer, scheduler, block schedule, loss weights, batch sizes,
seed, and output directory are configured in JSON.
Restore an inference-ready phase checkpoint (for the included SR schedule,
`model-4.pth`) or `best_model.pth` with
`deepmd_cace.model.load_checkpoint(path)`. `checkpoint.pt` is CACE's separate
training-state file, including optimizer state. The included schedule reserves
`model.pth` for the first block, as the campaign script does.

The integration requires fixed atom counts within each individual CACE batch,
as expected by the DeepMD extended-region builder. The adapter processes graphs
one at a time, so different systems and variable graph sizes can coexist in a
batch.

## Model reference

`model` accepts the keys below. Only `long_range` is expanded here in full,
because it is the part with non-obvious relationships between its options.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `type` | string | `deepmd_cace` | must be `deepmd_cace` |
| `arm` | string | unset | `sr` or `lr`; sets `long_range.enabled` and must agree with it when both are given |
| `type_map` | list | required | element symbols, unique; must match `type_map.raw` of every system |
| `descriptor` | object | required | passed to `DescrptSeA` after dropping `type`; `type` must be `se_a` and `rcut` is required |
| `fitting_net` | object | required | `Atomwise` options for the SR energy head: `n_layers`, `n_hidden`, `use_batchnorm`, `add_linear_nn`, `bias` |
| `long_range` | object | `{}` | the long-range branch; see below |

### `model.long_range`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | adds the latent-charge head and the Ewald energy; required for any long-range arm |
| `weight` | float | `0.02` | the coupling `w` in `E = E_sr + w * E_lr` |
| `combine_potentials` | bool | `false` | wrap the two branches in CACE's `CombinePotential` (with `weight`) instead of adding them with `FeatureAdd` |
| `share_descriptor` | bool | `false` | only inside `combine_potentials`: compute se_a once and share its node features with both branches |
| `charge_net` | object | `{}` | the latent-charge `Atomwise` head: `n_out` (default `1`, the number of charge channels), `n_layers` (`3`), `n_hidden` (`[24, 12]`), `bias` (`false`) |
| `ewald` | object | `{}` | kernel options, see below |
| `charge_eq_latent` | object | `{}` | the charge-equilibration style, see below |

The `ewald` object is passed to CACE's `EwaldPotential`, so it takes that
module's options: `dl` (`2.0`, the reciprocal-grid resolution), `sigma` (`1.0`,
the Gaussian width), `exponent` (`1`), `remove_self_interaction` (`true`),
`feature_key` (`q`), `output_key` (`ewald_potential`), and `aggregation_mode`
(`sum`).
The three defaults for `feature_key`, `output_key`, and `aggregation_mode` are
set by the integration rather than by `EwaldPotential`, and they are what the
downstream `FeatureAdd`/`CombinePotential` and `Forces` expect.
The remaining `EwaldPotential` options (`external_field`,
`external_field_direction`, `charge_neutral_lambda`, `compute_field`) are also
accepted here, but only while `charge_eq_latent` is off; see the compatibility
table.

The four layouts `combine_potentials` x `share_descriptor` all build and run,
and they differ in parameter names, so a checkpoint only loads into the layout
it was trained with.
The campaign uses `combine_potentials: true` with `share_descriptor: true`.

### The `charge_eq_latent` style

`charge_eq_latent` keeps the DeepMD descriptor, the latent-charge head, and the
Ewald kernel, and adds one step between the charges and the sum: the network's
proposal `q_r` is relaxed to a set that is self-consistent under the Coulomb
matrix before the energy is taken.
It takes these options.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | replace the kernel with the equalizing one |
| `regularization_weight` | float | `0.1` | fixed inverse hardness `2w`, or the positive initial hardness scale for a learned head; fixed `w=0` is allowed only when the constrained objective is convex |
| `total_charge` | float | `0.0` | the value the summed charge is held at |
| `total_charge_key` | string | `system_charge` | a batch entry that, when present, supplies the total charge per frame and overrides `total_charge` |
| `learnable_kappa` | bool | `false` | predict positive atomwise inverse hardness from local descriptor features using a separate 16-unit SiLU head initialized at `2w` |
| `systems` | array | absent | fixed one-based group definitions and CONP/CONQ constraints selected per DeepMD system; requires one charge channel |
| `slab_axis` | `0`, `1`, `2` or `null` | `null` | optional neutral-slab Yeh-Berkowitz correction along a Cartesian-aligned periodic-cell axis |

Without `systems`, the legacy step is the constrained minimization

```text
min_q  1/2 q^T A q + w ||q - q_r||^2    subject to    1^T q = Q
```

Here `A` is the Ewald Coulomb matrix built through the same kernel, `w` is `regularization_weight`, and `Q` is the total charge.
The generalized solve replaces `2w I` by `K_inv = diag(kappa_inv)` and uses all active constraints `C q = r`:

```text
[ A + K_inv   C^T ] [ q      ] = [ K_inv q_r ]
[ C           0   ] [ lambda ]   [ r         ]
```

With `slab_axis` set, the corrected Coulomb matrix is used consistently in the solve, potential constraints, and reported energy.

Two limits bracket it:

* `w -> infinity` pins the charges to the proposal, and the module falls back to
  the plain latent-charge Ewald term up to the one thing the constraint still
  imposes.
  The proposal carries a net charge, so the limit is `q_r` shifted uniformly by
  `(sum q_r - Q) / (n_atoms * n_out)` rather than `q_r` itself.
* `w -> 0` removes the hardness penalty; a unique minimum then exists only if
  the Coulomb matrix has positive curvature on the feasible charge subspace.
  The solver checks this condition and rejects an indefinite objective.

The Ewald adapter supplies `compute_coulomb_matrix`, using the kernel's native matrix API when available and a compatible dense contraction for CACE versions without it.
The Ewald operator is assembled once per frame and reused by the constraint rows,
KKT solve, and Coulomb energy evaluation.
The solve is a single linear system per frame and it is differentiable, so the
long-range force is exact autograd of the reported energy rather than a finite
difference.
Three keys come out: `q_eq` (the equalized charges, `[n_atoms, n_out]`), `q`
(the proposal, left untouched and still available to the trainer and to
scoring), and `ewald_potential` (the long-range energy, same shape and meaning
as the plain kernel's, so nothing downstream changes).
With fixed hardness, the module adds no trainable parameters.
With learned hardness, its atomwise feature head becomes part of the state dict;
the resulting checkpoint must be reloaded with the same equalization configuration.

For fixed electrodes, list each DeepMD `training_data` and `validation_data` system
under `systems` using either its `system` path or `system_aliases` for splits that
share the same atom order and fixed groups.
Paths, including `group_definition`, are resolved relative to the input JSON file.
Group JSON maps group names to nonempty, unique, **one-based** atom indices;
`all` is reserved for the entire system.
The `electrode_mode` is `CONP` or `CONQ`, and applies only to
`electrode_constraints` within that system.
The training loader attaches a `charge_eq_system` integer to each frame, indexing the configured `systems` array.
Inference batches must supply the same metadata and preserve the selected system's atom ordering.
The optional `global_constraints` are CONQ and may include `group: "all"` for
neutralization in either mode.
CONP fixes the difference in the *mean* electrostatic potentials of two
disjoint electrode groups using the same Coulomb matrix as the energy.
`delta_V` is expressed in the units of `A q`, with the CACE kernel normalization.
It must not be interpreted as a voltage in volts without a calibrated charge and energy convention.
CONQ fixes the sum of charges in a named group.
The network emits `q_eq` and retains the proposal `q` unchanged.
The regularization term is part of the charge solve but **not** part of the
reported long-range energy, matching the accepted design.
External potential is deferred until its DFT sign, gauge, and units are fixed.

An excerpt for two datasets with different electrode modes is:

```json
"charge_eq_latent": {
  "enabled": true,
  "regularization_weight": 0.1,
  "learnable_kappa": true,
  "systems": [
    {
      "system": "./data_1/train",
      "system_aliases": ["./data_1/valid"],
      "group_definition": "./data_1/groups.json",
      "electrode_mode": "CONP",
      "electrode_constraints": [
        {"type": "CONP", "group1": "electrodeA", "group2": "electrodeB", "delta_V": 4.0}
      ],
      "global_constraints": [{"type": "CONQ", "group": "all", "total_charge": 0.0}]
    },
    {
      "system": "./data_2/train",
      "system_aliases": ["./data_2/valid"],
      "group_definition": "./data_2/groups.json",
      "electrode_mode": "CONQ",
      "electrode_constraints": [
        {"type": "CONQ", "group": "electrodeA", "total_charge": 0.0},
        {"type": "CONQ", "group": "electrodeB", "total_charge": 0.0}
      ],
      "global_constraints": [{"type": "CONQ", "group": "all", "total_charge": 0.0}]
    }
  ]
}
```

The second example requires electrolyte atoms outside both electrode groups;
otherwise the last row is redundant and the configuration is rejected.
The `systems` option is currently limited to a single charge channel and
per-frame dense `N x N` Coulomb/KKT matrices; benchmark large cells before HPC use.

`regularization_weight` is a real coupling rather than a near-no-op, and the
default was picked from the measured sweep in `check_charge_eq_latent.py`: on a
trained checkpoint `w = 0.1` moves the charges by 17% of their RMS and reports
`E_lr = 57.1` eV where the untouched proposal gives `183.0` eV, while `w = 100`
moves them by 0.9% and reports `174.2` eV.

### Compatibility

| combination | result |
| --- | --- |
| `charge_eq_latent.enabled` with `long_range.enabled: false` | rejected: the equalization lives in the long-range branch |
| `charge_eq_latent.regularization_weight < 0`, or zero with `learnable_kappa` | rejected; a fixed zero value is allowed only if the constrained Hessian is positive definite |
| `ewald.exponent != 1` | rejected: this is the electrostatic kernel |
| `ewald` carrying `external_field`, `external_field_direction`, `charge_neutral_lambda`, or `compute_field`, with the equalization on | rejected rather than ignored: those options belong to the kernel's own forward path, which the equalizing module never calls |
| unknown key under `charge_eq_latent` or `ewald` (with the equalization on) | rejected at configuration time |
| `charge_eq_latent` with any `combine_potentials` x `share_descriptor` layout | supported; all four build and run |
| a checkpoint trained with the equalization off, loaded into fixed-hardness equalization | loads strictly; learned hardness adds parameters and requires its own checkpoint |

Two behaviours of the shared kernel are worth knowing before reading long-range
numbers out of a model.

* With `remove_self_interaction: true` and `charge_net.n_out > 1`, CACE's
  `EwaldPotential` subtracts `sum(q**2)` over all channels from every channel,
  so it counts the self term `n_out` times and reports an energy that is lower
  than the quadratic form of the field matrix it also reports, by exactly
  `(n_out - 1) * sum q^2 / (sigma (2 pi)^1.5)`.
  `charge_eq_latent` reports `0.5 sum_c q_c^T A q_c` instead, which is the
  self-consistent quadratic form and the quantity whose gradient is the force it
  emits.
  The two agree exactly when `remove_self_interaction` is false or `n_out` is 1,
  and the campaign sets `remove_self_interaction: false`.
* Inside `combine_potentials`, CACE's `CombinePotential.forward` scales each
  branch's output by `weight` **in place**, on the tensor that still belongs to
  the shared data dict.
  After a combined forward, `data["ewald_potential"]` and `data["ewald_forces"]`
  therefore hold the weighted values `w * E_lr` and `w * F_lr`, not the raw
  module outputs, so a diagnostic that reads those keys off a combined model
  must not apply `weight` a second time.
  The training loss is unaffected, because `CACE_energy` and `CACE_forces` are
  the correctly weighted sums.

An LR configuration in the campaign's layout, with the equalization on:

```json
"long_range": {
  "enabled": true,
  "combine_potentials": true,
  "share_descriptor": true,
  "weight": 0.02,
  "charge_net": {"n_out": 4, "n_layers": 3, "n_hidden": [24, 12], "bias": false},
  "ewald": {"dl": 2.0, "sigma": 1.0, "remove_self_interaction": false},
  "charge_eq_latent": {"enabled": true, "regularization_weight": 0.1, "total_charge": 0.0}
}
```

Run the numerical checks over it, on real campaign frames, with:

```bash
PYTHONPATH=desc_bridging python desc_bridging/check_charge_eq_latent.py
```

For local synthetic mixed-mode coverage (including a one-epoch CPU smoke run):

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=desc_bridging python desc_bridging/check_charge_eq_fixed_groups.py
```
