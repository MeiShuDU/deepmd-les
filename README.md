# deepmd-les

A development repository for **LES (Latent Ewald Summation)** fitting built on top of established machine-learning interatomic potential tooling.

It holds two efforts:

- **`deepmd-les`** - a custom build of **DeepMD-kit v3.1.2** integrated with the **LES** library, adding a custom model type **`hybrid_ener`** that combines short-range DeepMD interactions with a long-range LES/Ewald term, so the total energy is `E = E_SR + E_LR`.
- **`desc_bridging/`** (the **`deepmd_cace`** package) - an under-development module that bridges descriptors from DeepMD-kit with trainers and modules from CACE, to achieve LES fitting on a more computation-affordable schedule.

## Status

**The `desc_bridging/` bridge is under active development.**
It is the recommended path in this repository.
The `deepmd-les` `hybrid_ener` framework is valid, but `deepmd-cace` is more recommended.

| component | status | what it is |
| --- | --- | --- |
| `desc_bridging/` (`deepmd_cace`) | **under development, recommended** | DeepMD-kit's `DescrptSeA` descriptor substituted at the input seam of CACE's training graph, so the LES/Ewald long-range machinery, the atomwise heads and the training schedule come from CACE while the descriptor comes from DeepMD-kit. |
| `deepmd-les` (`hybrid_ener`) | **valid, not recommended as the training path** | The long-range model inside the modified DeepMD-kit. Force, virial, serialize and freeze are self-consistent and numerically verified, so the framework stands on its own; the LES payoff it demonstrated in the campaign was however small and unstable, which is why the bridge is the recommended route. |
| `charge_eq_latent` | **under development** | A charge style that replaces the network's latent-charge proposal with an electrostatically equalized set before the Ewald sum, on the bridge side. It is implemented, configurable and documented, and it is still being validated; see the honest limits in `desc_bridging/README.md`. |
| `charge_eq` (in `les/`) | implemented, DeepMD/LES side | The first version of the same step: `project_zero_mean(q_r, A, w)` in the LES package, enabled with `les_params.charge_eq` and documented in `les/src/les/CHARGE_EQ.md`. `charge_eq_latent` is its counterpart on the CACE kernel. |

Two things are deliberately **not** in this repository: the campaign training data, checkpoints and analysis dumps, and any pod credentials.
The campaign scripts, configuration and READMEs that produced the results are tracked under `DeePMD-kit-FastLearn/`, but the frames, `.pt`/`.pth` checkpoints and `scored_*.npz` dumps that sit next to them are not published here.

## Repository layout

```text
deepmd-kit-3.1.2/   modified DeepMD-kit v3.1.2 (PyTorch backend), installed editable
les/                the LES library (src/les), installed editable
tests/hybrid_ener/  small self-contained checks for the hybrid_ener model
desc_bridging/      the deepmd_cace bridge: model, trainer, descriptor adapter,
                    charge_eq_latent, and the BEC analysis notebooks and scripts
DeePMD-kit-FastLearn/
                    campaign scripts, input configs and per-arm READMEs
                    (data and checkpoints not published)
```

## Contributions of the upstream projects

This repository is a bridge and a set of integrations. The hard parts belong to the upstream projects, and they deserve the credit.

### DeepMD-kit (DeepModeling community)

- **The descriptor.** `DescrptSeA` supplies the short-range environment description that both the `hybrid_ener` model and the `deepmd_cace` bridge build on.
- **The training skeleton.** The PyTorch backend, the atomic-model / model split, the loss, the `exp` learning-rate scheduler, the checkpoint and `serialize`/`deserialize` machinery, the `dp` CLI and the freeze path are all upstream DeepMD-kit.
- **The deployment path.** `dp freeze` and the C++ API that `hybrid_ener` was made compatible with.
- Upstream: [https://github.com/deepmodeling/deepmd-kit](https://github.com/deepmodeling/deepmd-kit), based on the official v3.1.2 release.

### CACE (Cheng Group, UC Berkeley)

- **The long-range machinery.** The Ewald kernel, the latent-charge `Atomwise` head, `CombinePotential`, `FeatureAdd`, the `Forces` differentiation path and CACE's graph-based data pipeline are all upstream CACE.
- **The training loop used by the bridge.** The `deepmd_cace` trainer hands CACE's own heads, losses and `TrainingTask` the samples it builds, rather than reimplementing them.
- **The method.** Latent Ewald Summation, and the reference implementation of the Ewald summation this repository's kernel is checked against.
- Upstream: [https://github.com/ChengUCB/cace](https://github.com/ChengUCB/cace) and [https://github.com/ChengUCB/les](https://github.com/ChengUCB/les) for the standalone LES library.
- The method is described in: Cheng, Bingqing. "Latent Ewald Summation for Machine Learning of Long-Range Interactions." *npj Computational Materials*, vol. 11, 80, Springer Nature, 2025. doi:10.1038/s41524-025-01577-7.

### LES (Cheng Group, UC Berkeley)

- The standalone LES library provides the `Atomwise` / `FixedCharges` / `AtomicAlpha` charge layers, the Ewald summation and the BEC module that `hybrid_ener` drives.
- This work is based on the official v3.1.2 release of DeepMD-kit and on that LES library.

### What this repository adds

- The `hybrid_ener` model type, which wires LES into DeepMD-kit's model/atomic-model API with consistent force, virial, serialize and freeze behaviour.
- The `deepmd_cace` bridge, which substitutes DeepMD-kit's descriptor into CACE's training graph.
- The `charge_eq_latent` style, an electrostatically equalized latent-charge step that keeps the CACE workflow and parameter plumbing.
- The campaign scripts and analysis notebooks under `DeePMD-kit-FastLearn/` and `desc_bridging/`.

## The `hybrid_ener` model (`deepmd-les`)

This is the long-range model inside the modified DeepMD-kit. Its framework is valid and verified; the campaign's LES gain was small and unstable, so `deepmd-cace` is the recommended training path.

### Features

- **DeepMD-kit v3.1.2** with the PyTorch backend (`[torch]`).
- Integrated **LES library** for long-range electrostatics.
- Custom model type **`hybrid_ener`**, selected with `type: hybrid_ener` in the input file.
- **Consistent long-range force and virial.** The reported force and virial are exact derivatives of the reported energy, including the LES/Ewald contribution and its explicit cell dependence, so training with `pref_v > 0` and NPT runs are self-consistent.
- **Checkpoint round-trip.** `serialize()` / `deserialize()` carry the LES weights and `les_params`, so model export and convert-back preserve the long-range part.
- **Freezing.** The model is `torch.jit.script`-able, so `dp freeze` produces a deployable frozen model whose numerics match the eager model.

### Installation

```bash
git clone https://github.com/MeiShuDU/deepmd-les.git
cd deepmd-les

cd deepmd-kit-3.1.2
pip install -e .[torch]
cd ..

cd les
pip install -e .
cd ..
```

For the bridge, the extra requirement is only that DeepMD-kit, CACE, ASE, NumPy and PyTorch are importable in the same environment; see `desc_bridging/README.md`.

### Usage example

Create a DeepMD input file (e.g., `input.json`) with the following model section:

```json
{
    "model": {
        "type": "hybrid_ener",
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_a",
            "sel": [46, 92],
            "rcut_smth": 0.5,
            "rcut": 6.0,
            "neuron": [25, 50, 100],
            "axis_neuron": 16,
            "resnet_dt": false,
            "seed": 1
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
            "resnet_dt": true,
            "seed": 1
        },
        "les_params": {
            "use_atomwise": true,
            "sigma": 1.0,
            "dl": 1.5
        }
    },
    "learning_rate": {
        "type": "exp",
        "decay_steps": 5000,
        "start_lr": 0.001,
        "stop_lr": 3.51e-08
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 0.02,
        "limit_pref_e": 1.0,
        "start_pref_f": 1000.0,
        "limit_pref_f": 1.0,
        "start_pref_v": 0.0,
        "limit_pref_v": 0.0
    },
    "training": {
        "training_data": {
            "systems": ["../data/data_0/", "../data/data_1", "../data/data_2/"],
            "batch_size": "auto"
        },
        "validation_data": {
            "systems": ["../data/data_3"],
            "batch_size": 1,
            "numb_btch": 3
        },
        "numb_steps": 1000,
        "seed": 10,
        "disp_file": "lcurve.out",
        "disp_freq": 100,
        "save_freq": 500
    }
}
```

Then train, freeze, and run inference with the usual DeepMD commands:

```bash
dp --pt train input.json
dp --pt freeze -c model.ckpt.pt -o frozen_model.pth
```

### `les_params`

| Key | Default | Meaning |
| --- | --- | --- |
| `use_atomwise` | `false` | Predict the latent charges with a small MLP from the descriptor. Set `true` for `hybrid_ener`. |
| `sigma` | `1.0` | Ewald Gaussian screening width. |
| `dl` | `2.0` | Real-space grid spacing for the reciprocal-space sum. |
| `use_fixed_atomic_charges` | `false` | Add a fixed per-element charge baseline (H `+1`, O `-2`, ...) to the latent charges. |
| `fixed_atomic_charges_scaling_factor` | `0.5` | Scale applied to that baseline. |
| `use_atomic_alpha` | `false` | Add a fixed per-element polarizability baseline. |
| `use_epsilon_r_scaling` | `false` | Apply relative-permittivity scaling. |
| `verbose`, `log_freq` | `false`, `100` | Write periodic SR/LR energy and charge statistics to `les.log` for debugging whether LES is learning. |

`type_map` must contain real element symbols when `use_fixed_atomic_charges` or `use_atomic_alpha` is set, because the per-element tables are looked up by symbol.

### Charge layer

The charge layer decides where the latent charges come from, and the two options are mutually exclusive:

| Key | Meaning |
| --- | --- |
| `local_charge` (alias `use_atomwise`) | `q = Atomwise(descriptor)`: per atom, predicted from the local environment and trained by SGD. |
| `freeze_charge` | `q = per-type constant`: given up front and never updated, so the model is a classical Ewald sum over a fixed charge table. |

In local mode the NN output can carry a per-element baseline: `use_fixed_charges` (the oxidation-number table from `FixedCharges`, scaled by `fixed_atomic_charges_scaling_factor`, looked up by element symbol) or `initial_guess` (a user-supplied per-type vector).
Freeze mode rejects both, because the charges are already given.
`claim_total_charge=S` (`claim_neutral` as the shorthand for `S=0`) projects each frame's charges onto the `sum(q) = S` plane; the projection is linear and idempotent, so the constraint holds exactly at every step while gradients still flow.

The per-type tables are non-persistent buffers, so they stay out of the `state_dict` and older checkpoints still load strictly.

### Verification

`tests/hybrid_ener/` holds small self-contained checks for the hybrid model.
Each one runs against a synthetic H2O system and a randomly initialised model, so no training data or checkpoint is required.
Pass a checkpoint path as the first argument to run a check against a trained model instead.

```bash
cd tests/hybrid_ener

python check_consistency.py        # analytic force == -dE/dr (finite difference)
python check_lr.py                 # LES/Ewald force == -dE_LR/dr (finite difference)
python check_virial.py             # virial == -dE/deps over all nine strain components
python check_serialize.py          # serialize()/deserialize() round-trip
python try_freeze.py               # torch.jit.script -> save -> load
python check_frozen.py             # eager vs scripted numerics, every charge-layer mode
python check_charge_modes.py       # charge-layer properties on the bare Les module
python check_charge_modes_e2e.py   # charge-layer properties through the real hybrid_ener model
python check_ewald_reference.py    # Ewald kernel vs analytic Coulomb / bilinearity
python check_forward_lower.py      # forward_lower: cell handling, virial, scripting

# against a trained model
python check_virial.py ../../model.ckpt.pt
```

Each script prints `RESULT: PASS` or `RESULT: FAIL` and exits non-zero on failure.

### Implementation Notes

The `hybrid_ener` path is `HybridLESModel` (`deepmd/pt/model/model/hybridles_model.py`) built on `HybridLESAtomicModel` (`deepmd/pt/model/atomic_model/hybridles.py`), which owns the `Les` module from the `les` package.
The forward pass computes the short-range energy through the normal DeepMD path, hands the descriptor it already built down to the LES channel instead of recomputing it, and obtains the long-range force and virial by autograd so that both remain exact derivatives of the reported energy.
The per-frame LES loop is vectorized into a single batched call so the whole forward is `torch.jit.script`-able.

## The `desc_bridging` module (`deepmd_cace`)

`desc_bridging/` holds the recommended path: a lightweight training integration that drives CACE's graph data, atomwise heads, force differentiation, losses and `TrainingTask` with DeepMD-kit PyTorch's existing `DescrptSeA` descriptor.
It does not modify or fork either dependency.

Start with `desc_bridging/README.md`, which documents the dataset layout, the JSON configuration, the four SR/LR layouts, the `charge_eq_latent` style and its compatibility rules.
It also lists the honest limits of the current state, including the behaviours of CACE's shared Ewald kernel that a reader of long-range numbers needs to know.

## A Note on Repository Management

I am relatively new to GitHub and the open-source collaboration workflow. This repository was created by directly pushing local files rather than through a formal fork of the upstream repositories. As a result, the commit history does not preserve the original contribution history of DeepMD-kit or LES. Full credit for the original work belongs to the DeepModeling community and the Cheng Group (UC Berkeley), as cited above. I welcome any guidance on improving the repository structure or collaboration practices.

## License

This repository includes code from:
- **DeepMD-kit** (LGPL-3.0)
- **LES** (CC BY-NC 4.0, as stated in its repository)

Please refer to the respective licenses for terms of use.
