# deepmd-les

This repository provides a custom build of **DeepMD-kit v3.1.2** integrated with the **Latent Ewald Summation (LES)** library for long-range electrostatic interactions in machine learning interatomic potentials.

It adds a custom model type **`hybrid_ener`** that combines short-range DeepMD interactions with a long-range LES/Ewald term, so the total energy is `E = E_SR + E_LR`.

## Acknowledgements and Origins

- **DeepMD-kit** is developed and maintained by the DeepModeling community.  
  Official repository: [https://github.com/deepmodeling/deepmd-kit](https://github.com/deepmodeling/deepmd-kit)  
  This work is based on the official v3.1.2 release.

- **LES (Latent Ewald Summation)** is a plug-in library developed by the Cheng Group (UC Berkeley) for adding long-range interactions to short-ranged MLIPs.  
  Official repository: [https://github.com/ChengUCB/les](https://github.com/ChengUCB/les)  
  The method is described in:  
  Cheng, Bingqing. "Latent Ewald Summation for Machine Learning of Long-Range Interactions." *npj Computational Materials*, vol. 11, 80, Springer Nature, 2025. doi:10.1038/s41524-025-01577-7.

## Features

- **DeepMD-kit v3.1.2** with the PyTorch backend (`[torch]`).
- Integrated **LES library** for long-range electrostatics.
- Custom model type **`hybrid_ener`**, selected with `type: hybrid_ener` in the input file.
- **Consistent long-range force and virial.** The reported force and virial are exact derivatives of the reported energy, including the LES/Ewald contribution and its explicit cell dependence, so training with `pref_v > 0` and NPT runs are self-consistent.
- **Checkpoint round-trip.** `serialize()` / `deserialize()` carry the LES weights and `les_params`, so model export and convert-back preserve the long-range part.
- **Freezing.** The model is `torch.jit.script`-able, so `dp freeze` produces a deployable frozen model whose numerics match the eager model.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/MeiShuDU/deepmd-les.git
cd deepmd-les
```

### 2. Install DeepMD-kit (modified version) with PyTorch backend

```bash
cd deepmd-kit-3.1.2
pip install -e .[torch]
cd ..
```

### 3. Install the LES library

```bash
cd les
pip install -e .
cd ..
```

## Usage Example

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

## Verification

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
python check_frozen.py             # eager vs scripted numerics

# against a trained model
python check_virial.py ../../model.ckpt.pt
```

Each script prints `RESULT: PASS` or `RESULT: FAIL` and exits non-zero on failure.

## Implementation Notes

The `hybrid_ener` path is `HybridLESModel` (`deepmd/pt/model/model/hybridles_model.py`) built on `HybridLESAtomicModel` (`deepmd/pt/model/atomic_model/hybridles.py`), which owns the `Les` module from the `les` package.
The forward pass computes the short-range energy through the normal DeepMD path, recomputes the descriptor to feed LES, and obtains the long-range force and virial by autograd so that both remain exact derivatives of the reported energy.
The per-frame LES loop is vectorized into a single batched call so the whole forward is `torch.jit.script`-able.

## A Note on Repository Management

I am relatively new to GitHub and the open-source collaboration workflow. This repository was created by directly pushing local files rather than through a formal fork of the upstream repositories. As a result, the commit history does not preserve the original contribution history of DeepMD-kit or LES. Full credit for the original work belongs to the DeepModeling community and the Cheng Group (UC Berkeley), as cited above. I welcome any guidance on improving the repository structure or collaboration practices.

## License

This repository includes code from:
- **DeepMD-kit** (LGPL-3.0)  
- **LES** (CC BY-NC 4.0, as stated in its repository)

Please refer to the respective licenses for terms of use.
