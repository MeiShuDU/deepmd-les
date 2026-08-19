# deepmd-les

This repository provides a custom build of **DeepMD-kit v3.1.2** integrated with the **Latent Ewald Summation (LES)** library for long-range electrostatic interactions in machine learning interatomic potentials.

## Acknowledgements and Origins

- **DeepMD-kit** is developed and maintained by the DeepModeling community.  
  Official repository: [https://github.com/deepmodeling/deepmd-kit](https://github.com/deepmodeling/deepmd-kit)  
  This work is based on the official v3.1.2 release[reference:0].

- **LES (Latent Ewald Summation)** is a plug-in library developed by the Cheng Group (UC Berkeley) for adding long-range interactions to short-ranged MLIPs.  
  Official repository: [https://github.com/ChengUCB/les](https://github.com/ChengUCB/les)[reference:1][reference:2]  
  The method is described in:  
  Cheng, Bingqing. "Latent Ewald Summation for Machine Learning of Long-Range Interactions." *npj Computational Materials*, vol. 11, 80, Springer Nature, 2025. doi:10.1038/s41524-025-01577-7[reference:3][reference:4].

## Features

- **DeepMD-kit v3.1.2** with PyTorch backend (`[torch]`).
- Integrated **LES library** for long-range electrostatics.
- Added a custom model type **`hybrid_ener`** that combines short-range (DeepMD) and long-range (LES) interactions.
- To use the LES-enhanced model, simply specify `type: hybrid_ener` in your DeepMD input file (see example below).

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

Create a DeepMD input file (e.g., `input.yaml`) with the following model section:

```yaml
model:
  type: hybrid_ener
  type_map: ["O", "H"]
  descriptor:
    type: se_a
    sel: [46, 92]
    rcut: 6.0
    rcut_smth: 0.5
    neuron: [25, 50, 100]
    axis_neuron: 16
  fitting_net:
    neuron: [240, 240, 240]
  les_params:
    use_atomwise: true
    sigma: 1.0
    dl: 1.5
```

Then train with:

```bash
dp --pt train input.yaml
```

## A Note on Repository Management

I am relatively new to GitHub and the open-source collaboration workflow. This repository was created by directly pushing local files rather than through a formal fork of the upstream repositories. As a result, the commit history does not preserve the original contribution history of DeepMD-kit or LES. Full credit for the original work belongs to the DeepModeling community and the Cheng Group (UC Berkeley), as cited above. I welcome any guidance on improving the repository structure or collaboration practices.

## License

This repository includes code from:
- **DeepMD-kit** (LGPL-3.0)  
- **LES** (CC BY-NC 4.0, as stated in its repository)

Please refer to the respective licenses for terms of use.
