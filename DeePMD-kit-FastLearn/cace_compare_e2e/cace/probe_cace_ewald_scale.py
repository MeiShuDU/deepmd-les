#!/usr/bin/env python
# coding: utf-8
"""What scale does the author's cace Ewald kernel work at, with no training involved?

`inspect_cace_les.py` shows that in the trained checkpoint the long-range term is
negligible. This asks the separate question of whether that is a property of the
checkpoint or of the kernel itself, by driving the kernel with a KNOWN physical
charge assignment - the same SPC/E table the les `freeze_spce` arm uses - on the
same valid frames, and comparing against what the les kernel returns for that
assignment.

If the trained model's small E_lr were merely an untrained head, a physical
assignment would come back at les's order of magnitude. It does not: it comes back
about two orders of magnitude smaller, which is the `norm_factor = 1` convention.

The author's construction is `EwaldPotential(dl=2, sigma=1., norm_factor=1,
remove_self_interaction=False)`. `norm_factor` is hardcoded to 1.0 with the source
comment "when using a norm_factor = 1, all charges are scaled by sqrt(90.0474)";
the physical value is `1/(2 eps_0) = 90.0474`, so the returned potential is low by
that factor by construction. `sigma=1` A also smears each charge over a Gaussian
far wider than an O-H bond, which suppresses the reciprocal sum's high-k terms.

Also reports the library default `remove_self_interaction=True`, which unlike the
author's setting subtracts the Gaussian self-energy. That changes only the level:
the frame-to-frame std is identical, because the self term is a per-frame constant
for a fixed charge set. The analytic value of that shift is
`sum(q^2) / (sigma * (2 pi)^1.5)`, printed as a check.

Usage:
    python probe_cace_ewald_scale.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cace
from cace.modules import EwaldPotential
from cace.tasks.load_data import get_dataset_from_xyz, load_data_loader

cace.tools.setup_logger(level='ERROR')
torch.set_default_dtype(torch.float32)

REF = {1: -187.6043857100553, 8: -93.80219285502734}
SPCE = {8: -0.8476, 1: +0.4238}          # the les freeze_spce table
CUTOFF = 5.5
N_BATCHES = 12                            # 48 frames, enough for a scale statement


def main():
    collection = get_dataset_from_xyz(
        train_path=os.path.join(HERE, 'water.xyz'), valid_fraction=0.1, seed=1,
        cutoff=CUTOFF, data_key={'energy': 'energy', 'forces': 'force'},
        atomic_energies=REF)

    kernels = {
        'author (rsi=False)': EwaldPotential(
            dl=2, sigma=1., feature_key='q', output_key='ewald_potential',
            remove_self_interaction=False, aggregation_mode='sum'),
        'default (rsi=True)': EwaldPotential(
            dl=2, sigma=1., feature_key='q', output_key='ewald_potential',
            remove_self_interaction=True, aggregation_mode='sum'),
    }

    vals = {k: [] for k in kernels}
    for i, batch in enumerate(load_data_loader(collection, 'valid', 4)):
        if i >= N_BATCHES:
            break
        bd = batch.to_dict()
        z = bd['atomic_numbers']
        q = torch.tensor([SPCE[int(zi)] for zi in z],
                         dtype=torch.float32).unsqueeze(1)
        for name, kern in kernels.items():
            out = kern({'positions': bd['positions'], 'cell': bd['cell'],
                        'batch': bd['batch'], 'q': q})
            vals[name].extend(np.asarray(out['ewald_potential']).ravel().tolist())

    print(f'physical SPC/E charges through the author\'s kernel, '
          f'{len(vals["author (rsi=False)"])} valid frames\n')
    for name, v in vals.items():
        a = np.array(v)
        print(f'  {name:20s} mean {a.mean():+10.4f}  std {a.std():7.4f}  '
              f'min {a.min():+10.4f}  max {a.max():+10.4f}  eV/frame')

    # the self term the two settings differ by, checked against its closed form
    q_all = np.array([SPCE[8]] * 64 + [SPCE[1]] * 128)
    analytic = float((q_all ** 2).sum() / (1.0 * (2 * np.pi) ** 1.5))
    got = (np.array(vals['author (rsi=False)']).mean()
           - np.array(vals['default (rsi=True)']).mean())
    print(f'\n  self term: measured {got:+.4f} vs sum(q^2)/(sigma*(2pi)^1.5) '
          f'= {analytic:+.4f} eV/frame')

    print('\nreference, from CACE_COMPARISON.md (les `freeze_spce`, same table,'
          ' weight 1.0, physical kernel):')
    print('  mean -381.53  std 2.98  eV/frame')
    print('\nso the author\'s kernel returns ~90x less variation for the same'
          ' physical charges.')


if __name__ == '__main__':
    main()
