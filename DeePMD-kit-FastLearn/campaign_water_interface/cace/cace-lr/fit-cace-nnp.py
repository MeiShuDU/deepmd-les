#!/usr/bin/env python
# coding: utf-8
#
# cace's own fit-interface-mp0 script, copied verbatim except for the edits
# listed in campaign_water_interface/cace/README.md. It is the cace half of the
# water-interface campaign: same net, same schedule, run on the campaign's GPU so
# its wall clock is comparable with the deepmd arms.
#
# Run it with the working directory set to the replicate's run directory,
# campaign_water_interface/cace/runs/cace-lr_s<tag>/ :
#
#     cd runs/cace-lr_sA && python ../../cace-lr/fit-cace-nnp.py 1
#     cd runs/cace-lr_sA && python ../../cace-lr/fit-cace-nnp.py 1 --smoke
#
# The training path below and every output (water-model*.pth, checkpoint.pt,
# best_model.pth, blocks.json) are relative to that working directory, which is
# what keeps the authors' own directory and its published model-*.pth untouched.

import sys
sys.path.append('../cace/')

import numpy as np
import torch
import torch.nn as nn
import logging
import json
import os
import time

import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)

# --- replicate, smoke and timing knobs (added for the campaign's pod runs) -----
# cace's Cace representation and its Atomwise heads are built from torch's global
# RNG and TrainingTask takes no seed argument (cace/tasks/train.py:17-52), so the
# replicate is torch.manual_seed() before the first draw - the model is built at
# "building CACE representation" below, and the training DataLoader's shuffle
# order is drawn from the same RNG (cace/tasks/load_data.py:42).
ARM = 'cace-lr'
_args = [a for a in sys.argv[1:] if not a.startswith('--')]
SEED = int(_args[0]) if _args else 1
SMOKE = '--smoke' in sys.argv
# cace's script is a single process that runs all of its training loops and cannot
# resume: it writes checkpoint.pt every 10 epochs (cace/tasks/train.py:261-262) but
# never reads one back, and TrainingTask.checkpoint() (there, :284) does not even
# record global_step, so the restore path that does exist (:293) could not recover
# the epoch counter the schedule runs on. So "run only the first N blocks" (the
# campaign's cost probe, and the point of --smoke) can only be expressed as a stop
# inside the process.
_STOP_AFTER = int(os.environ.get('CACE_STOP_AFTER', '0') or 0)
torch.manual_seed(SEED)

_BLOCKS = []


def PH(epochs):
    """Epochs for one of cace's training loops; --smoke shrinks them to 1.

    Not the cace schedule - the flag exists so the plumbing (two loaders, five
    fresh tasks, the w_E phases, the checkpoints) can be exercised in minutes.
    """
    return 1 if SMOKE else epochs


def _dump(finished=False):
    """Rewrite blocks.json, so a run that dies mid-way still has its timings."""
    with open('blocks.json', 'w') as fh:
        json.dump({'arm': ARM, 'replicate': SEED, 'smoke': SMOKE,
                   'steps_per_epoch': len(train_loader),
                   'max_gpu_mb': (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                                  if torch.cuda.is_available() else None),
                   'finished': finished, 'blocks': _BLOCKS}, fh, indent=2)


def _fit(label, task, epochs):
    """cace's fit call, timed; one entry per cace training loop (a "block")."""
    t0 = time.perf_counter()
    task.fit(train_loader, valid_loader, epochs=PH(epochs), screen_nan=False)
    dt = time.perf_counter() - t0
    n = PH(epochs) * len(train_loader)
    _BLOCKS.append({'n': len(_BLOCKS) + 1, 'label': label, 'epochs': PH(epochs),
                    'num_steps': n, 'seconds': round(dt, 2),
                    's_per_batch': round(dt / n, 6), 'ok': True})
    print(f'  block {len(_BLOCKS)} {label}: {dt:.1f} s for {n} batches '
          f'({dt / n * 1000:.1f} ms/batch)', flush=True)
    _dump()
    if _STOP_AFTER and len(_BLOCKS) >= _STOP_AFTER:
        print(f'  CACE_STOP_AFTER={_STOP_AFTER} reached; stopping before block '
              f'{len(_BLOCKS) + 1} - the schedule did NOT complete', flush=True)
        raise SystemExit(0)


cace.tools.setup_logger(level='INFO')
cutoff = 5.5

logging.info("reading data")
collection = cace.tasks.get_dataset_from_xyz(train_path='../../data/slab-fps-n-500.xyz',
                                 valid_fraction=0.1,
                                 seed=1,
                                 cutoff=cutoff,
                                 data_key={'energy': 'energy', 'forces':'forces'}, 
                                 atomic_energies= {1: -187.42397696905275, 8: -93.71198848452647}# avg
                                 )
batch_size = 1

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              )

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=1,
                              )

use_device = os.environ.get('CACE_DEVICE', 'cuda')
device = cace.tools.init_device(use_device)
logging.info(f"device: {use_device}")


logging.info("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
#cutoff_fn = CosineCutoff(cutoff=cutoff)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[1,8],
    n_atom_basis=3,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_nu=3,
    num_message_passing=0,
    type_message_passing=['Bchi'],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    device=device,
    timeit=False
           )

cace_representation.to(device)
logging.info(f"Representation: {cace_representation}")

atomwise = cace.modules.atomwise.Atomwise(n_layers=3,
                                         output_key='CACE_energy',
                                         n_hidden=[32,16],
                                         use_batchnorm=False,
                                         add_linear_nn=True)


forces = cace.modules.forces.Forces(energy_key='CACE_energy',
                                    forces_key='CACE_forces')

logging.info("building CACE NNP")
cace_nnp_sr = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[atomwise, forces]
)


q = cace.modules.Atomwise(
    n_layers=3,
    n_hidden=[24,12],
    n_out=4,
    per_atom_output_key='q',
    output_key = 'tot_q',
    residual=False,
    add_linear_nn=True,
    bias=False)

ep = cace.modules.EwaldPotential(dl=2,
                    sigma=1,
                    feature_key='q',
                    output_key='ewald_potential',
                   aggregation_mode='sum')

forces_lr = cace.modules.Forces(energy_key='ewald_potential',
                                    forces_key='ewald_forces')

cace_nnp_lr = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[q, ep, forces_lr]
)

pot2 = {'CACE_energy': 'ewald_potential', 
        'CACE_forces': 'ewald_forces',
        'weight': 0.02
       }

pot1 = {'CACE_energy': 'CACE_energy', 
        'CACE_forces': 'CACE_forces',
       }

cace_nnp = cace.models.CombinePotential([cace_nnp_sr, cace_nnp_lr], [pot1, pot2])
cace_nnp.to(device)


logging.info(f"First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.01
)

force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

from cace.tools import Metrics

e_metric = Metrics(
    target_name='energy',
    predict_name='CACE_energy',
    name='e/atom',
    per_atom=True
)

f_metric = Metrics(
    target_name='forces',
    predict_name='CACE_forces',
    name='f'
)

# Example usage
logging.info("creating training task")

optimizer_args = {'lr': 1e-2, 'betas': (0.99, 0.999)}  
scheduler_args = {'step_size': 20, 'gamma': 0.5}

for i in range(5):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss, force_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=False, #True,
        ema_start=10,
        warmup_steps=5,
    )

    logging.info("training")
    _fit('phase0.%d' % i, task, 40)

task.save_model('water-model.pth')
cace_nnp.to(device)

logging.info(f"Second train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1
)

task.update_loss([energy_loss, force_loss])
logging.info("training")
_fit('w_e=1', task, 100)


task.save_model('water-model-2.pth')
cace_nnp.to(device)

logging.info(f"Third train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10 
)

task.update_loss([energy_loss, force_loss])
_fit('w_e=10', task, 100)

task.save_model('water-model-3.pth')

logging.info(f"Fourth train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

task.update_loss([energy_loss, force_loss])
_fit('w_e=1000', task, 100)

task.save_model('water-model-4.pth')

logging.info(f"Finished")


trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
logging.info(f"Number of trainable parameters: {trainable_params}")
_dump(finished=True)



