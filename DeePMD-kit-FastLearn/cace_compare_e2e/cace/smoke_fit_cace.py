#!/usr/bin/env python
"""Local coding smoke for fit-cace-nnp.py (author's cace-LES reference).

Same architecture and pipeline as fit-cace-nnp.py, but on a 60-frame subset of
water.xyz with 1 epoch per stage, so the whole script can be validated locally
(build -> forward -> backward -> optimizer step -> update_loss -> save) in a
couple of minutes without an HPC run.
"""
import sys

sys.path.append('../cace/')

import logging

import torch

import cace
from cace.representations import Cace
from cace.modules import BesselRBF, PolynomialCutoff
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)

cace.tools.setup_logger(level='INFO')
cutoff = 5.5

logging.info("reading smoke data")
collection = cace.tasks.get_dataset_from_xyz(
    train_path='./smoke_water.xyz',
    valid_fraction=0.1,
    seed=1,
    cutoff=cutoff,
    data_key={'energy': 'energy', 'forces': 'force'},
    atomic_energies={1: -187.6043857100553, 8: -93.80219285502734},
)
batch_size = 2

train_loader = cace.tasks.load_data_loader(collection=collection, data_type='train', batch_size=batch_size)
valid_loader = cace.tasks.load_data_loader(collection=collection, data_type='valid', batch_size=4)

device = cace.tools.init_device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"device: {device}")

radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[1, 8], n_atom_basis=3, embed_receiver_nodes=True, cutoff=cutoff,
    cutoff_fn=cutoff_fn, radial_basis=radial_basis, n_radial_basis=12,
    max_l=3, max_nu=3, num_message_passing=0, type_message_passing=['Bchi'],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    device=device, timeit=False,
)
cace_representation.to(device)

atomwise = cace.modules.atomwise.Atomwise(
    n_layers=3, output_key='CACE_energy', n_hidden=[32, 16], use_batchnorm=False, add_linear_nn=True)
forces = cace.modules.forces.Forces(energy_key='CACE_energy', forces_key='CACE_forces')

cace_nnp_sr = NeuralNetworkPotential(input_modules=None, representation=cace_representation,
                                     output_modules=[atomwise, forces])

q = cace.modules.Atomwise(n_layers=3, n_hidden=[24, 12], n_out=4, per_atom_output_key='q',
                          output_key='tot_q', residual=False, add_linear_nn=True, bias=False)
ep = cace.modules.EwaldPotential(dl=2, sigma=1., feature_key='q', output_key='ewald_potential',
                                 remove_self_interaction=False, aggregation_mode='sum')
forces_lr = cace.modules.Forces(energy_key='ewald_potential', forces_key='ewald_forces')

cace_nnp_lr = NeuralNetworkPotential(input_modules=None, representation=cace_representation,
                                     output_modules=[q, ep, forces_lr])

pot2 = {'CACE_energy': 'ewald_potential', 'CACE_forces': 'ewald_forces', 'weight': 0.01}
pot1 = {'CACE_energy': 'CACE_energy', 'CACE_forces': 'CACE_forces'}

cace_nnp = cace.models.CombinePotential([cace_nnp_sr, cace_nnp_lr], [pot1, pot2])
cace_nnp.to(device)


def nparam(mod):
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


logging.info(f"params: SR={nparam(cace_nnp_sr)} LR={nparam(cace_nnp_lr)} total={nparam(cace_nnp)}")

energy_loss = cace.tasks.GetLoss(target_name='energy', predict_name='CACE_energy',
                                 loss_fn=torch.nn.MSELoss(), loss_weight=0.1)
force_loss = cace.tasks.GetLoss(target_name='forces', predict_name='CACE_forces',
                                loss_fn=torch.nn.MSELoss(), loss_weight=1000)

from cace.tools import Metrics

e_metric = Metrics(target_name='energy', predict_name='CACE_energy', name='e/atom', per_atom=True)
f_metric = Metrics(target_name='forces', predict_name='CACE_forces', name='f')

optimizer_args = {'lr': 1e-2, 'betas': (0.99, 0.999)}
scheduler_args = {'step_size': 20, 'gamma': 0.5}

# Two stages (vs the reference's five) with 1 epoch each: exercises the
# task.fit loop and the task.update_loss path between stages.
for i in range(2):
    task = TrainingTask(model=cace_nnp, losses=[energy_loss, force_loss], metrics=[e_metric, f_metric],
                        device=device, optimizer_args=optimizer_args,
                        scheduler_cls=torch.optim.lr_scheduler.StepLR, scheduler_args=scheduler_args,
                        max_grad_norm=10, ema=False, ema_start=10, warmup_steps=5)
    logging.info(f"stage {i}: training 1 epoch")
    task.fit(train_loader, valid_loader, epochs=1, screen_nan=False, val_stride=1)
    if i == 0:
        energy_loss = cace.tasks.GetLoss(target_name='energy', predict_name='CACE_energy',
                                         loss_fn=torch.nn.MSELoss(), loss_weight=1)
        task.update_loss([energy_loss, force_loss])

task.save_model('smoke_cace_model.pth')

# dump a forward on the valid set to confirm LES outputs + charge stats.
# NOTE: cannot use torch.no_grad() here - cace's Forces module computes forces
# via torch.autograd.grad, so the energy graph must stay live.
cace_nnp.eval()
b = next(iter(valid_loader))
b.to(device)
pred = cace_nnp(b.to_dict(), training=False)
qv = pred['q'].detach().cpu()
logging.info(f"q shape {tuple(qv.shape)} mean {qv.mean():.4f} std {qv.std():.4f} max|q| {qv.abs().max():.4f}")
logging.info(f"E_sr {pred['CACE_energy'].detach().cpu().numpy()[:3]}")
logging.info(f"E_lr {pred['ewald_potential'].detach().cpu().numpy()[:3]}")
logging.info(f"E_ref {b.energy.detach().cpu().numpy()[:3]}")
logging.info("SMOKE OK")
