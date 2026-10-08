#!/usr/bin/env python
"""cace + LES end-to-end training on the shared residual-energy dataset.

Mirrors the deepmd hybrid_ener run in ../deepmd/input.json:
  same data/split, batch 4, Adam lr 1e-3 with exponential decay over the run,
  loss weights energy:1 / force:1000, ~2000 optimizer steps (33 epochs x 60).
Short-range capacity matched to the deepmd side (cace ab=8 ~ 346k SR params
vs deepmd fitting [96,96,96] ~ 358k SR params).

The long-range term uses the same les package ``Les`` that the deepmd hybrid
model wraps, with the same arguments (sigma 1.0, dl 1.5, atomwise latent
charges plus the fixed per-element baseline O -1 / H +0.5), so the only thing
that differs between the two runs is the short-range representation fed to LES.

Precision: the les package is float64 throughout (FixedCharges.charge_table is
a hardcoded float64 buffer, ewald kernels follow r.dtype), and the deepmd side
runs float64 (GLOBAL_PT_FLOAT_PRECISION). cace is therefore run in float64 too
by setting the default dtype *before* any data or module is built - cace builds
every float tensor with ``torch.get_default_dtype()``. Mixing dtypes here would
promote the LES graph to float64 inside a float32 cace model and break backward.

Run:  python train_cace.py          # full E2E training + valid dump
      python train_cace.py --smoke  # 2 steps + forward checks, no training
"""
import os
import sys
import logging

import numpy as np
import torch

# float64 must be set before cace builds any tensor/model. See module docstring.
torch.set_default_dtype(torch.float64)

import cace
from cace.representations import Cace
from cace.modules import BesselRBF, PolynomialCutoff
from cace.modules.les_wrapper import LesWrapper
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.load_data import get_dataset_from_xyz, load_data_loader
from cace.tasks.loss import GetLoss
from cace.tasks.train import TrainingTask
from cace.tools import Metrics

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
SMOKE = '--smoke' in sys.argv

cace.tools.setup_logger(level='INFO')

# ----------------------------------------------------------------------------
# shared hyperparameters (kept in sync with ../deepmd/input.json)
# ----------------------------------------------------------------------------
CUTOFF = 6.0
SIGMA, DL = 1.0, 1.5
BATCH = 4
LR0, LR_END = 1e-3, 3.51e-8
EPOCHS = 33                      # 33 * 60 = 1980 steps ~ deepmd's 2000
W_E, W_F = 1.0, 1000.0
SEED = 10

# fixed per-element charge baseline, same as the deepmd les_params path:
# O -1, H +1, then scaled by 0.5 -> O -0.5, H +0.5 (the les default baseline).
LES_ARGS = {
    "use_atomwise": True,
    "use_fixed_atomic_charges": True,
    "fixed_atomic_charges_scaling_factor": 0.5,
    "sigma": SIGMA,
    "dl": DL,
    "verbose": True,
    "log_freq": 100,
}

torch.manual_seed(SEED)
np.random.seed(SEED)


class LesFixed(LesWrapper):
    """LesWrapper whose internal ``les.Les`` is built with our own arguments.

    ``LesWrapper.__init__`` hardcodes ``les_arguments`` (only use_atomwise /
    use_atomic_alpha / use_epsilon_r_scaling), so it cannot enable the fixed
    per-element charge baseline. We keep the wrapper's data-dict plumbing and
    replace only the inner Les, so the cace LES path uses exactly the same
    sigma/dl/fixed-charge configuration as the deepmd hybrid_ener model.
    """

    def __init__(self, feature_key='node_feats', energy_key='LES_energy',
                 charge_key='LES_charge', atomic_number_key='atomic_numbers',
                 les_arguments=None):
        super().__init__(feature_key=feature_key, energy_key=energy_key,
                         charge_key=charge_key, atomic_number_key=atomic_number_key,
                         compute_energy=True, compute_bec=False)
        from les import Les
        self.les = Les(les_arguments=les_arguments or {})


class AddEnergy(torch.nn.Module):
    """total_energy = short-range energy + long-range (LES) energy."""

    def __init__(self, sr_key='CACE_energy', lr_key='LES_energy', out_key='total_energy'):
        super().__init__()
        self.sr_key, self.lr_key, self.out_key = sr_key, lr_key, out_key
        self.model_outputs = [out_key]
        self.required_derivatives = []

    def forward(self, data, **kwargs):
        data[self.out_key] = data[self.sr_key] + data[self.lr_key]
        return data


def build_dataset():
    collection = get_dataset_from_xyz(train_path=BASE + "/train.xyz",
                                      valid_path=BASE + "/valid.xyz",
                                      cutoff=CUTOFF)
    return collection, load_data_loader(collection, "train", BATCH), load_data_loader(collection, "valid", BATCH)


def build_model(device):
    radial_basis = BesselRBF(cutoff=CUTOFF, n_rbf=12, trainable=True)
    cutoff_fn = PolynomialCutoff(cutoff=CUTOFF)
    representation = Cace(
        zs=[1, 8],
        n_atom_basis=8,
        embed_receiver_nodes=True,
        cutoff=CUTOFF,
        cutoff_fn=cutoff_fn,
        radial_basis=radial_basis,
        n_radial_basis=12,
        max_l=3,
        max_nu=3,
        num_message_passing=1,
        type_message_passing=['Bchi'],
        device=device,
        timeit=False,
        forward_features=['atomic_numbers'],
    )
    atomwise = cace.modules.atomwise.Atomwise(n_layers=3,
                                              output_key='CACE_energy',
                                              n_hidden=[32, 16],
                                              use_batchnorm=False,
                                              add_linear_nn=True)
    les = LesFixed(feature_key='node_feats',
                   energy_key='LES_energy',
                   charge_key='LES_charge',
                   atomic_number_key='atomic_numbers',
                   les_arguments=LES_ARGS)
    combine = AddEnergy()
    forces = cace.modules.forces.Forces(energy_key='total_energy',
                                        forces_key='CACE_forces',
                                        calc_stress=False)
    model = NeuralNetworkPotential(input_modules=None,
                                   representation=representation,
                                   output_modules=[atomwise, les, combine, forces])
    return model, representation, atomwise, les


def main():
    device = cace.tools.init_device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"device: {device}  default dtype: {torch.get_default_dtype()}")
    logging.info(f"data dtype check: {torch.tensor([1.0]).dtype}")

    collection, train_loader, valid_loader = build_dataset()
    logging.info(f"train frames={len(collection.train)} valid frames={len(collection.valid)} "
                 f"train batches/epoch={len(train_loader)}")

    # Build and materialize on CPU, then move to `device`. The LES charge MLP is
    # an nn.LazyLinear whose uninitialized parameter reports device=cpu and is
    # not reliably retargeted by `model.to(device)`; materializing with a CPU
    # batch avoids a cuda-input / cpu-weight mismatch inside les.
    model, representation, atomwise, les = build_model(torch.device('cpu'))
    first = next(iter(train_loader))
    model(first.to_dict(), training=False)
    model.to(device)

    def nparam(mod):
        return sum(p.numel() for p in mod.parameters())

    sr = nparam(representation) + nparam(atomwise)
    les_p = nparam(les)
    logging.info(f"params: SR={sr}  LES={les_p}  total={nparam(model)}")

    e_loss = GetLoss(target_name='energy', predict_name='total_energy',
                     loss_fn=torch.nn.MSELoss(), loss_weight=W_E)
    f_loss = GetLoss(target_name='forces', predict_name='CACE_forces',
                     loss_fn=torch.nn.MSELoss(), loss_weight=W_F)
    e_metric = Metrics(target_name='energy', predict_name='total_energy',
                       name='e', per_atom=True)
    f_metric = Metrics(target_name='forces', predict_name='CACE_forces', name='f')

    gamma = (LR_END / LR0) ** (1.0 / EPOCHS)
    task = TrainingTask(model=model, losses=[e_loss, f_loss],
                        metrics=[e_metric, f_metric], device=device,
                        optimizer_args={'lr': LR0},
                        scheduler_cls=torch.optim.lr_scheduler.ExponentialLR,
                        scheduler_args={'gamma': gamma},
                        max_grad_norm=None, ema=False, warmup_steps=0)

    if SMOKE:
        model.eval()
        with torch.no_grad():
            b = next(iter(valid_loader))
            b.to(device)
            bd = b.to_dict()
            pred = model(bd, training=False)
            q = pred['LES_charge'].detach().cpu().numpy()
            z = bd['atomic_numbers'].detach().cpu().numpy()
            for e in [1, 8]:
                sel = z == e
                logging.info(f"Z={e}: n={sel.sum()} q_mean={q[sel].mean():.4f} q_std={q[sel].std():.4f}")
            logging.info(f"E_lr={pred['LES_energy'].detach().cpu().numpy()} "
                         f"E_tot={pred['total_energy'].detach().cpu().numpy()} "
                         f"E_ref={bd['energy'].detach().cpu().numpy()}")
        logging.info("smoke: 2 training steps")
        for i, batch in enumerate(train_loader):
            if i >= 2:
                break
            loss = task.train_step(batch)
            logging.info(f"  step {i}: loss={loss:.4f}")
        logging.info("SMOKE OK")
        return

    logging.info("training")
    task.fit(train_loader, valid_loader, epochs=EPOCHS, screen_nan=False,
             checkpoint_path=None, bestmodel_path=None, print_stride=1)
    task.save_model(HERE + '/cace_les_model.pth', device=torch.device('cpu'))
    logging.info("done; model -> cace_les_model.pth")

    # dump the LES charges q on the valid set for the cross-code consistency check
    model.eval()
    qs, q_ref_at, e_pred, e_ref, f_pred, f_ref = [], [], [], [], [], []
    with torch.no_grad():
        for batch in valid_loader:
            batch.to(device)
            bd = batch.to_dict()
            pred = model(bd, training=False)
            qs.append(pred['LES_charge'].detach().cpu().numpy())
            q_ref_at.append(bd['atomic_numbers'].detach().cpu().numpy())
            e_pred.append(pred['total_energy'].detach().cpu().numpy())
            e_ref.append(bd['energy'].detach().cpu().numpy())
            f_pred.append(pred['CACE_forces'].detach().cpu().numpy())
            f_ref.append(bd['forces'].detach().cpu().numpy())
    np.savez(HERE + '/cace_les_valid.npz',
             q=np.concatenate(qs), atomic_numbers=np.concatenate(q_ref_at),
             e_pred=np.concatenate(e_pred), e_ref=np.concatenate(e_ref),
             f_pred=np.concatenate(f_pred), f_ref=np.concatenate(f_ref))
    logging.info("wrote cace_les_valid.npz (q, energies, forces)")


if __name__ == '__main__':
    main()
