#!/usr/bin/env python
"""Drive cace's learning loop with DeepMD's se_a descriptor instead of Cace's.

The idea under test: the two frameworks agree on the only thing a fitting net
needs, which is a flat per-atom feature row, so the descriptor can be swapped
without touching the loop. Concretely, deepmd's se_a emits a
``[nframes, nloc, ng * axis_neuron]`` tensor and cace's ``Atomwise`` accepts
``[n_nodes, n_in]`` and does its own ``reshape`` - the same row semantics, no
semantic check on either side.

The one piece of real work is the seam, because the two disagree on how a batch
is shaped: deepmd wants ``[nframes, nloc, 3]`` plus an explicit extended region
and a type-grouped neighbor list, while cace wants ``[n_nodes, 3]`` plus a
``batch`` index and a ``[nbatch, 3, 3]`` cell. ``DeepmdSeAInput`` below is that
seam: it is a cace *input module*, so it runs inside
``NeuralNetworkPotential.forward`` after ``initialize_derivatives`` has put
``requires_grad_`` on the positions, which is what keeps the graph unbroken from
the local coordinates through the descriptor to the energy and on to the forces.

Assembled model (the lr arm's topology, one head reading the se_a features):

    positions --\
                 > DeepmdSeAInput(se_a, DeepMD nlist) -> node_feats
    atype   ----/                                             |
                        +-------------------------------------+-----+
                        |                                           |
                  Atomwise('SR_energy')                    Atomwise('q', per-atom)
                                                                  |
                                                        EwaldPotential(cell)
                                                                  |
                       FeatureAdd -> CACE_energy <--------------- +
                                        |
                              Forces -> CACE_forces = -dE/dr

Everything downstream of ``node_feats`` is the campaign's own recipe, unchanged:
the same loss modules, the same ``Forces``, the same schedule.

Usage: python stitch_se_a_atomwise.py
"""
import os
import sys
import time

import torch

CACE_ROOT = "/root/app/cace-ts"  # the campaign's recorded cace_root
sys.path.insert(0, CACE_ROOT)
import cace  # noqa: E402

assert os.path.abspath(cace.__file__).startswith(CACE_ROOT + os.sep), cace.__file__

import cace.modules as cm  # noqa: E402
from cace.models.atomistic import NeuralNetworkPotential  # noqa: E402
from cace.tasks.load_data import get_dataset_from_xyz, load_data_loader  # noqa: E402

from deepmd.pt.model.descriptor.se_a import DescrptSeA  # noqa: E402
from deepmd.pt.utils import env  # noqa: E402
from deepmd.pt.utils.nlist import (  # noqa: E402
    extend_input_and_build_neighbor_list,
)

HERE = os.path.dirname(os.path.abspath(__file__))
XYZ = os.path.join(HERE, "..", "xyz")
RUNS = os.path.join(HERE, "runs")
CUTOFF = 5.5
ATOMIC_ENERGIES = {1: -187.6043857100553, 8: -93.80219285502734}
# the campaign's descriptor block, verbatim (see runs/deepmd-lr_sA/input.json)
SE_A = dict(
    rcut=CUTOFF, rcut_smth=0.5, sel=[39, 73], neuron=[25, 50, 100],
    axis_neuron=16, resnet_dt=False, seed=1,
    # NOT the class default (True): without a "type_one_side" key argcheck
    # supplies False, and the campaign ran with 4 embedding nets, not 2.
    type_one_side=False,
)
FORCE_WEIGHT, ENERGY_WEIGHT = 1000.0, 0.1  # the campaign's phase-1 weights
STEPS, LR = 20, 1e-2
torch.set_default_dtype(torch.float32)


class DeepmdSeAInput(torch.nn.Module):
    """cace input module: DeepMD extended region + nlist, then se_a.

    Reads cace's batch convention and writes deepmd's featurization back into it
    as ``node_feats``, in float32, because se_a hands back
    GLOBAL_PT_FLOAT_PRECISION (float64) while cace's modules are float32 (its
    triclinic Ewald is float32-only). That cast is the seam's one real
    compromise: the descriptor's own weights stay float64 and autograd casts the
    gradient back up at the same point.
    """

    def __init__(self, descriptor: DescrptSeA, out_key: str = "node_feats") -> None:
        super().__init__()
        self.descriptor = descriptor
        self.out_key = out_key

    def forward(self, data, **kwargs):  # cace passes compute_stress/virials
        pos = data["positions"]
        batch = data["batch"]
        cell = data["cell"]
        atype = data["atype"]
        # cace stores the cell flattened as [nframes * 3, 3] (EwaldPotential does
        # the same .view(-1, 3, 3)); the frame count comes from the batch index.
        cell = cell.reshape(-1, 3, 3)
        nframes = int(cell.shape[0])
        n_nodes = int(pos.shape[0])
        nloc = n_nodes // nframes
        if nloc * nframes != n_nodes or int(batch.max()) + 1 != nframes:
            raise ValueError("uniform frames required for the deepmd seam")

        # cace [n_nodes, 3] + batch  ->  deepmd [nframes, nloc, 3] + box
        # The extended-region builder works in deepmd's global precision (float64
        # by default), the same cast input_type_cast applies inside a real model.
        prec = env.GLOBAL_PT_FLOAT_PRECISION
        coord = pos.reshape(nframes, nloc, 3).to(prec)
        at = atype.reshape(nframes, nloc)
        box = cell.reshape(nframes, 9).to(prec)

        ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            coord, at, self.descriptor.get_rcut(), self.descriptor.get_sel(),
            mixed_types=self.descriptor.mixed_types(), box=box,
        )
        # se_a wants (coord_ext, atype_ext, nlist, mapping) and returns a 5-tuple
        g1 = self.descriptor(ext_coord, ext_atype, nlist, mapping)[0]
        data[self.out_key] = g1.reshape(n_nodes, -1).to(pos.dtype)
        return data


def build_batch(device):
    collection = get_dataset_from_xyz(
        train_path=os.path.join(XYZ, "train.xyz"),
        valid_path=os.path.join(XYZ, "valid.xyz"), cutoff=CUTOFF,
        data_key={"energy": "E_total", "forces": "force"},
        atomic_energies=ATOMIC_ENERGIES,
    )
    loader = load_data_loader(collection, "valid", 2)
    batch = next(iter(loader)).to(device)
    bd = batch.to_dict()
    bd["positions"] = bd["positions"].detach().clone().requires_grad_(True)
    # deepmd type ids from the cace atomic numbers, type_map ['O', 'H']
    bd["atype"] = (bd["atomic_numbers"] != 8).long()
    return bd


def report(section, ok_list):
    print(f"\n--- {section} ---")
    return ok_list


def main() -> int:
    device = torch.device("cpu")
    cace.tools.setup_logger(level="WARNING")

    descriptor = DescrptSeA(**SE_A).to(device)
    print("1. descriptor (rebuilt from the campaign's descriptor block)")
    print(f"   se_a            dim_out={descriptor.get_dim_out()}  nsel={descriptor.get_nsel()}"
          f"  rcut={descriptor.get_rcut()}  mixed_types={descriptor.mixed_types()}"
          f"  type_one_side={descriptor.sea.type_one_side}")
    print(f"   buffers         mean=zeros stddev=ones (a fresh se_a does NOT NaN without"
          f" compute_input_stats), device={descriptor.sea.mean.device}")

    se_a_input = DeepmdSeAInput(descriptor)
    model = NeuralNetworkPotential(
        representation=None,  # the descriptor IS the input module here
        # Preprocess stays first, exactly as in the campaign's arms: it is what
        # supplies the requires-grad `displacement` tensor (and rebuilds shifts)
        # that cace's Forces needs, and it leaves the positions numerically
        # untouched at zero strain. Dropping it makes Forces raise.
        input_modules=[cm.Preprocess(), se_a_input],
        output_modules=[
            cm.Atomwise(n_in=descriptor.get_dim_out(), n_layers=3, n_hidden=[32, 16],
                        output_key="SR_energy", add_linear_nn=True),
            cm.Atomwise(n_in=descriptor.get_dim_out(), n_layers=3, n_hidden=[24, 12],
                        n_out=1, per_atom_output_key="q", output_key="tot_q",
                        bias=False, add_linear_nn=True),
            cm.EwaldPotential(dl=2, sigma=1.0, feature_key="q",
                              output_key="ewald_potential", remove_self_interaction=False,
                              aggregation_mode="sum"),
            cm.FeatureAdd(feature_keys=["SR_energy", "ewald_potential"],
                          output_key="CACE_energy"),
            cm.Forces(energy_key="CACE_energy", forces_key="CACE_forces"),
        ],
    ).to(device)

    n_desc = sum(p.numel() for p in descriptor.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print("2. assembled cace model")
    print(f"   output modules  {[type(m).__name__ for m in model.output_modules]}")
    print(f"   required_derivatives {model.required_derivatives}")
    print(f"   params: {n_all} total, {n_desc} in se_a, {n_all - n_desc} in the cace heads")
    print(f"   Atomwise n_in   {model.output_modules[0].n_in} (set explicitly, not lazily adopted)")

    bd = build_batch(device)
    out = model(bd, training=True)
    feat = se_a_input(build_batch(device))["node_feats"]
    print("3. data flow, one real batch")
    print(f"   input           positions {tuple(bd['positions'].shape)}"
          f"  batch {tuple(bd['batch'].shape)}  cell {tuple(bd['cell'].shape)}"
          f"  atype {tuple(bd['atype'].shape)}")
    print(f"   se_a features   {tuple(feat.shape)}  ({feat.shape[-1]} = ng {SE_A['neuron'][-1]}"
          f" x axis_neuron {SE_A['axis_neuron']}), dtype {feat.dtype}")
    print(f"   outputs         CACE_energy {tuple(out['CACE_energy'].shape)}"
          f"  CACE_forces {tuple(out['CACE_forces'].shape)}"
          f"  q {tuple(out['q'].shape)}  ewald {tuple(out['ewald_potential'].shape)}")

    ok = []
    dE = float((out["CACE_energy"] - (out["SR_energy"] + out["ewald_potential"])).detach().abs().max())
    print(f"   E == SR + LR: max|d| = {dE:.3e}")
    ok.append(dE < 1e-6)
    print(f"   E_SR = {float(out['SR_energy'].detach().abs().max()):.4e}   "
          f"E_LR = {float(out['ewald_potential'].detach().abs().max()):.4e}"
          f"   max|q| = {float(out['q'].detach().abs().max()):.3e}")
    print("          (init: cace's q head is unbiased, so the LR channel starts near zero;"
          " the campaign starts it that way too)")
    g = -torch.autograd.grad(out["CACE_energy"].sum(), bd["positions"], retain_graph=True)[0]
    dF = float((out["CACE_forces"] - g).detach().abs().max())
    scale = float(out["CACE_forces"].detach().abs().max())
    print(f"4. forces are the energy gradient: |F - (-dE/dr)| = {dF:.3e} (rel {dF / scale:.2e})")
    ok.append(dF < 1e-4 * scale)

    loss = ENERGY_WEIGHT * torch.nn.functional.mse_loss(out["CACE_energy"], bd["energy"]) \
        + FORCE_WEIGHT * torch.nn.functional.mse_loss(out["CACE_forces"], bd["forces"])
    model.zero_grad(set_to_none=True)
    loss.backward()
    live = [p for p in descriptor.parameters() if p.numel() > 0]  # se_a also holds
    dg = [p for p in live if p.grad is not None]  # zero-size compress_* params
    gnorm = sum(float(p.grad.norm() ** 2) for p in dg) ** 0.5
    print(f"5. trainability: force+energy loss back-props into the DeepMD descriptor")
    print(f"   {len(dg)}/{len(live)} non-empty se_a tensors hold a grad"
          f"  ||g|| = {gnorm:.3e}   (loss {float(loss.detach()):.4e})")
    ok.append(len(dg) == len(live) and gnorm > 0)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"6. cace's loop, {STEPS} steps (weights: energy {ENERGY_WEIGHT}, force {FORCE_WEIGHT})")
    t0 = time.time()
    for step in range(STEPS):
        bd = build_batch(device)
        out = model(bd, training=True)
        loss = ENERGY_WEIGHT * torch.nn.functional.mse_loss(out["CACE_energy"], bd["energy"]) \
            + FORCE_WEIGHT * torch.nn.functional.mse_loss(out["CACE_forces"], bd["forces"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # the campaign's max_grad_norm
        opt.step()
        if step in (0, STEPS - 1):
            print(f"   step {step:3d}  loss {float(loss.detach()):.4e}")
    dt = (time.time() - t0) / STEPS
    print(f"   {dt * 1e3:.0f} ms/step on cpu (2 frames x 192 atoms)")
    ok.append(True)

    print(f"\nRESULT: {'PASS' if all(ok) else 'FAIL'}  ({sum(ok)}/{len(ok)} checks)")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
