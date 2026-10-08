#!/usr/bin/env python
# coding: utf-8
"""Drive cace's learning loop with DeepMD's se_a descriptor.

The idea: the two frameworks agree on the only thing a fitting net needs, which
is a flat per-atom feature row, so the descriptor can be swapped without touching
the loop. DeepMD's se_a emits ``[nframes, nloc, ng * axis_neuron]`` and cace's
``Atomwise`` accepts ``[n_nodes, n_in]`` and does its own ``reshape`` - the same
row semantics, and neither side checks anything else about the other.

The one piece of real work is the seam, because the two disagree on how a batch
is shaped. DeepMD wants ``[nframes, nloc, 3]`` plus an explicit extended region
and a type-grouped neighbor list; cace wants ``[n_nodes, 3]`` plus a ``batch``
index and a ``[nframes, 3, 3]`` cell. ``DeepmdSeAInput`` is that seam, written as
a cace *input module* so it runs inside ``NeuralNetworkPotential.forward`` right
after ``initialize_derivatives`` has put ``requires_grad_`` on the positions.
That ordering is what keeps the graph unbroken from the local coordinates through
the descriptor to the energy and on to the forces.

Four things here are not obvious and are each worth a comment in place:

1. cace's ``cell`` is stored flattened as ``[nframes * 3, 3]``, not
   ``[nframes, 3, 3]`` (``EwaldPotential`` does the same ``.view(-1, 3, 3)``).
2. The extended-region builder works in deepmd's global precision (float64) and
   fails on float32; se_a returns float64 while cace is float32 (its triclinic
   Ewald is float32-only). So coordinates go up and features come back down at
   the seam. That cast is the seam's one real compromise: autograd casts the
   gradient back at the same point, so the descriptor's weights stay float64.
3. cace's batches carry ``atomic_numbers`` (Z), not deepmd type ids, so the seam
   derives ``atype`` from the campaign's ``type_map``.
4. cace's ``Forces`` differentiates ``[positions, displacement]`` and gets a
   requires-grad ``displacement`` from ``Preprocess``; nothing else supplies one
   (``torch.zeros_like`` does not inherit ``requires_grad`` in torch 2.8), so a
   model built here must keep ``Preprocess`` first.

A fifth is on the way out rather than the way in: a deepmd network cannot be
pickled, so these arms save checkpoints differently from the campaign's cace arms.
`save_sea_model` and `load_sea_checkpoint` are that pair.
"""
import os
from typing import Dict, Optional

import numpy as np
import torch

from deepmd.pt.model.descriptor.se_a import DescrptSeA
from deepmd.pt.utils import env
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

# The descriptor block of the campaign's deepmd arms, verbatim from
# deepmd/runs/deepmd_sA/input.yaml. `type_one_side` is NOT a key there, and
# argcheck then supplies False while the CLASS default is True - so it is passed
# explicitly here. The trained checkpoint confirms the choice: it holds 4
# embedding networks, which is what type_one_side=False builds for 2 types.
SE_A = dict(
    rcut=5.5, rcut_smth=0.50, sel=[39, 73], neuron=[25, 50, 100],
    axis_neuron=16, resnet_dt=False, seed=1, type_one_side=False,
)
# the campaign's deepmd arms declare type_map ["O", "H"], which is what fixes
# type 0 = O and type 1 = H (cace hands over Z, so the seam has to map)
TYPE_MAP = ("O", "H")


def build_descriptor(**overrides) -> DescrptSeA:
    """A fresh se_a, structurally identical to the campaign's deepmd arms."""
    kw = dict(SE_A)
    kw.update(overrides)
    return DescrptSeA(**kw)


def read_xyz_systems(path: str, type_map=TYPE_MAP) -> list[dict]:
    """The frames of an xyz file as one deepmd stat-input system.

    ``EnvMatStatSe.iter`` reads ``coord`` (``[nframes, nloc, 3]``), ``atype``
    (``[nframes, nloc]``), ``box`` (``[nframes, 9]``) and ``natoms``; the coord
    is NOT flattened, because that iterator reshapes with
    ``coord.shape[0] * coord.shape[1]`` as the atom count. One system holding
    every frame rather than deepmd's ``numb_btch``-batches-per-system is
    deliberate: it makes the statistic a function of the data alone.
    """
    from ase.io import read

    frames = read(path, index=":")
    coord = np.array([f.get_positions() for f in frames], dtype=np.float64)
    atype = np.array(
        [[type_map.index(s) for s in f.get_chemical_symbols()] for f in frames],
        dtype=np.int64,
    )
    box = np.array([f.get_cell()[:].reshape(-1) for f in frames], dtype=np.float64)
    return [{
        "coord": torch.tensor(coord), "atype": torch.tensor(atype),
        "box": torch.tensor(box), "natoms": int(coord.shape[1]),
    }]


def compute_descriptor_stats(descriptor: DescrptSeA, systems: list[dict]) -> DescrptSeA:
    """Install deepmd's env-mat mean/stddev, its own code path over every frame.

    A se_a is not defined without these: the embedding net reads
    ``(env_mat - mean) / stddev``, so leaving the constructor's zeros/ones in
    place would hand the net raw ``~0.15`` magnitudes, which is not the
    configuration deepmd ever trains. The deepmd arms sample ``numb_btch``
    batches per system, so their four checkpoints disagree by ``~1e-5`` on
    ``mean``; using all frames makes this deterministic and lands within
    ``~1e-3`` relative of any one of them.

    The samples are moved to ``env.DEVICE`` first: deepmd's env-mat code builds
    its zero_mean/one_stddev on that global device, so passing CPU tensors on a
    host where cuda is visible mixes devices. There is no per-call device in
    deepmd - to run this on CPU, hide the GPU (``CUDA_VISIBLE_DEVICES=""``) so
    the global agrees.
    """
    moved = [
        {k: (v.to(env.DEVICE) if torch.is_tensor(v) else v) for k, v in sys_.items()}
        for sys_ in systems
    ]
    descriptor.compute_input_stats(moved)
    return descriptor


class DeepmdSeAInput(torch.nn.Module):
    """cace input module: DeepMD extended region + nlist, then se_a.

    Reads cace's batch convention and writes deepmd's featurization back into it
    as ``node_feats``. Both directions of the precision seam (coord up, features
    down) live here and nowhere else.

    ``forward`` takes ``compute_stress``/``compute_virials`` explicitly rather than
    ``**kwargs``, the signature ``Preprocess`` uses, because TorchScript rejects
    variable-arity methods. That is what keeps the model scriptable, so a run here
    still produces the ``best-scripted.pt`` the cace arms produce.
    """

    def __init__(
        self,
        descriptor: DescrptSeA,
        out_key: str = "node_feats",
        type_map=TYPE_MAP,
    ) -> None:
        super().__init__()
        self.descriptor = descriptor
        self.out_key = out_key
        self.type_map = tuple(type_map)
        from ase.data import atomic_numbers

        # Z -> deepmd type id, e.g. O(8) -> 0, H(1) -> 1, as a table indexed by Z.
        # A table rather than a dict because this runs in the scripted forward: a
        # dict lookup by tensor value is not a TorchScript expression, and the
        # numpy-based version of this was what made the class unscriptable. Not
        # persistent, so it stays out of the state dict (the rebuild recipe already
        # records the type_map it is derived from).
        lut = torch.full((max(atomic_numbers.values()) + 1,), -1, dtype=torch.long)
        for sym, z in atomic_numbers.items():
            if sym in self.type_map:
                lut[z] = self.type_map.index(sym)
        self.register_buffer("z_to_type", lut, persistent=False)

    def _atype(self, data: Dict[str, torch.Tensor], nframes: int, nloc: int) -> torch.Tensor:
        """deepmd type ids from the batch's atomic numbers (Z)."""
        at = self.z_to_type[data["atomic_numbers"].reshape(nframes, nloc).long()]
        if bool((at < 0).any()):
            # a Z the type_map does not name: without this the -1 would reach the
            # embedding net as a type id and quietly index the wrong network
            raise ValueError(f"an atom's Z is not in type_map {self.type_map}")
        return at

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        compute_stress: bool = False,
        compute_virials: bool = False,
    ) -> Dict[str, torch.Tensor]:
        pos = data["positions"]
        batch = data["batch"]
        # cace stores the cell flattened as [nframes * 3, 3]
        cell = data["cell"].reshape(-1, 3, 3)
        nframes = int(cell.shape[0])
        n_nodes = int(pos.shape[0])
        nloc = n_nodes // nframes
        if nloc * nframes != n_nodes or int(batch.max()) + 1 != nframes:
            raise ValueError("uniform frames required for the deepmd seam")

        # cace [n_nodes, 3] + batch  ->  deepmd [nframes, nloc, 3] + box, in the
        # precision the extended-region builder insists on
        prec = env.GLOBAL_PT_FLOAT_PRECISION
        coord = pos.reshape(nframes, nloc, 3).to(prec)
        at = self._atype(data, nframes, nloc)
        box = cell.reshape(nframes, 9).to(prec)

        ext_coord, ext_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            coord, at, self.descriptor.get_rcut(), self.descriptor.get_sel(),
            mixed_types=self.descriptor.mixed_types(), box=box,
        )
        # se_a takes (coord_ext, atype_ext, nlist, mapping); [0] is the descriptor
        g1 = self.descriptor(ext_coord, ext_atype, nlist, mapping)[0]
        data[self.out_key] = g1.reshape(n_nodes, -1).to(pos.dtype)
        return data


def forces_modules(model):
    """The model's ``Forces`` modules (exact name match, not ``DirectForces``)."""
    return [m for m in model.modules() if type(m).__name__ == "Forces"]


# Checkpoints written by `save_sea_model` carry this tag. They are NOT cace's own
# whole-module pickles - see `save_sea_model` for why that is not an option here.
REBUILD_FORMAT = "sea-state-dict-1"


def build_model(cace, descriptor: DescrptSeA, device, arm: str, type_map=TYPE_MAP):
    """se_a in a cace input module, then the campaign's own head(s).

    ``arm='sr'`` is the campaign's ``cace-sr`` with the descriptor swapped for
    se_a: ``Atomwise('CACE_energy') -> Forces``. ``arm='lr'`` is ``cace-lr`` with
    the same swap and the same SR head, plus its LES block verbatim:
    ``Atomwise('SR_energy')`` + ``Atomwise('q')`` -> ``EwaldPotential`` ->
    ``FeatureAdd('CACE_energy')`` -> ``Forces``.

    The SR head is the same module in both arms - same width, same seed - so the
    pair differs only by the LES block. `fit_cace_sea.py --check-only` checks that
    by comparing the two arms' initial SR energy, which is why the width is passed
    explicitly rather than adopted from the first batch (see the module docstring).
    """
    se_a_input = DeepmdSeAInput(descriptor, out_key="node_feats", type_map=type_map)
    n_in = descriptor.get_dim_out()

    # Atomwise('CACE_energy') reads node_feats, not ewald_potential; FeatureAdd
    # and Forces read whatever key they are told to
    sr_energy = cace.modules.atomwise.Atomwise(
        n_in=n_in, n_layers=3,
        output_key="CACE_energy" if arm == "sr" else "SR_energy",
        n_hidden=[32, 16], use_batchnorm=False, add_linear_nn=True)
    forces = cace.modules.Forces(energy_key="CACE_energy", forces_key="CACE_forces")

    if arm == "lr":
        q = cace.modules.Atomwise(
            n_in=n_in, n_layers=3, n_hidden=[24, 12], n_out=1,
            per_atom_output_key="q", output_key="tot_q", residual=False,
            add_linear_nn=True, bias=False)
        ep = cace.modules.EwaldPotential(
            dl=2, sigma=1.0, feature_key="q", output_key="ewald_potential",
            remove_self_interaction=False, aggregation_mode="sum")
        e_add = cace.modules.FeatureAdd(
            feature_keys=["SR_energy", "ewald_potential"], output_key="CACE_energy")
        output_modules = [sr_energy, q, ep, e_add, forces]
    else:
        output_modules = [sr_energy, forces]

    from cace.models.atomistic import NeuralNetworkPotential

    return NeuralNetworkPotential(
        representation=None,  # the descriptor IS an input module here
        # Preprocess stays first, exactly as in the campaign's arms: it supplies
        # the requires-grad `displacement` that cace's Forces differentiates.
        # Dropping it makes Forces raise (see the module docstring).
        input_modules=[cace.modules.Preprocess(), se_a_input],
        output_modules=output_modules,
    ).to(device)


def rebuild_spec(arm: str, descriptor: DescrptSeA, seed: int, prov: dict) -> dict:
    """Everything `build_model` needs to be re-run, for the checkpoint to carry."""
    return {
        "format": REBUILD_FORMAT,
        "arm": arm,
        "se_a_block": dict(SE_A),
        "type_map": list(TYPE_MAP),
        "dim_out": descriptor.get_dim_out(),
        "nsel": descriptor.get_nsel(),
        "n_embed_nets": len(descriptor.sea.filter_layers.networks),
        "seed": seed,
        "cace_provenance": prov,
    }


def save_sea_model(model, path: str, rebuild: dict) -> None:
    """Save a checkpoint cace's own ``save_model`` cannot: a state dict + recipe.

    ``cace.tasks.train.TrainingTask.save_model`` pickles the whole module, and a
    deepmd network cannot be pickled at all - ``make_embedding_network`` builds its
    class *inside* the factory, so the instance's type is
    ``make_embedding_network.<locals>.EN`` and pickle cannot name it. deepmd
    therefore never pickles these nets either; its own checkpoints hold a state
    dict plus a params record. This follows that convention.

    It is not a workaround with a cosmetic cost: the state dict is smaller, it
    carries the descriptor's ``mean``/``stddev`` buffers (so the trained stats come
    back with the weights instead of having to be recomputed), and `build_model` is
    deterministic given the arm, so the recipe is all the loader needs.

    NOTE for whoever reads a sea run's directory: ``best_model.pth`` and
    ``model-*.pth`` here are THESE dicts, not the module pickles the cace arms
    wrote. Load them with `load_sea_checkpoint`, not with a bare ``torch.load``.
    """
    torch.save({
        "format": REBUILD_FORMAT,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "rebuild": rebuild,
    }, path)


def load_sea_checkpoint(cace, path: str, device):
    """Rebuild and load a checkpoint written by `save_sea_model`.

    Returns ``(model, rebuild)``. The recipe in the file is the source of truth for
    the architecture, so a checkpoint from the other arm (or from a different se_a
    block) loads as whatever it was trained as, rather than as whatever this file
    happens to define.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    rb = blob.get("rebuild") or {}
    if blob.get("format") != REBUILD_FORMAT or rb.get("format") != REBUILD_FORMAT:
        raise ValueError(
            f"{path} is not a {REBUILD_FORMAT} checkpoint - a cace arm's whole-module "
            f"pickle has to be loaded with torch.load, not with this loader")
    descriptor = build_descriptor(**rb["se_a_block"])
    if (descriptor.get_dim_out(), descriptor.get_nsel()) != (rb["dim_out"], rb["nsel"]):
        raise ValueError(f"{path} was trained on a different se_a than the one built here")
    model = build_model(cace, descriptor, device, rb["arm"], type_map=tuple(rb["type_map"]))
    # strict by default: a state dict missing the descriptor's stats would otherwise
    # restore mean=zeros/stddev=ones and silently evaluate a different descriptor
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, rb


class SeaTermLogger:
    """Per-epoch SR/LR decomposition of the validation split, as a TSV.

    Validation, not train: it is the split the arm did not fit, so the
    decomposition says how the model spends its capacity rather than how well it
    can memorize.

    The force split needs its own passes. cace's ``Forces`` consumes the graph it
    differentiates, so ``F_SR`` and ``F_LR`` cannot be read off one forward; each
    is priced by re-pointing every ``Forces`` module's ``energy_key`` at the named
    sub-energy and reading ``CACE_forces`` from a FRESH batch dict - the same
    mechanism ``score_cace.py`` uses, and the only way to price a named energy in
    this codebase. Three passes per batch, on 20 validation batches, against 160
    training steps per epoch: a few percent of the run.
    """

    HEADER = (
        "global_epoch", "phase", "block", "block_epoch",
        "val_e_sr_mean", "val_e_sr_rms", "val_e_lr_mean", "val_e_lr_std",
        "val_e_lr_rms", "val_q_mean", "val_q_rms", "val_q_max_abs",
        "val_net_charge", "val_sum_q2", "val_f_sr_rms", "val_f_lr_rms",
        "val_f_tot_rms", "val_f_lr_over_f_tot",
    )

    def __init__(
        self,
        path: str,
        device,
        sr_key: str = "SR_energy",
        lr_key: Optional[str] = None,
        q_key: str = "q",
    ) -> None:
        """``sr_key`` is the key holding the short-range energy, which is
        ``CACE_energy`` on the sr arm (its only head) and ``SR_energy`` on the lr
        arm; ``lr_key`` is None there. With no ``lr_key`` the short-range force is
        the total force and no second pass is needed.
        """
        self.path = path
        self.device = device
        self.sr_key = sr_key
        self.lr_key = lr_key
        self.with_lr = lr_key is not None
        self.q_key = q_key
        self.global_epoch = 0
        self.phase = 0
        self.block = 0
        self.block_epoch = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # opened now and flushed per row: a killed run keeps every epoch it
        # already measured, which is the same reason cace checkpoints every 10
        self.fh = open(path, "w")
        self.fh.write("\t".join(self.HEADER) + "\n")
        self.fh.flush()

    def set_block(self, phase: int, block: int) -> None:
        self.phase, self.block, self.block_epoch = phase, block, 0

    def close(self) -> None:
        self.fh.close()

    def _pass(self, model, batch, energy_key: Optional[str]):
        """One forward on a fresh dict, optionally re-pointing the Forces module."""
        ms = forces_modules(model)
        old = [m.energy_key for m in ms]
        if energy_key is not None:
            for m in ms:
                m.energy_key = energy_key
        try:
            bd = batch.to(self.device).to_dict()
            bd["positions"] = bd["positions"].detach().clone().requires_grad_(True)
            with torch.enable_grad():
                return model(bd, training=False)
        finally:
            for m, o in zip(ms, old):
                m.energy_key = o

    def log_epoch(self, model, loader) -> dict:
        self.global_epoch += 1
        self.block_epoch += 1
        e_sr, e_lr, f_sr, f_lr, f_tot, q = [], [], [], [], [], []
        was_training = model.training
        model.eval()
        try:
            for batch in loader:
                out = self._pass(model, batch, None)
                e_sr.append(out[self.sr_key].detach().cpu().numpy().reshape(-1))
                f_tot.append(out["CACE_forces"].detach().cpu().numpy())
                if self.with_lr:
                    e_lr.append(out[self.lr_key].detach().cpu().numpy().reshape(-1))
                    q.append(out[self.q_key].detach().cpu().numpy())
                    f_sr.append(self._pass(model, batch, self.sr_key)["CACE_forces"]
                                .detach().cpu().numpy())
                    f_lr.append(self._pass(model, batch, self.lr_key)["CACE_forces"]
                                .detach().cpu().numpy())
                else:
                    f_sr.append(f_tot[-1])
        finally:
            if was_training:
                model.train()

        e_sr = np.concatenate(e_sr)
        f_tot = np.concatenate(f_tot)
        f_sr = np.concatenate(f_sr)
        row = {
            "global_epoch": self.global_epoch, "phase": self.phase,
            "block": self.block, "block_epoch": self.block_epoch,
            "val_e_sr_mean": float(e_sr.mean()), "val_e_sr_rms": float(np.sqrt((e_sr ** 2).mean())),
            "val_f_sr_rms": float(np.sqrt((f_sr ** 2).mean())),
            "val_f_tot_rms": float(np.sqrt((f_tot ** 2).mean())),
        }
        if self.with_lr:
            e_lr = np.concatenate(e_lr)
            f_lr = np.concatenate(f_lr)
            qf = np.concatenate(q).reshape(len(e_lr), -1)  # [nframes, nloc * channels]
            row.update({
                "val_e_lr_mean": float(e_lr.mean()), "val_e_lr_std": float(e_lr.std()),
                "val_e_lr_rms": float(np.sqrt((e_lr ** 2).mean())),
                "val_q_mean": float(qf.mean()), "val_q_rms": float(np.sqrt((qf ** 2).mean())),
                "val_q_max_abs": float(np.abs(qf).max()),
                "val_net_charge": float(qf.sum(axis=1).mean()),
                "val_sum_q2": float((qf ** 2).sum(axis=1).mean()),
                "val_f_lr_rms": float(np.sqrt((f_lr ** 2).mean())),
                "val_f_lr_over_f_tot": float(
                    np.sqrt((f_lr ** 2).mean()) / np.sqrt((f_tot ** 2).mean())),
            })
        self.fh.write("\t".join(
            "" if row.get(c) is None else f"{row[c]:.8e}" if isinstance(row[c], float)
            else str(row[c]) for c in self.HEADER) + "\n")
        self.fh.flush()
        return row
