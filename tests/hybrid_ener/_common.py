"""Shared helpers for the hybrid_ener (DeepMD + LES) verification scripts.

Each check runs against a small synthetic H2O system and a randomly initialised
hybrid_ener model, so no training data or checkpoint is required. The weights do
not need to be physical: the checks compare analytic derivatives (autograd)
against finite differences, which is a property of the code path, not of the
values fitted into it. To run a check against a trained model instead, pass a
DeepMD checkpoint path as the first argument (see each script's usage line).

The LES model type and the surrounding DeepMD model machinery are exercised
exactly as in training: get_model() -> HybridLESModel -> HybridLESAtomicModel.
"""
from typing import Optional, Tuple

import copy

import numpy as np
import torch

from deepmd.utils.argcheck import normalize
from deepmd.pt.model.model import get_model

# Minimal input that normalize() accepts. Only the "model" section is used to
# build the network; the other sections exist because normalize() expects a full
# input document. Paths are placeholders and are never read.
BASE_CONFIG = {
    "model": {
        "type": "hybrid_ener",
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_a",
            "sel": [46, 92],
            "rcut_smth": 0.50,
            "rcut": 6.00,
            "neuron": [25, 50, 100],
            "axis_neuron": 16,
            "resnet_dt": False,
            "seed": 1,
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
            "resnet_dt": True,
            "seed": 1,
        },
        "les_params": {
            "use_atomwise": True,
            "sigma": 1.0,
            "dl": 1.5,
            "use_fixed_atomic_charges": False,
            "verbose": False,
        },
    },
    "learning_rate": {
        "type": "exp",
        "decay_steps": 1000,
        "start_lr": 0.001,
        "stop_lr": 3.51e-8,
    },
    "loss": {
        "type": "ener",
        "start_pref_e": 0.02,
        "limit_pref_e": 1.0,
        "start_pref_f": 1000.0,
        "limit_pref_f": 1.0,
        "start_pref_v": 0.0,
        "limit_pref_v": 0.0,
    },
    "training": {
        "training_data": {"systems": ["."], "batch_size": 1},
        "validation_data": {"systems": ["."], "batch_size": 1, "numb_btch": 1},
        "numb_steps": 1,
        "seed": 10,
    },
}


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Requested device, or cuda when available, else cpu."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    use_fixed_charges: bool = False,
    device: Optional[str] = None,
    seed: int = 0,
):
    """Random-weight hybrid_ener model in float64, eval mode.

    use_fixed_charges toggles les_params.use_fixed_atomic_charges, which adds the
    FixedCharges baseline to the latent charges (H=+1, O=-2 for type_map O,H).
    """
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["model"]["les_params"]["use_fixed_atomic_charges"] = use_fixed_charges
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = get_model(normalize(cfg)["model"])
    model.to(resolve_device(device)).double().eval()
    return model


def build_water(
    n_molecules: int = 2,
    box_len: float = 9.0,
    device: Optional[str] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic H2O system: (coord [1,n,3], atype [1,n], box [1,9]).

    Atoms are O,H,H per molecule to match type_map ["O","H"].
    """
    rng = np.random.RandomState(seed)
    base = np.array(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=np.float64
    )
    coord = np.tile(base, (n_molecules, 1))
    coord += rng.randn(*coord.shape) * 0.05
    coord += np.array([box_len / 2, box_len / 2, box_len / 2])
    atype = np.array([0, 1, 1] * n_molecules, dtype=np.int64)
    box = np.diag([box_len, box_len, box_len]).reshape(1, 9)
    dev = resolve_device(device)
    return (
        torch.tensor(coord).reshape(1, -1, 3).to(dev),
        torch.tensor(atype).reshape(1, -1).to(dev),
        torch.tensor(box).to(dev),
    )


def load_checkpoint(path: str, device: Optional[str] = None):
    """Load a trained hybrid_ener model from a DeepMD .pt checkpoint."""
    from deepmd.pt.train.wrapper import ModelWrapper

    state = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    model = get_model(copy.deepcopy(state["_extra_state"]["model_params"]))
    ModelWrapper(model).load_state_dict(state)
    model.to(resolve_device(device)).eval()
    return model
