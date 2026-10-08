"""Read and validate the JSON configuration for the DeepMD-CACE trainer."""

import json
import os

from .model import CHARGE_EQ_KEYS, LONG_RANGE_KERNEL_KEYS


def _validate_long_range(long_range):
    """Check the long-range branch and the relationships inside it.

    Everything here is a relation *between* options, which is why it lives at the
    configuration boundary and not in any one module: the kernel, the charge head
    and the equalization module each see only their own slice.
    """
    equalization = long_range.get("charge_eq_latent")
    if equalization is None:
        return
    if not isinstance(equalization, dict):
        raise ValueError("model.long_range.charge_eq_latent must be an object")
    unknown = sorted(set(equalization) - set(CHARGE_EQ_KEYS))
    if unknown:
        raise ValueError(
            "unknown model.long_range.charge_eq_latent option(s): " + ", ".join(unknown)
        )
    if not equalization.get("enabled", False):
        return
    if not long_range.get("enabled", False):
        raise ValueError(
            "model.long_range.charge_eq_latent needs the long-range branch "
            "(set model.arm to 'lr' or model.long_range.enabled to true)"
        )
    if float(equalization.get("regularization_weight", 0.1)) <= 0:
        raise ValueError(
            "model.long_range.charge_eq_latent.regularization_weight must be positive"
        )
    # the equalization module drives EwaldPotential as a kernel, so options that
    # only exist in the kernel's own forward path cannot be honoured
    ewald = long_range.get("ewald", {})
    unsupported = sorted(set(ewald) - set(LONG_RANGE_KERNEL_KEYS))
    if unsupported:
        raise ValueError(
            "model.long_range.ewald option(s) not supported by charge_eq_latent: "
            + ", ".join(unsupported)
        )
    if int(ewald.get("exponent", 1)) != 1:
        raise ValueError("charge_eq_latent implements the electrostatic kernel, exponent must be 1")


def load_config(path):
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)

    model = config.get("model", {})
    training = config.get("training", {})
    descriptor = model.get("descriptor", {})
    type_map = model.get("type_map")
    if model.get("type", "deepmd_cace") != "deepmd_cace":
        raise ValueError("model.type must be 'deepmd_cace'")
    if not isinstance(type_map, list) or not type_map or len(set(type_map)) != len(type_map):
        raise ValueError("model.type_map must be a non-empty list of unique element symbols")
    if descriptor.get("type", "se_a") != "se_a":
        raise ValueError("only DeepMD descriptor type 'se_a' is supported")
    if not descriptor.get("rcut"):
        raise ValueError("model.descriptor.rcut is required")
    arm = model.get("arm")
    if arm is not None and arm not in ("sr", "lr"):
        raise ValueError("model.arm must be 'sr' or 'lr'")
    long_range = model.setdefault("long_range", {})
    if arm is not None:
        expected_long_range = arm == "lr"
        if "enabled" in long_range and bool(long_range["enabled"]) != expected_long_range:
            raise ValueError("model.arm and model.long_range.enabled disagree")
        long_range["enabled"] = expected_long_range
    _validate_long_range(long_range)

    for key in ("training_data", "validation_data"):
        data = training.get(key, {})
        systems = data.get("systems")
        if not isinstance(systems, list) or not systems:
            raise ValueError(f"training.{key}.systems must be a non-empty list")
        if int(data.get("batch_size", 1)) < 1:
            raise ValueError(f"training.{key}.batch_size must be positive")

    blocks = training.get("blocks")
    if blocks is None:
        if int(training.get("epochs", 0)) < 1:
            raise ValueError("training.epochs must be a positive integer when blocks is absent")
    else:
        if not isinstance(blocks, list) or not blocks:
            raise ValueError("training.blocks must be a non-empty list")
        for index, block in enumerate(blocks):
            if int(block.get("repeat", 1)) < 1 or int(block.get("epochs", 0)) < 1:
                raise ValueError(f"training.blocks[{index}] needs positive repeat and epochs")
            if "energy_weight" not in block:
                raise ValueError(f"training.blocks[{index}].energy_weight is required")
        if not blocks[0].get("fresh_task", True):
            raise ValueError("the first training block must use fresh_task=true")

    root = os.path.dirname(path)
    config["atomic_energies"] = {
        int(atomic_number): float(energy)
        for atomic_number, energy in config.get("atomic_energies", {}).items()
    }
    for split in ("training_data", "validation_data"):
        systems = training[split]["systems"]
        training[split]["systems"] = [
            os.path.abspath(os.path.join(root, item)) for item in systems
        ]
    training["output_dir"] = os.path.abspath(
        os.path.join(root, training.get("output_dir", "./cace_run"))
    )
    config["_config_path"] = path
    return config