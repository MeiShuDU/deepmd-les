"""Read and validate the JSON configuration for the DeepMD-CACE trainer."""

import json
import math
import os

import numpy as np

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
    weight = float(equalization.get("regularization_weight", 0.1))
    if not math.isfinite(weight) or weight < 0 or (equalization.get("learnable_kappa", False) and weight == 0):
        raise ValueError(
            "regularization_weight must be finite and nonnegative, and positive for learnable_kappa initialization"
        )
    if not isinstance(equalization.get("learnable_kappa", False), bool):
        raise ValueError("learnable_kappa must be boolean")
    axis = equalization.get("slab_axis")
    if axis is not None and (type(axis) is not int or axis not in (0, 1, 2)):
        raise ValueError("slab_axis must be 0, 1, 2 or null")
    if not math.isfinite(float(equalization.get("total_charge", 0.0))):
        raise ValueError("total_charge must be finite")
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


def _resolve_charge_systems(equalization, root, training):
    """Resolve fixed one-based groups once, embedding portable solver metadata."""
    specs = equalization.get("systems")
    if specs is None:
        return
    if not isinstance(specs, list) or not specs:
        raise ValueError("charge_eq_latent.systems must be a non-empty list")
    seen = set()
    resolved = []
    for system_id, spec in enumerate(specs):
        label = f"charge_eq_latent.systems[{system_id}]"
        if not isinstance(spec, dict) or not isinstance(spec.get("system"), str):
            raise ValueError(f"{label} requires a system path")
        aliases = spec.get("system_aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(x, str) for x in aliases):
            raise ValueError(f"{label}.system_aliases must be a list of paths")
        paths = [os.path.realpath(os.path.join(root, x)) for x in [spec["system"], *aliases]]
        if len(set(paths)) != len(paths) or seen.intersection(paths):
            raise ValueError(f"{label} has duplicate or ambiguous system paths")
        seen.update(paths)
        atom_types = np.loadtxt(os.path.join(paths[0], "type.raw"), dtype=np.int64, ndmin=1).reshape(-1)
        n = atom_types.size
        if not n:
            raise ValueError(f"{label} has no atoms")
        for alias in paths[1:]:
            other = np.loadtxt(os.path.join(alias, "type.raw"), dtype=np.int64, ndmin=1).reshape(-1)
            if not np.array_equal(atom_types, other):
                raise ValueError(f"{label} aliases must preserve atom count and type ordering")
        filename = spec.get("group_definition")
        if not isinstance(filename, str):
            raise ValueError(f"{label} requires group_definition")
        filename = os.path.realpath(os.path.join(root, filename))
        with open(filename, encoding="utf-8") as stream:
            groups = json.load(stream)
        if not isinstance(groups, dict) or "all" in groups:
            raise ValueError(f"{filename}: groups must be an object; 'all' is reserved")
        for name, indices in groups.items():
            if (not isinstance(indices, list) or not indices or
                    any(type(i) is not int or not 1 <= i <= n for i in indices) or
                    len(set(indices)) != len(indices)):
                raise ValueError(f"{filename}: group {name} requires unique one-based indices in [1, {n}]")
            groups[name] = [i - 1 for i in indices]
        groups["all"] = list(range(n))
        mode = spec.get("electrode_mode")
        electrodes = spec.get("electrode_constraints", [])
        globals_ = spec.get("global_constraints", [])
        if mode not in ("CONP", "CONQ"):
            raise ValueError(f"{label}.electrode_mode must be CONP or CONQ")
        if not isinstance(electrodes, list) or not electrodes or not isinstance(globals_, list):
            raise ValueError(f"{label} requires electrode_constraints and a global_constraints list")
        constraints, electrode_atoms = [], set()
        for is_electrode, items in ((True, electrodes), (False, globals_)):
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError(f"{label}: constraints must be objects")
                kind = item.get("type")
                if kind != (mode if is_electrode else "CONQ"):
                    raise ValueError(f"{label}: electrode constraints must use {mode}; global constraints must use CONQ")
                keys = ("type", "group", "total_charge") if kind == "CONQ" else ("type", "group1", "group2", "delta_V")
                if set(item) != set(keys):
                    raise ValueError(f"{label}: {kind} requires exactly {keys}")
                names = [item["group"]] if kind == "CONQ" else [item["group1"], item["group2"]]
                if any(not isinstance(name, str) or name not in groups for name in names):
                    raise ValueError(f"{label}: unknown constraint group {names}")
                if is_electrode and "all" in names:
                    raise ValueError(f"{label}: 'all' belongs in global_constraints")
                if kind == "CONP" and set(groups[names[0]]).intersection(groups[names[1]]):
                    raise ValueError(f"{label}: CONP electrode groups must be disjoint")
                if is_electrode:
                    electrode_atoms.update(i for name in names for i in groups[name])
                elif names[0] != "all" and electrode_atoms.intersection(groups[names[0]]):
                    raise ValueError(f"{label}: global group must be 'all' or outside electrode atoms")
                target_key = "total_charge" if kind == "CONQ" else "delta_V"
                target = item[target_key]
                if isinstance(target, bool) or not isinstance(target, (float, int)) or not math.isfinite(target):
                    raise ValueError(f"{label}: {target_key} must be a finite number")
                row = dict(item, indices=groups[names[0]])
                if kind == "CONP":
                    row["indices2"] = groups[names[1]]
                constraints.append(row)
        charge_rows = [np.isin(np.arange(n), item["indices"]).astype(float)
                       for item in constraints if item["type"] == "CONQ"]
        if charge_rows and np.linalg.matrix_rank(np.stack(charge_rows)) != len(charge_rows):
            raise ValueError(f"{label}: redundant or contradictory CONQ constraints")
        if equalization.get("slab_axis") is not None and not any(
                item["type"] == "CONQ" and item.get("group") == "all" and item["total_charge"] == 0
                for item in constraints):
            raise ValueError(f"{label}: slab correction requires global CONQ neutrality")
        resolved.append(dict(spec, system=paths[0], system_aliases=paths[1:],
                             group_definition=filename, n_atoms=int(n), constraints=constraints))
    required = {os.path.realpath(p) for split in ("training_data", "validation_data")
                for p in training[split]["systems"]}
    if required - seen:
        raise ValueError("missing charge_eq_latent system definitions for: " + ", ".join(sorted(required - seen)))
    equalization["systems"] = resolved


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
    equalization = long_range.get("charge_eq_latent", {})
    if equalization.get("enabled", False):
        _resolve_charge_systems(equalization, root, training)
    config["_config_path"] = path
    return config
