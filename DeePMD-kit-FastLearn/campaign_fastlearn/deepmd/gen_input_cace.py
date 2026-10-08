"""Generate the ``deepmd_cace`` replicate of ``cace-sea-lr`` + ``charge_eq_latent``.

The arm is one of the FastLearn campaign's LR arms, but trained through
``desc_bridging/deepmd_cace`` (DeepMD ``se_a`` descriptor + CACE heads, JSON config,
``python -m deepmd_cace input.json``) rather than through the native DeepMD-kit
``hybrid_ener``/les path. It is a **replicate of** ``cace/runs/cace-sea-lr``:
every hyperparameter below is copied from the cace source, and the only
deliberate addition is ``long_range.charge_eq_latent``.

Provenance of each block of settings (all read here, not retyped):

  ``SE_A``        parsed out of ``cace/sea_seam.py`` - the ``se_a`` descriptor
                  ``cace-sea-lr`` was built on (``rcut 5.5``, ``sel [39, 73]``,
                  ``neuron [25, 50, 100]``, ``axis_neuron 16``, ``seed 1``,
                  ``type_one_side false``, ``resnet_dt false``).
  recipe          imported from ``cace/fit_cace.py``: ``CUTOFF``,
                  ``TRAIN_BATCH``/``VALID_BATCH``, ``ATOMIC_ENERGIES``,
                  ``ENV_PHASES``, ``FORCE_WEIGHT``, ``PHASE_CKPT``.
  model           ``cace/sea_seam.py:build_model(arm='lr')``: SR head
                  ``Atomwise [32, 16]``, charge head ``Atomwise [24, 12]``
                  ``n_out=1 bias=false``, ``EwaldPotential(dl=2, sigma=1.0,
                  remove_self_interaction=False)`` and ``FeatureAdd``.
  data            ``DeePMD-kit-FastLearn/data`` - 64 H2O (192 atoms) per frame,
                  ``data_0`` + ``data_1`` (both ``set.*``) + ``data_2`` = 320
                  train frames, ``data_3`` = 80 validation frames. This is the
                  same 320/80 split, in the same order, with the same total
                  energies as ``campaign_fastlearn/xyz/{train,valid}.xyz``, which
                  is what the cace arms read; ``--verify`` checks that.

Five things differ from ``cace-sea-lr`` on purpose, all recorded in the README
beside the generated config:

1. ``long_range.charge_eq_latent`` is enabled at ``w = 10`` - the point of the run.
2. ``combine_potentials`` is false, i.e. the ``SR_energy`` + ``ewald_potential``
   -> ``CACE_energy`` ``FeatureAdd`` path inside a single model, which is
   structurally what ``cace-sea-lr`` is. The LR term is therefore added at unit
   weight with no ``weight`` key (cace's ``FeatureAdd`` is a pure add), and the
   BEC scale of this arm is ``sqrt(1.0 * eps_inf) = 1.3342``.
3. ``ewald.remove_self_interaction`` stays ``false``, matching ``cace-sea-lr``
   and cace's own default. ``ChargeEqLatent`` shifts the solve's trust region by
   ``1/(sigma (2 pi)^1.5) = 0.063`` (0.6% at w=10) but is well posed either way.
4. the descriptor keeps ``precision float64`` - ``SE_A`` omits the key, so
   ``cace-sea-lr``'s ``se_a`` ran in double, and so do the campaign's native
   deepmd arms. The CACE heads are float32, exactly as in ``cace-sea-lr``.
   Run it with ``DP_INTERFACE_PREC=high`` (the ``runs_smoke`` ``smoke.sh`` and
   ``run.sh`` both set it; ``run_deepmd_cace.py`` wants ``--precision float64``).
5. ``training.seed = 10`` is ``fit_cace_sea.py``'s default and the seed the
   recorded ``cace-sea-lr`` run used.

Usage:
    python gen_input_cace.py             # write runs/ and runs_smoke/
    python gen_input_cace.py --check     # report drift against what is on disk
    python gen_input_cace.py --verify    # also re-check the data split
"""

import argparse
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
CACE_DIR = os.path.join(CAMPAIGN, "cace")
DATA_ROOT = os.path.join(CAMPAIGN, "..", "data")
for _path in (CACE_DIR, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# stdlib-only at import time (torch/cace are imported inside its main), so this
# is the campaign's own recipe constants without paying for a deepmd import
import fit_cace as FC  # noqa: E402

ARM = "deepmd-cace-eq_sA"
REPLICATE_SEED = 10
CHARGE_EQ_WEIGHT = 10.0
# from runs/<arm>/ and runs_smoke/<arm>/ alike, which sit at the same depth
DATA_REL = os.path.join("..", "..", "..", "..", "data")
TRAIN_DIRS = ("data_0", "data_1", "data_2")
VALID_DIRS = ("data_3",)
EXPECTED = {"train": 320, "valid": 80, "atoms": 192, "O": 64, "H": 128}


def sea_a_options():
    """``SE_A`` out of ``cace/sea_seam.py`` without importing it (it pulls deepmd)."""
    path = os.path.join(CACE_DIR, "sea_seam.py")
    with open(path, encoding="utf-8") as stream:
        tree = ast.parse(stream.read())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(target, "id", None) == "SE_A" for target in node.targets):
            continue
        call = node.value
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "dict":
            return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        return ast.literal_eval(call)
    raise SystemExit(f"SE_A not found in {path}")


def frames_of(system_dir):
    """(frames, natoms, per-element counts) for one DeepMD raw system."""
    import numpy as np

    total = 0
    for set_dir in sorted(
        entry.path
        for entry in os.scandir(system_dir)
        if entry.is_dir() and entry.name.startswith("set.")
    ):
        coord = np.load(os.path.join(set_dir, "coord.npy"), mmap_mode="r")
        total += int(coord.shape[0])
        shape = coord.shape[1]
    types = np.loadtxt(os.path.join(system_dir, "type.raw"), dtype=int).reshape(-1)
    counts = np.bincount(types, minlength=2).tolist()
    return total, int(shape) // 3, counts


def verify_data():
    """Fail loudly if the 64-H2O split is not the one the cace arms trained on."""
    for group, dirs in (("train", TRAIN_DIRS), ("valid", VALID_DIRS)):
        frames = 0
        for name in dirs:
            system_dir = os.path.join(DATA_ROOT, name)
            with open(os.path.join(system_dir, "type_map.raw"), encoding="utf-8") as fh:
                type_map = [line.strip() for line in fh if line.strip()]
            if type_map != ["O", "H"]:
                raise SystemExit(f"{name}/type_map.raw is {type_map}, expected O, H")
            n_frames, natoms, counts = frames_of(system_dir)
            if natoms != EXPECTED["atoms"] or counts != [EXPECTED["O"], EXPECTED["H"]]:
                raise SystemExit(
                    f"{name} is {natoms} atoms {counts}, expected "
                    f"{EXPECTED['atoms']} atoms [{EXPECTED['O']}, {EXPECTED['H']}]"
                )
            frames += n_frames
            print(f"  {name:8s} {n_frames:3d} frames x {natoms} atoms "
                  f"({counts[0]} O + {counts[1]} H)")
        if frames != EXPECTED[group]:
            raise SystemExit(f"{group} has {frames} frames, expected {EXPECTED[group]}")
    print(f"  -> {EXPECTED['train']} train / {EXPECTED['valid']} valid frames, "
          f"{EXPECTED['atoms']} atoms = {EXPECTED['O']} H2O per frame")


def build_config():
    se_a = sea_a_options()
    se_a["precision"] = "float64"

    blocks = []
    for phase, (energy_weight, repeat, epochs) in enumerate(FC.ENV_PHASES):
        blocks.append({
            "energy_weight": energy_weight,
            "repeat": repeat,
            "epochs": epochs,
            # phase 0's five blocks each rebuild the TrainingTask, which is what
            # restarts Adam and the StepLR at lr 1e-2; the later three share the
            # fifth task and only swap the loss. Same split as fit_cace.py.
            "fresh_task": phase == 0,
            "checkpoint": FC.PHASE_CKPT[phase],
        })

    return {
        "model": {
            "type": "deepmd_cace",
            "arm": "lr",
            "type_map": ["O", "H"],
            "descriptor": {"type": "se_a", **se_a},
            "fitting_net": {
                "n_layers": 3,
                "n_hidden": [32, 16],
                "add_linear_nn": True,
                "use_batchnorm": False,
            },
            "long_range": {
                "enabled": True,
                # false -> energy head emits SR_energy and FeatureAdd sums it with
                # ewald_potential at unit weight: cace-sea-lr's own graph
                "combine_potentials": False,
                "charge_net": {
                    "n_layers": 3,
                    "n_hidden": [24, 12],
                    "n_out": 1,
                    "bias": False,
                },
                "ewald": {
                    "dl": 2.0,
                    "sigma": 1.0,
                    "remove_self_interaction": False,
                },
                "charge_eq_latent": {
                    "enabled": True,
                    "regularization_weight": CHARGE_EQ_WEIGHT,
                    "total_charge": 0.0,
                },
            },
        },
        # cace subtracts these from the frame total, so the fitted target is the
        # same residual cace-sea-lr saw. Top-level key: deepmd_cace reads it from
        # the config root, not from the model block.
        "atomic_energies": {
            str(number): energy for number, energy in FC.ATOMIC_ENERGIES.items()
        },
        "loss": {"energy_weight": FC.ENV_PHASES[0][0], "force_weight": FC.FORCE_WEIGHT},
        "training": {
            "training_data": {
                "systems": [os.path.join(DATA_REL, name) for name in TRAIN_DIRS],
                "batch_size": FC.TRAIN_BATCH,
            },
            "validation_data": {
                "systems": [os.path.join(DATA_REL, name) for name in VALID_DIRS],
                "batch_size": FC.VALID_BATCH,
            },
            "optimizer": {"type": "adam", "lr": 0.01, "betas": [0.99, 0.999]},
            "scheduler": {"type": "step", "step_size": 20, "gamma": 0.5},
            "blocks": blocks,
            "seed": REPLICATE_SEED,
            "device": "cuda",
            "max_grad_norm": 10.0,
            "warmup_steps": 5,
            "ema": False,
            "ema_start": 10,
            "disp_freq": 1,
            "save_freq": 10,
            # fit_cace.py calls task.fit(..., screen_nan=False)
            "screen_nan": False,
            # "." keeps best_model.pth / checkpoint.pt next to input.json, which
            # is where check_bec*.py and the cace arms both expect them
            "output_dir": ".",
        },
    }


def dumps(config):
    return json.dumps(config, indent=2) + "\n"


def targets():
    return [
        os.path.join(HERE, "runs", ARM, "input.json"),
        os.path.join(HERE, "runs_smoke", ARM, "input.json"),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="compare with the files on disk instead of writing")
    parser.add_argument("--verify", action="store_true",
                        help="also re-check the data split (reads the npy headers)")
    args = parser.parse_args(argv)

    if args.verify:
        print("data split:")
        verify_data()

    text = dumps(build_config())
    config = json.loads(text)
    blocks = config["training"]["blocks"]
    steps_per_epoch = EXPECTED["train"] // FC.TRAIN_BATCH
    total_steps = sum(block["repeat"] * block["epochs"] for block in blocks)
    total_steps *= steps_per_epoch

    print(f"\n{ARM}: {len(blocks)} blocks, "
          f"{sum(b['repeat'] for b in blocks)} fits, {total_steps} steps "
          f"({steps_per_epoch}/epoch, batch {FC.TRAIN_BATCH} over "
          f"{EXPECTED['train']} frames)")

    stale = []
    for path in targets():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = None
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as stream:
                existing = stream.read()
        if args.check:
            state = "up to date" if existing == text else "DRIFT"
            print(f"  {state:2s} {os.path.relpath(path, HERE)}")
            if existing != text:
                stale.append(path)
            continue
        if existing != text:
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(text)
            print(f"  wrote {os.path.relpath(path, HERE)}")
        else:
            print(f"  unchanged {os.path.relpath(path, HERE)}")

    if args.check and stale:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
