#!/usr/bin/env python
# coding: utf-8
"""CACE-side runners for the water/slab-interface arms (campaign_water_iface).

This is a faithful, re-instrumented port of the author's two shipped scripts:

    cace-lr-fit-datarepo/BingqingCheng-cace-lr-fit-0211150/
        fit-water-interface/fit-interface-mp0/fit-cace-nnp.py       (arm 'lr')
        fit-water-interface/fit-interface-mp0-sr/fit-cace-nnp.py    (arm 'sr')

Every training setting is preserved verbatim; only two real bugs in the shipped
files are fixed, plus instrumentation is added (see below).

Fixes
-----
1. Data path. The shipped scripts read '../waterslab-datasets/slab-fps-n-500.xyz',
   which does not exist anywhere in the repo. The real file is one level up at
   fit-water-interface/slab-fps-n-500.xyz. TRAIN_XYZ below points at it and the
   path plus its sha256 are printed at start.
2. fit-interface-mp0-sr/fit-cace-nnp.py:27 is literally `valid_fraction=VVVVV,`
   -- an unfilled template placeholder, so the shipped SR script is not valid
   Python and cannot have produced the SR checkpoints as written. The intact LR
   script uses 0.1; the SR arm uses 0.1 here too.

Preserved verbatim (do not "improve" these)
-------------------------------------------
cutoff 5.5; Cace(zs=[1,8], n_atom_basis=3, embed_receiver_nodes=True,
cutoff_fn=PolynomialCutoff(5.5), radial_basis=BesselRBF(5.5, n_rbf=6,
trainable=True), n_radial_basis=12, max_l=3, max_nu=3, num_message_passing=0,
type_message_passing=['Bchi'], args_message_passing={'Bchi':{shared_channels:
False, shared_l: False}}); SR Atomwise(n_layers=3, output_key='CACE_energy',
n_hidden=[32,16], use_batchnorm=False, add_linear_nn=True); charge
Atomwise(n_layers=3, n_hidden=[24,12], n_out=4, per_atom_output_key='q',
output_key='tot_q', residual=False, add_linear_nn=True, bias=False);
EwaldPotential(dl=2, sigma=1, feature_key='q', output_key='ewald_potential',
aggregation_mode='sum'); CombinePotential([sr, lr], [pot1, pot2]) with
pot2 = {CACE_energy: ewald_potential, CACE_forces: ewald_forces, weight: 0.02};
energy loss weight 0.01 in the 5x40 loop then 1, 10, 1000 (lr) / 1, 10 (sr);
force loss weight 1000; Adam lr 1e-2 betas (0.99, 0.999); StepLR(step_size=20,
gamma=0.5); max_grad_norm 10; warmup_steps 5; batch_size 1 train and valid;
get_dataset_from_xyz(valid_fraction=0.1, seed=1, cutoff=5.5); atomic_energies
{1: -187.42397696905275, 8: -93.71198848452647}; data_key {'energy':'energy',
'forces':'forces'}; torch.set_default_dtype(torch.float32); device 'cuda'.

remove_self_interaction is deliberately NOT passed to EwaldPotential. The
author's scripts pass none, and the installed cace (editable at /root/app/cace,
see cace_provenance in run_settings.json) defaults it to True, which matches the
shipped checkpoints under fit-interface-mp0/. Adding the argument would change
the physics.

Block structure (preserved exactly)
-----------------------------------
arm 'lr': five separate TrainingTask constructions, each fit(epochs=40), then
          fit(epochs=100) x3 on the fifth task.
arm 'sr': five separate TrainingTask constructions, each fit(epochs=40), then
          fit(epochs=100) x2 on the fifth task.
Each of the five loop iterations builds a FRESH task, so it gets a fresh Adam
state, a fresh StepLR (last_epoch=0) and a fresh warmup -- the base lr is 1e-2
at the start of every one of the five blocks. The later fits reuse the fifth
task, so its Adam state and StepLR last_epoch continue.

Replicates
----------
The train/valid split uses numpy's *local* rng (np.random.default_rng(seed=1)
inside random_train_valid_split), so the split membership is identical for both
replicates and is not touched by torch seeding. The replicate is expressed by
torch.manual_seed(SEEDS[rep]) set at the very top, which controls weight init
and the train-loader shuffle.

Instrumentation (no training setting changed)
--------------------------------------------
* epoch_metrics.tsv  one row per epoch, written incrementally
* timing.json        per-block wall time / s_per_step, plus totals
* run_settings.json  the full settings record for the run
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import socket
import time

import torch

import cace
from cace.representations import Cace
from cace.modules import PolynomialCutoff
from cace.modules import BesselRBF
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask
from cace.tools import Metrics

HERE = os.path.dirname(os.path.abspath(__file__))

# --- the two fixes: real data path, real valid_fraction -----------------------
TRAIN_XYZ = os.path.abspath(
    os.path.join(
        HERE,
        "..",
        "..",
        "cace-lr-fit-datarepo",
        "BingqingCheng-cace-lr-fit-0211150",
        "fit-water-interface",
        "slab-fps-n-500.xyz",
    )
)
VALID_FRACTION = 0.1  # was the literal `VVVVV` in the shipped mp0-sr script

CUTOFF = 5.5
BATCH_SIZE = 1
ATOMIC_ENERGIES = {1: -187.42397696905275, 8: -93.71198848452647}
DATA_KEY = {"energy": "energy", "forces": "forces"}

OPTIMIZER_ARGS = {"lr": 1e-2, "betas": (0.99, 0.999)}
SCHEDULER_ARGS = {"step_size": 20, "gamma": 0.5}
MAX_GRAD_NORM = 10
WARMUP_STEPS = 5
EMA = False
EMA_START = 10
FORCE_WEIGHT = 1000.0

SEEDS = {"sA": 10, "sB": 20}

# block plan: (phase, energy_weight, epochs, fresh_task, save_after)
LOOP_BLOCK = ("loop", 0.01, 40, True, None)
BLOCKS = {
    "lr": [
        LOOP_BLOCK,
        LOOP_BLOCK,
        LOOP_BLOCK,
        LOOP_BLOCK,
        ("loop", 0.01, 40, True, "water-model.pth"),
        ("ew1", 1.0, 100, False, "water-model-2.pth"),
        ("ew10", 10.0, 100, False, "water-model-3.pth"),
        ("ew1000", 1000.0, 100, False, "water-model-4.pth"),
    ],
    "sr": [
        LOOP_BLOCK,
        LOOP_BLOCK,
        LOOP_BLOCK,
        LOOP_BLOCK,
        ("loop", 0.01, 40, True, "water-model.pth"),
        ("ew1", 1.0, 100, False, "water-model-2.pth"),
        ("ew10", 10.0, 100, False, "water-model-3.pth"),
    ],
}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_energy_loss(weight):
    return cace.tasks.GetLoss(
        target_name="energy",
        predict_name="CACE_energy",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=weight,
    )


def make_force_loss():
    return cace.tasks.GetLoss(
        target_name="forces",
        predict_name="CACE_forces",
        loss_fn=torch.nn.MSELoss(),
        loss_weight=FORCE_WEIGHT,
    )


def make_metrics():
    return [
        Metrics(target_name="energy", predict_name="CACE_energy", name="e/atom", per_atom=True),
        Metrics(target_name="forces", predict_name="CACE_forces", name="f"),
    ]


def build_model(device):
    """Build the CACE representation and both NNP heads, exactly as the author does.

    Returns the combined model plus a summary dict for run_settings.json. Both
    the SR and the LR head are always built (the author does this in both
    scripts); which heads are combined decides the arm.
    """
    radial_basis = BesselRBF(cutoff=CUTOFF, n_rbf=6, trainable=True)
    cutoff_fn = PolynomialCutoff(cutoff=CUTOFF)
    representation = Cace(
        zs=[1, 8],
        n_atom_basis=3,
        embed_receiver_nodes=True,
        cutoff=CUTOFF,
        cutoff_fn=cutoff_fn,
        radial_basis=radial_basis,
        n_radial_basis=12,
        max_l=3,
        max_nu=3,
        num_message_passing=0,
        type_message_passing=["Bchi"],
        args_message_passing={"Bchi": {"shared_channels": False, "shared_l": False}},
        device=device,
        timeit=False,
    )
    representation.to(device)

    atomwise = cace.modules.atomwise.Atomwise(
        n_layers=3,
        output_key="CACE_energy",
        n_hidden=[32, 16],
        use_batchnorm=False,
        add_linear_nn=True,
    )
    forces = cace.modules.forces.Forces(energy_key="CACE_energy", forces_key="CACE_forces")
    cace_nnp_sr = NeuralNetworkPotential(
        input_modules=None, representation=representation, output_modules=[atomwise, forces]
    )

    q = cace.modules.Atomwise(
        n_layers=3,
        n_hidden=[24, 12],
        n_out=4,
        per_atom_output_key="q",
        output_key="tot_q",
        residual=False,
        add_linear_nn=True,
        bias=False,
    )
    # NOTE: remove_self_interaction is intentionally not passed; the installed
    # cace defaults it to True (matches the shipped fit-interface-mp0 checkpoints).
    ep = cace.modules.EwaldPotential(
        dl=2,
        sigma=1,
        feature_key="q",
        output_key="ewald_potential",
        aggregation_mode="sum",
    )
    forces_lr = cace.modules.Forces(energy_key="ewald_potential", forces_key="ewald_forces")
    cace_nnp_lr = NeuralNetworkPotential(
        input_modules=None, representation=representation, output_modules=[q, ep, forces_lr]
    )

    pot1 = {"CACE_energy": "CACE_energy", "CACE_forces": "CACE_forces"}
    pot2 = {"CACE_energy": "ewald_potential", "CACE_forces": "ewald_forces", "weight": 0.02}

    summary = {
        "representation": {
            "cls": "Cace",
            "zs": [1, 8],
            "n_atom_basis": 3,
            "embed_receiver_nodes": True,
            "cutoff": CUTOFF,
            "cutoff_fn": "PolynomialCutoff",
            "radial_basis": "BesselRBF(n_rbf=6, trainable=True)",
            "n_radial_basis": 12,
            "max_l": 3,
            "max_nu": 3,
            "num_message_passing": 0,
            "type_message_passing": ["Bchi"],
            "args_message_passing": {"Bchi": {"shared_channels": False, "shared_l": False}},
        },
        "sr_head": {
            "cls": "Atomwise",
            "n_layers": 3,
            "n_hidden": [32, 16],
            "use_batchnorm": False,
            "add_linear_nn": True,
            "output_key": "CACE_energy",
        },
        "lr_head": {
            "cls": "Atomwise",
            "n_layers": 3,
            "n_hidden": [24, 12],
            "n_out": 4,
            "per_atom_output_key": "q",
            "output_key": "tot_q",
            "residual": False,
            "add_linear_nn": True,
            "bias": False,
        },
        "ewald": {
            "cls": "EwaldPotential",
            "dl": 2,
            "sigma": 1,
            "feature_key": "q",
            "output_key": "ewald_potential",
            "aggregation_mode": "sum",
            "remove_self_interaction": True,
            "remove_self_interaction_note": "not passed; installed cace default is True",
        },
        "combine": {
            "cls": "CombinePotential",
            "pot1": pot1,
            "pot2": pot2,
        },
    }
    return cace_nnp_sr, cace_nnp_lr, summary


def combine_for_arm(arm, cace_nnp_sr, cace_nnp_lr):
    pot1 = {"CACE_energy": "CACE_energy", "CACE_forces": "CACE_forces"}
    pot2 = {"CACE_energy": "ewald_potential", "CACE_forces": "ewald_forces", "weight": 0.02}
    if arm == "lr":
        return cace.models.CombinePotential([cace_nnp_sr, cace_nnp_lr], [pot1, pot2])
    return cace.models.CombinePotential([cace_nnp_sr], [pot1])


# ---------------------------------------------------------------------------
# per-epoch instrumentation: wrap the task's train_step/validate once per block
# ---------------------------------------------------------------------------


class EpochRecorder:
    """Records one row per epoch without touching any training setting."""

    def __init__(self, tsv_path):
        self.tsv_path = tsv_path
        self.rows = []
        with open(self.tsv_path, "w") as f:
            f.write(
                "global_epoch\tblock\tblock_epoch\tphase\tenergy_weight\tlr"
                "\ttrain_loss\tval_loss\tseconds\n"
            )

    def install(self, task, block, phase, energy_weight):
        rec = self  # the recorder; `self` inside the wrappers below is the Task
        cls = type(task)  # class-level lookup: subclass-aware and never chains
        state = {"loss_sum": 0.0, "n": 0, "block_epoch": 0, "last_mark": time.time()}

        def train_step(self, batch, screen_nan=True, output_index=None, loss_index=None):
            loss = cls.train_step(
                self, batch, screen_nan=screen_nan, output_index=output_index, loss_index=loss_index
            )
            state["loss_sum"] += loss
            state["n"] += 1
            return loss

        def validate(self, val_loader, output_index=None):
            state["block_epoch"] += 1
            val_loss = cls.validate(self, val_loader, output_index=output_index)
            now = time.time()
            row = {
                "global_epoch": self.global_step + 1,  # global_step increments at epoch end
                "block": block,
                "block_epoch": state["block_epoch"],
                "phase": phase,
                "energy_weight": energy_weight,
                "lr": self.optimizer.param_groups[0]["lr"],
                "train_loss": state["loss_sum"] / max(state["n"], 1),
                "val_loss": val_loss,
                "seconds": now - state["last_mark"],
            }
            state["last_mark"] = now
            state["loss_sum"] = 0.0
            state["n"] = 0
            rec.rows.append(row)
            with open(rec.tsv_path, "a") as f:
                f.write(
                    "{global_epoch}\t{block}\t{block_epoch}\t{phase}\t{energy_weight}"
                    "\t{lr:.10g}\t{train_loss:.10g}\t{val_loss:.10g}\t{seconds:.4f}\n".format(**row)
                )
            return val_loss

        # instance attributes shadow the class methods, so fit() picks these up
        task.train_step = train_step.__get__(task, TrainingTask)
        task.validate = validate.__get__(task, TrainingTask)


def run_block(task, train_loader, val_loader, blk, block_idx, recorder, epochs, smoke):
    phase, w_e, _, fresh, save_after = blk
    n_steps = epochs * len(train_loader)
    recorder.install(task, block_idx, phase, w_e)
    started = time.time()
    ok = True
    try:
        logging.info(
            f"### block {block_idx} phase={phase} ew={w_e} epochs={epochs} "
            f"steps={n_steps} fresh_task={fresh}"
        )
        task.fit(train_loader, val_loader, epochs=epochs, screen_nan=False)
    except Exception:
        ok = False
        raise
    finally:
        elapsed = time.time() - started
        if save_after is not None and ok:
            task.save_model(save_after)
    return {
        "n": block_idx,
        "phase": phase,
        "energy_weight": w_e,
        "epochs": epochs,
        "num_steps": n_steps,
        "fresh_task": fresh,
        "seconds": elapsed,
        "s_per_step": elapsed / n_steps if n_steps else None,
        "started": started,
        "ended": started + elapsed,
        "ok": ok,
        "smoke": smoke,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["lr", "sr"], required=True)
    ap.add_argument("--replicate", choices=sorted(SEEDS), required=True)
    ap.add_argument("--out", required=True, help="run output directory (created)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="shrink every block to 2 epochs")
    ap.add_argument(
        "--check",
        action="store_true",
        help="build data/model/plan and exit without fitting (config validation only)",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)  # fit() writes checkpoint.pt / best_model.pth relative to CWD

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(SEEDS[args.replicate])  # replicate seed: weight init + loader shuffle

    cace.tools.setup_logger(level="INFO")
    logging.info(f"arm={args.arm} replicate={args.replicate} out={args.out}")
    logging.info(f"train_xyz={TRAIN_XYZ}")
    if not os.path.isfile(TRAIN_XYZ):
        raise SystemExit(f"training file not found: {TRAIN_XYZ}")
    xyz_sha = sha256_file(TRAIN_XYZ)
    logging.info(f"train_xyz sha256={xyz_sha}")

    logging.info("reading data")
    collection = cace.tasks.get_dataset_from_xyz(
        train_path=TRAIN_XYZ,
        valid_fraction=VALID_FRACTION,  # fix: shipped mp0-sr had the literal `VVVVV`
        seed=1,
        cutoff=CUTOFF,
        data_key=DATA_KEY,
        atomic_energies=ATOMIC_ENERGIES,
    )
    train_loader = cace.tasks.load_data_loader(
        collection=collection, data_type="train", batch_size=BATCH_SIZE
    )
    valid_loader = cace.tasks.load_data_loader(
        collection=collection, data_type="valid", batch_size=BATCH_SIZE
    )
    n_train = len(collection.train)
    n_valid = len(collection.valid)
    steps_per_epoch = len(train_loader)
    logging.info(f"n_train={n_train} n_valid={n_valid} steps_per_epoch={steps_per_epoch}")

    use_device = args.device
    device = cace.tools.init_device(use_device)
    logging.info(f"device: {use_device}")

    logging.info("building CACE representation")
    cace_nnp_sr, cace_nnp_lr, model_summary = build_model(device)
    cace_nnp = combine_for_arm(args.arm, cace_nnp_sr, cace_nnp_lr)
    cace_nnp.to(device)

    force_loss = make_force_loss()
    metrics = make_metrics()

    plan = BLOCKS[args.arm]
    if args.smoke:
        plan = [(p, w, min(2, e), fr, sv) for (p, w, e, fr, sv) in plan]

    total_epochs = sum(b[2] for b in plan)
    total_steps = total_epochs * steps_per_epoch
    # Fidelity note: Atomwise builds its MLP lazily on the first forward, and the
    # author's scripts pass no `n_in`. Before any forward pass only the
    # representation counts, so the first block's fresh optimizer is built
    # without the output head in its param groups. That quirk is preserved here
    # by keeping the author's construction order; both counts are recorded.
    n_params_repr_only = count_params(cace_nnp)
    logging.info(f"plan: {len(plan)} blocks, {total_epochs} epochs, {total_steps} optimizer steps")

    if args.check:
        # One forward on one real frame, purely to materialize the lazy output
        # heads so the parameter count is meaningful. No optimizer step is taken.
        torch.set_grad_enabled(True)
        batch = next(iter(valid_loader))
        batch.to(device)
        cace_nnp(batch.to_dict(), training=False)
        n_params = count_params(cace_nnp)
        logging.info("--check: config validated, exiting without training")
        print(f"arm={args.arm} replicate={args.replicate}")
        print(f"n_train={n_train} n_valid={n_valid} steps_per_epoch={steps_per_epoch}")
        print(f"blocks={len(plan)} epochs={total_epochs} steps={total_steps}")
        print(f"n_trainable_params={n_params}")
        print(f"n_trainable_params_representation_only={n_params_repr_only}")
        for i, (p, w, e, fr, sv) in enumerate(plan, 1):
            print(f"  block {i}: phase={p} energy_weight={w} epochs={e} fresh_task={fr} save={sv}")
        return

    recorder = EpochRecorder("epoch_metrics.tsv")
    timing = []
    t_start = time.time()
    task = None
    for i, blk in enumerate(plan, start=1):
        phase, w_e, epochs, fresh, _ = blk
        if fresh:
            task = TrainingTask(
                model=cace_nnp,
                losses=[build_energy_loss(w_e), force_loss],
                metrics=metrics,
                device=device,
                optimizer_args=OPTIMIZER_ARGS,
                scheduler_cls=torch.optim.lr_scheduler.StepLR,
                scheduler_args=SCHEDULER_ARGS,
                max_grad_norm=MAX_GRAD_NORM,
                ema=EMA,
                ema_start=EMA_START,
                warmup_steps=WARMUP_STEPS,
            )
        else:
            task.update_loss([build_energy_loss(w_e), force_loss])
        timing.append(run_block(task, train_loader, val_loader, blk, i, recorder, epochs, args.smoke))
        cace_nnp.to(device)

    total_seconds = time.time() - t_start
    # the author's placement: counted after training, when the lazy heads exist
    trainable_params = count_params(cace_nnp)
    with open("timing.json", "w") as f:
        json.dump(
            {
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "device": use_device,
                "total_seconds": total_seconds,
                "total_steps": total_steps,
                "s_per_step": total_seconds / total_steps if total_steps else None,
                "blocks": timing,
            },
            f,
            indent=2,
        )

    phases = [
        {
            "block": i,
            "phase": p,
            "energy_weight": w,
            "epochs": e,
            "steps": e * steps_per_epoch,
            "fresh_task": fr,
            "ckpt": sv,
        }
        for i, (p, w, e, fr, sv) in enumerate(plan, 1)
    ]
    with open("run_settings.json", "w") as f:
        json.dump(
            {
                "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                "arm": f"cace-{args.arm}",
                "replicate": args.replicate,
                "torch_seed": SEEDS[args.replicate],
                "split_seed": 1,
                "cace_provenance": {
                    "imported_from": cace.__file__,
                    "version": getattr(cace, "__version__", None),
                    "requested_root": os.path.dirname(os.path.dirname(cace.__file__)),
                },
                "source_script": "fit-interface-mp0" + ("-sr" if args.arm == "sr" else ""),
                "train_xyz": TRAIN_XYZ,
                "train_sha256": xyz_sha,
                "n_train_frames": n_train,
                "n_valid_frames": n_valid,
                "batch_size": {"train": BATCH_SIZE, "valid": BATCH_SIZE},
                "steps_per_epoch": steps_per_epoch,
                "total_epochs": total_epochs,
                "total_steps": total_steps,
                "phases": phases,
                "cutoff": CUTOFF,
                "atomic_energies": {str(k): v for k, v in ATOMIC_ENERGIES.items()},
                "optimizer": {"lr": OPTIMIZER_ARGS["lr"], "betas": list(OPTIMIZER_ARGS["betas"])},
                "scheduler": {"cls": "StepLR", "step_size": 20, "gamma": 0.5},
                "max_grad_norm": MAX_GRAD_NORM,
                "warmup_steps": WARMUP_STEPS,
                "warmup_unit": "epochs (cace increments global_step per epoch)",
                "ema": EMA,
                "dtype": "float32",
                "force_weight": FORCE_WEIGHT,
                "n_trainable_params": trainable_params,
                "n_trainable_params_representation_only": n_params_repr_only,
                "n_trainable_params_note": (
                    "counted after training, matching the author's script; Atomwise "
                    "builds its MLP lazily on the first forward, so the first block's "
                    "optimizer is built before the output head exists"
                ),
                "data_key": DATA_KEY,
                "valid_fraction": VALID_FRACTION,
                "remove_self_interaction": True,
                "smoke": args.smoke,
                "model": model_summary,
            },
            f,
            indent=2,
        )

    logging.info(f"finished: {total_steps} steps in {total_seconds:.1f}s")
    logging.info(f"Number of trainable parameters: {trainable_params}")


if __name__ == "__main__":
    main()
