#!/usr/bin/env python
# coding: utf-8
"""The campaign's cace arms, with DeepMD's se_a in place of cace's Cace.

Two arms, both driven by the same loop as `fit_cace.py`:

  ``--arm sr``  se_a -> DeepmdSeAInput('node_feats') -> Atomwise('CACE_energy')
                -> Forces, i.e. `cace-sr` with the descriptor swapped for se_a
  ``--arm lr``  the same se_a and the same SR head, plus `cace-lr`'s LES block
                verbatim: Atomwise('SR_energy') + Atomwise('q') -> EwaldPotential
                -> FeatureAdd('CACE_energy') -> Forces

The only intended difference from the corresponding `cace-<arm>` run is the
descriptor and the dimensions downstream of it. So the schedule, the losses, the
optimizer, the phases, the data and the guards are all IMPORTED from
`fit_cace.py` rather than retyped - the campaign's own rule ("one script rather
than two so the two families cannot drift apart") applied to this pair.

One unavoidable difference: the checkpoints here are state dicts plus a rebuild
recipe, not cace's whole-module pickles, because a deepmd network cannot be
pickled at all. Load them with `sea_seam.load_sea_checkpoint`, never with a bare
``torch.load``. The reason, and why a state dict is the better artifact anyway
(it carries the descriptor's env-mat statistics with the weights), is in
`sea_seam.save_sea_model`.

What the SR head's input dimension becomes, and why it is passed explicitly:
`DescrptSeA` here has ``neuron [25, 50, 100]`` and ``axis_neuron 16``, so
``dim_out = 100 * 16 = 1600`` over ``nsel = 39 + 73 = 112`` neighbours. cace's
`Atomwise` adopts `n_in` from its first batch when it is not given, which would
hide a seam that emitted the wrong width; the arms therefore pass
``n_in=descriptor.get_dim_out()`` and the smoke run asserts it.

Extra logging, on top of the shipped recipe (which records only the energy and
force losses/metrics): `SeaTermLogger` writes `sea_terms.tsv`, one row per epoch,
holding the validation split's E_sr, E_lr, q and the F_sr / F_lr split. See
`sea_seam.py` for how the force split is priced.

Usage:
    python fit_cace_sea.py --arm sr --out-dir runs/cace-sea-sr
    python fit_cace_sea.py --arm lr --out-dir runs/cace-sea-lr
    python fit_cace_sea.py --arm lr --out-dir runs/_smoke --smoke --device cpu
"""
import argparse
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# the schedule, the losses, the data guard and the provenance helper all come
# from the cace arms' own script, so this pair cannot drift away from them
from fit_cace import (  # noqa: E402
    ATOMIC_ENERGIES, BEST, CUTOFF, ENV_PHASES, FORCE_WEIGHT, PHASE_CKPT,
    TRAIN_BATCH, VALID_BATCH, cace_provenance, guard_batches, sha256,
)
import sea_seam  # noqa: E402

# Two 2-epoch phases for --smoke. Phase 0 exercises the fresh-task path and phase 1
# the update_loss path, which is where the epoch/block bookkeeping could break; and
# 2 epochs is the shortest block that reaches `fit`'s own best-model save, which
# needs `epoch > val_stride` (val_stride is 1, so epoch 1 never saves). That save is
# the one that crashed on the module pickle, so the smoke has to cover it.
SMOKE_PHASES = [(0.1, 1, 2), (1.0, 1, 2)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["lr", "sr"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train", default=os.path.join(HERE, "..", "xyz", "train.xyz"))
    ap.add_argument("--valid", default=os.path.join(HERE, "..", "xyz", "valid.xyz"))
    ap.add_argument("--cace-root", default="/root/app/cace-ts")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=10,
                    help="the cace arms were run unseeded; these arms fix the draw "
                         "so a bad result can be attributed. default mirrors the "
                         "deepmd arms' training seed.")
    ap.add_argument("--check-only", action="store_true",
                    help="load data, apply the guards, build the model, run one "
                         "forward+backward and one term-log epoch, then exit")
    ap.add_argument("--smoke", action="store_true",
                    help="a two-epoch two-phase schedule, for a wiring check")
    args = ap.parse_args()

    # resolve before the chdir below: `out_dir` is used for abspath-joined paths
    # (term log, record) after the chdir, where a relative --out-dir would resolve
    # against itself and land the term log in a nested `runs/x/runs/x`
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)  # the shipped idiom: best_model.pth lands in the CWD

    sys.path.insert(0, args.cace_root)
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    import cace  # noqa: E402
    from cace.tasks.train import TrainingTask  # noqa: E402
    from cace.tasks import GetLoss, get_dataset_from_xyz, load_data_loader  # noqa: E402
    from cace.tools import Metrics  # noqa: E402

    torch.set_default_dtype(torch.float32)
    cace.tools.setup_logger(level="INFO")
    prov = cace_provenance(args.cace_root)
    print(f"cace     {prov['imported_from']}")
    if prov.get("imported_from") and not os.path.abspath(
            prov["imported_from"]).startswith(os.path.abspath(args.cace_root) + os.sep):
        raise SystemExit(f"cace came from {prov['imported_from']}, not {args.cace_root}")

    # fix the draw before anything random is built
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"seed     {args.seed}")

    phases = SMOKE_PHASES if args.smoke else ENV_PHASES
    device = cace.tools.init_device(args.device)
    print(f"device   {args.device}")

    print("reading data")
    collection = get_dataset_from_xyz(
        train_path=args.train, valid_path=args.valid, cutoff=CUTOFF,
        data_key={"energy": "E_total", "forces": "force"},
        atomic_energies=ATOMIC_ENERGIES,
    )
    train_loader = load_data_loader(collection=collection, data_type="train",
                                    batch_size=TRAIN_BATCH)
    valid_loader = load_data_loader(collection=collection, data_type="valid",
                                    batch_size=VALID_BATCH)
    steps_per_epoch = len(train_loader)
    total_steps = sum(n * e for _, n, e in phases) * steps_per_epoch
    print(f"train {len(collection.train)} frames -> {steps_per_epoch} steps/epoch; "
          f"valid {len(collection.valid)} frames -> {len(valid_loader)} batches")
    print(f"schedule: {[(w, n * e) for w, n, e in phases]} epochs -> "
          f"{total_steps} optimizer steps")

    train_probe = guard_batches(train_loader, ("energy", "forces"), "train")
    guard_batches(valid_loader, ("energy", "forces"), "valid")

    print("building the se_a descriptor and its input statistics")
    descriptor = sea_seam.build_descriptor()
    print(f"  se_a            dim_out={descriptor.get_dim_out()} "
          f"nsel={descriptor.get_nsel()} rcut={descriptor.get_rcut()} "
          f"mixed_types={descriptor.mixed_types()} "
          f"type_one_side={descriptor.sea.type_one_side} "
          f"n_embed_nets={len(descriptor.sea.filter_layers.networks)}")
    systems = sea_seam.read_xyz_systems(args.train)
    sea_seam.compute_descriptor_stats(descriptor, systems)
    mean, stddev = descriptor.get_stat_mean_and_stddev()
    print(f"  env-mat stats   mean {tuple(mean.shape)} "
          f"|mean|_max={float(mean.abs().max()):.4e} "
          f"stddev_mean={float(stddev.mean()):.4e} (deepmd's own stat path, "
          f"all {systems[0]['coord'].shape[0]} train frames)")
    # a cheap audit of the stat: the campaign's deepmd arms computed theirs from a
    # numb_btch sample, so they differ slightly from each other and from this
    stat_digest = {
        "mean_abs_max": float(mean.abs().max()),
        "stddev_mean": float(stddev.mean()),
        "sha256_12": sha256_np(mean, stddev),
        "source": "deepmd EnvMatStatSe over every train frame (no sampling)",
    }

    model = sea_seam.build_model(cace, descriptor, device, args.arm)
    # the checkpoint carries this so `load_sea_checkpoint` can re-run build_model
    rebuild = sea_seam.rebuild_spec(args.arm, descriptor, args.seed, prov)
    n_desc = sum(p.numel() for p in descriptor.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    heads = [type(m).__name__ for m in model.output_modules]
    print(f"  output modules  {heads}")
    print(f"  params: {n_all} total, {n_desc} in se_a, {n_all - n_desc} in the heads")
    print(f"  Atomwise n_in   {model.output_modules[0].n_in} "
          f"(dim_out, passed explicitly, not lazily adopted)")
    print(f"  descriptor params dtype {next(descriptor.parameters()).dtype}")

    terms_path = os.path.join(out_dir, "sea_terms.tsv")
    # the sr arm's only head writes CACE_energy; the lr arm splits it into
    # SR_energy + ewald_potential
    term_logger = sea_seam.SeaTermLogger(
        terms_path, device,
        sr_key="CACE_energy" if args.arm == "sr" else "SR_energy",
        lr_key=None if args.arm == "sr" else "ewald_potential")

    class SeaTrainingTask(TrainingTask):
        """The campaign's task, plus one decomposition measurement per epoch.

        It also replaces ``save_model``: cace's version pickles the whole module,
        which a deepmd network cannot survive (see `sea_seam.save_sea_model`). That
        override catches `fit`'s own best-model saves too - `fit` calls
        ``save_model(val_stride=1``, so from epoch 2 on, on every improvement) - and
        those are the saves that matter, since they write ``best_model.pth``.
        """

        def __init__(self, *a, term_logger=None, rebuild=None, **kw):
            super().__init__(*a, **kw)
            self.term_logger = term_logger
            self.rebuild = rebuild

        def validate(self, val_loader, output_index=None):
            val_loss = super().validate(val_loader, output_index=output_index)
            if self.term_logger is not None:
                self.term_logger.log_epoch(self.model, val_loader)
            return val_loss

        def save_model(self, path, device=torch.device("cpu")):
            sea_seam.save_sea_model(self.model, path, self.rebuild)

    if args.check_only:
        batch = train_probe.to(device)
        pred = model(batch.to_dict(), training=True)
        for key in ("CACE_energy", "CACE_forces"):
            t = pred[key].detach()
            print(f"  {key} {tuple(t.shape)}")
        pred["CACE_energy"].sum().backward()
        g = sum(float(p.grad.abs().sum()) for p in model.parameters()
                if p.grad is not None)
        live = [p for p in descriptor.parameters() if p.numel() > 0]
        got = [p for p in live if p.grad is not None]
        print(f"  backward reached the model: sum|grad| = {g:.4e}; "
              f"se_a {len(got)}/{len(live)} non-empty tensors hold a grad")

        # the save/load pair, on the real code path: a module pickle would raise
        # here (this is the check that caught "Can't pickle local object ...EN")
        probe = "checkpoint_probe.pth"
        e_before = pred["CACE_energy"].detach().clone()
        sea_seam.save_sea_model(model, probe, rebuild)
        reloaded, rb = sea_seam.load_sea_checkpoint(cace, probe, device)
        out2 = reloaded(batch.to_dict(), training=False)
        d_e = float((out2["CACE_energy"] - e_before).abs().max())
        print(f"  save/reload   {os.path.getsize(probe)} bytes -> arm {rb['arm']}, "
              f"max|dE| = {d_e:.3e}, se_a mean|.|_max restores to "
              f"{float(reloaded.input_modules[1].descriptor.sea.mean.abs().max()):.4e}")

        # scriptability is a tested property here, not a hope: the cace arm ships a
        # best-scripted.pt and a seam that TorchScript rejects would silently drop it.
        # The script is compared against the eager model on a real batch, because a
        # script that compiles can still be a different model.
        script_ok = True
        try:
            eager_out = reloaded(batch.to_dict(), training=False)
            scripted = torch.jit.script(reloaded)
            scripted.save("checkpoint_probe_scripted.pt")
            s_out = scripted(batch.to_dict(), training=False)
            d_e_s = float((s_out["CACE_energy"] - eager_out["CACE_energy"]).abs().max())
            d_f_s = float((s_out["CACE_forces"] - eager_out["CACE_forces"]).abs().max())
            e_scale = max(1.0, float(eager_out["CACE_energy"].abs().max()))
            f_scale = max(1.0, float(eager_out["CACE_forces"].abs().max()))
            script_ok = d_e_s < 1e-5 * e_scale and d_f_s < 1e-5 * f_scale
            print(f"  scripting     {'OK' if script_ok else 'MISMATCH'} -> "
                  f"checkpoint_probe_scripted.pt; scripted vs eager "
                  f"max|dE| = {d_e_s:.3e}, max|dF| = {d_f_s:.3e}")
        except Exception as exc:  # noqa: BLE001
            script_ok = False
            print(f"  scripting     FAILED: {exc}")

        row = term_logger.log_epoch(model, valid_loader)
        print(f"  term log first row: " + json.dumps(
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}))
        term_logger.close()
        # relative: the energies are O(70) eV and the model is float32, so an
        # absolute epsilon would be either vacuous or flaky
        ok = d_e < 1e-5 * max(1.0, float(e_before.abs().max())) and script_ok
        print(f"check-only: {'OK' if ok else 'FAILED'}, nothing trained")
        return 0 if ok else 1

    force_loss = GetLoss(target_name="forces", predict_name="CACE_forces",
                         loss_fn=torch.nn.MSELoss(), loss_weight=FORCE_WEIGHT)
    e_metric = Metrics(target_name="energy", predict_name="CACE_energy",
                       name="e/atom", per_atom=True)
    f_metric = Metrics(target_name="forces", predict_name="CACE_forces", name="f")
    optimizer_args = {"lr": 1e-2, "betas": (0.99, 0.999)}
    scheduler_args = {"step_size": 20, "gamma": 0.5}
    phase_records = []

    # the shipped shape, reproduced from fit_cace.py: five fresh-task 40-epoch
    # blocks at E weight 0.1, then 1, 10, 1000 sharing the fifth task
    for phase, (e_weight, n_repeat, epochs) in enumerate(phases):
        energy_loss = GetLoss(target_name="energy", predict_name="CACE_energy",
                              loss_fn=torch.nn.MSELoss(), loss_weight=e_weight)
        for rep in range(n_repeat):
            if phase == 0:
                task = SeaTrainingTask(
                    model=model, losses=[energy_loss, force_loss],
                    metrics=[e_metric, f_metric], device=device,
                    optimizer_args=optimizer_args,
                    scheduler_cls=torch.optim.lr_scheduler.StepLR,
                    scheduler_args=scheduler_args, max_grad_norm=10,
                    ema=False, ema_start=10, warmup_steps=5,
                    term_logger=term_logger, rebuild=rebuild)
            else:
                task.update_loss([energy_loss, force_loss])
            print(f"phase {phase} (E weight {e_weight}) block {rep + 1}/{n_repeat}, "
                  f"{epochs} epochs, fresh task: {phase == 0}")
            term_logger.set_block(phase, rep)
            task.fit(train_loader, valid_loader, epochs=epochs, screen_nan=False)
        task.save_model(PHASE_CKPT[phase])
        phase_records.append({"energy_weight": e_weight, "epochs": n_repeat * epochs,
                              "steps": n_repeat * epochs * steps_per_epoch,
                              "ckpt": PHASE_CKPT[phase]})

    term_logger.close()
    print(f"best checkpoint from the final phase: {BEST}")

    if os.path.exists(BEST):
        # a load failure at the end of a ~4 h run has to be loud, so this is
        # outside the try: only the scripting itself is best-effort, as shipped
        best_model, best_rb = sea_seam.load_sea_checkpoint(cace, BEST, device)
        print(f"reloaded {BEST}: arm {best_rb['arm']}, "
              f"{sum(p.numel() for p in best_model.parameters())} params")
        try:
            torch.jit.script(best_model).save("best-scripted.pt")
            print("scripted best -> best-scripted.pt")
        except Exception as exc:  # noqa: BLE001 - scripting is best-effort, as shipped
            print(f"scripting the best model failed (as in the shipped script): {exc}")

    record = {
        "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "arm": f"sea-{args.arm}",
        "descriptor": "deepmd se_a (DeepmdSeAInput)",
        "cace_provenance": prov,
        "train_xyz": os.path.abspath(args.train),
        "train_sha256": sha256(args.train),
        "valid_xyz": os.path.abspath(args.valid),
        "valid_sha256": sha256(args.valid),
        "n_train_frames": len(collection.train),
        "n_valid_frames": len(collection.valid),
        "batch_size": {"train": TRAIN_BATCH, "valid": VALID_BATCH},
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "phases": phase_records,
        "cutoff": CUTOFF,
        "atomic_energies": ATOMIC_ENERGIES,
        "optimizer": optimizer_args,
        "scheduler": {"cls": "StepLR", **scheduler_args},
        "max_grad_norm": 10,
        "warmup_steps": 5,
        "ema": False,
        "dtype": "float32",
        "force_weight": FORCE_WEIGHT,
        "n_trainable_params": int(n_all),
        "data_key": {"energy": "E_total", "forces": "force"},
        "seed": args.seed,
        "seed_note": "the cace-sr/cace-lr arms were run unseeded; these arms fix "
                     "the draw so a result can be attributed to the descriptor",
        # the device is part of the draw, not just of the speed: the se_a is
        # initialized on deepmd's global DEVICE, so a run on a nvidia gpu draws its
        # descriptor weights from the cuda generator and the heads from the cpu one,
        # interleaved differently than on a host with no gpu visible. Same-device
        # runs are reproducible; a cpu check-only will not match a cuda run's
        # first epochs, and is not supposed to.
        "device": args.device,
        "smoke": bool(args.smoke),
        "se_a": {**sea_seam.SE_A, "type_map": list(sea_seam.TYPE_MAP),
                 "dim_out": descriptor.get_dim_out(),
                 "nsel": descriptor.get_nsel(),
                 "n_embed_nets": len(descriptor.sea.filter_layers.networks)},
        "se_a_stats": stat_digest,
        "head_n_in": descriptor.get_dim_out(),
        "armed_vs": "cace-sr" if args.arm == "sr" else "cace-lr",
        "terms_tsv": terms_path,
        "term_columns": list(sea_seam.SeaTermLogger.HEADER),
        "checkpoint_format": sea_seam.REBUILD_FORMAT,
        "checkpoint_note": "best_model.pth and model-*.pth hold a state dict plus a "
                           "rebuild recipe, not cace's whole-module pickle (a deepmd "
                           "network is not picklable); load them with "
                           "sea_seam.load_sea_checkpoint",
        "rebuild": rebuild,
    }
    with open("run_settings.json", "w") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    print("wrote run_settings.json")
    return 0


def sha256_np(*arrays) -> str:
    """A short digest of the descriptor's stat tensors, for the run record."""
    import hashlib

    import numpy as np

    h = hashlib.sha256()
    for a in arrays:
        h.update(np.asarray(a.detach().cpu(), dtype="<f8").tobytes())
    return h.hexdigest()[:12]


if __name__ == "__main__":
    raise SystemExit(main())
