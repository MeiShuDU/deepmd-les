"""CACE training entry point for a DeepMD se_a descriptor."""

import argparse
import copy
import datetime as dt
import json
import os
import time

import numpy as np
import torch
import cace
from cace.tasks import GetLoss
from cace.tasks.train import TrainingTask
from cace.tools import Metrics

from .config import load_config
from .data import load_split
from .descriptor import build_descriptor, compute_descriptor_stats
from .model import build_model


OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}
SCHEDULERS = {
    "step": torch.optim.lr_scheduler.StepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
}


def _seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _optimizer(config):
    options = dict(config or {"type": "adam", "lr": 1e-3})
    name = options.pop("type", "adam").lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"unsupported optimizer {name!r}; choose {sorted(OPTIMIZERS)}")
    return OPTIMIZERS[name], options


def _scheduler(config):
    if not config or config.get("type", "none").lower() == "none":
        return None, None
    options = dict(config)
    name = options.pop("type").lower()
    if name not in SCHEDULERS:
        raise ValueError(f"unsupported scheduler {name!r}; choose {sorted(SCHEDULERS)}")
    return SCHEDULERS[name], options


def _assert_targets(loader, split_name):
    batch = next(iter(loader))
    missing = [key for key in ("energy", "forces") if batch[key] is None]
    if missing:
        raise ValueError(f"{split_name} batch is missing labels: {missing}")
    print(f"{split_name}: {len(loader.dataset)} frames, "
          f"energy {tuple(batch['energy'].shape)}, forces {tuple(batch['forces'].shape)}")
    return batch


def _make_losses(config, energy_weight):
    loss_config = config.get("loss", {})
    return [
        GetLoss("energy", "CACE_energy", loss_fn=torch.nn.MSELoss(),
                loss_weight=float(energy_weight)),
        GetLoss("forces", "CACE_forces", loss_fn=torch.nn.MSELoss(),
                loss_weight=float(loss_config.get("force_weight", 1000.0))),
    ]


def _make_task(model, config, device, energy_weight):
    metrics = [
        Metrics("energy", "CACE_energy", name="e/atom", per_atom=True),
        Metrics("forces", "CACE_forces", name="f"),
    ]
    optimizer_cls, optimizer_args = _optimizer(config["training"].get("optimizer"))
    scheduler_cls, scheduler_args = _scheduler(config["training"].get("scheduler"))

    class StateDictTrainingTask(TrainingTask):
        def train_step(self, batch, screen_nan=True, output_index=None, loss_index=None):
            batch.to(self.device)
            batch_dict = batch.to_dict()
            self.train()
            self.optimizer.zero_grad()
            prediction = self.model(batch_dict, training=True, output_index=output_index)
            self.log_metrics("train", prediction, batch_dict)
            loss = self.loss_fn(
                prediction,
                batch_dict,
                {"epochs": self.global_step, "training": True},
                loss_index,
            )
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
            if not screen_nan or finite:
                if self.global_step < self.warmup_steps:
                    lr_scale = min(1.0, float(self.global_step + 1) / self.warmup_steps)
                    for group in self.optimizer.param_groups:
                        group["lr"] = lr_scale * self.lr
                self.optimizer.step()
                if self.ema and self.global_step >= self.ema_start:
                    self.ema_model.update_parameters(self.model)
            return float(loss.detach().cpu())

        def save_model(self, path, device=torch.device("cpu")):
            state = {key: value.detach().cpu()
                     for key, value in self.model.state_dict().items()}
            torch.save({
                "format": "deepmd-cace-state-dict-1",
                "state_dict": state,
                "config": config,
            }, path)

    training_config = config["training"]
    return StateDictTrainingTask(
        model=model,
        losses=_make_losses(config, energy_weight),
        metrics=metrics,
        device=device,
        optimizer_cls=optimizer_cls,
        optimizer_args=optimizer_args,
        scheduler_cls=scheduler_cls,
        scheduler_args=scheduler_args,
        max_grad_norm=float(training_config.get("max_grad_norm", 10.0)),
        warmup_steps=int(training_config.get("warmup_steps", 0)),
        ema=bool(training_config.get("ema", False)),
        ema_start=int(training_config.get("ema_start", 0)),
    )


def _write_block_record(path, record):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def run(config_path, check_only=False, smoke=False):
    config = load_config(config_path)
    model_config = config["model"]
    training_config = config["training"]
    output_dir = training_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    _seed_everything(int(training_config.get("seed", 1)))
    torch.set_default_dtype(torch.float32)
    cace.tools.setup_logger(level=training_config.get("log_level", "INFO"))
    requested_device = os.environ.get(
        "DEEPMD_CACE_DEVICE", training_config.get("device", "auto")
    )
    device = _select_device(requested_device)
    print(f"device: {device}; cace: {os.path.abspath(cace.__file__)}")

    descriptor_config = dict(model_config["descriptor"])
    cutoff = float(descriptor_config["rcut"])
    train_loader, train_stats = load_split(
        training_config["training_data"]["systems"],
        model_config["type_map"],
        cutoff,
        int(training_config["training_data"].get("batch_size", 1)),
        shuffle=True,
        atomic_energies=config.get("atomic_energies"),
    )
    valid_loader, _ = load_split(
        training_config["validation_data"]["systems"],
        model_config["type_map"],
        cutoff,
        int(training_config["validation_data"].get("batch_size", 1)),
        shuffle=False,
        collect_stats=False,
        atomic_energies=config.get("atomic_energies"),
    )
    train_probe = _assert_targets(train_loader, "train")
    _assert_targets(valid_loader, "valid")

    descriptor = build_descriptor(descriptor_config)
    compute_descriptor_stats(
        descriptor,
        train_stats,
        batch_size=int(training_config.get("descriptor_stats_batch_size", 1)),
    )
    mean, stddev = descriptor.get_stat_mean_and_stddev()
    print(f"se_a: dim_out={descriptor.get_dim_out()}, nsel={descriptor.get_nsel()}, "
          f"|mean|max={float(mean.abs().max()):.3e}, "
          f"mean(stddev)={float(stddev.mean()):.3e}")
    model = build_model(model_config, descriptor, device)
    if check_only:
        train_probe.to(device)
        prediction = model(train_probe.to_dict(), training=True)
        loss = prediction["CACE_energy"].square().mean()
        loss = loss + prediction["CACE_forces"].square().mean()
        loss.backward()
        descriptor_grads = [parameter.grad for parameter in descriptor.parameters()
                            if parameter.requires_grad and parameter.numel()]
        active_grads = [grad for grad in descriptor_grads if grad is not None]
        if not active_grads or not all(torch.isfinite(grad).all() for grad in active_grads):
            raise RuntimeError("the check forward did not propagate gradients into se_a")
        print("check-only: forward, force derivative, and descriptor gradients OK; "
              "no training performed")
        return

    blocks = training_config.get("blocks")
    if blocks is None:
        blocks = [{
            "energy_weight": float(config.get("loss", {}).get("energy_weight", 1.0)),
            "repeat": 1,
            "epochs": int(training_config["epochs"]),
            "fresh_task": True,
        }]

    steps_per_epoch = len(train_loader)
    blocks_path = os.path.join(output_dir, "blocks.json")
    timing = {
        "arm": model_config.get("arm", "sr"),
        "replicate_seed": int(training_config.get("seed", 1)),
        "smoke": bool(smoke),
        "steps_per_epoch": steps_per_epoch,
        "max_gpu_mb": None,
        "finished": False,
        "blocks": [],
    }
    _write_block_record(blocks_path, timing)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    task = None
    for block_index, block in enumerate(blocks):
        for repeat_index in range(int(block.get("repeat", 1))):
            fresh_task = bool(block.get("fresh_task", block_index == 0))
            if task is None or fresh_task:
                task = _make_task(model, config, device, block["energy_weight"])
            else:
                task.update_loss(_make_losses(config, block["energy_weight"]))

            block_number = len(timing["blocks"]) + 1
            block_epochs = 1 if smoke else int(block["epochs"])
            started_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            started = time.perf_counter()
            print(
                f"block {block_number} "
                f"(energy weight {block['energy_weight']}) repeat "
                f"{repeat_index + 1}/{block.get('repeat', 1)}, "
                f"{block_epochs} epochs, fresh task: {fresh_task}"
            )
            task.fit(
                train_loader,
                valid_loader,
                epochs=block_epochs,
                checkpoint_path=os.path.join(output_dir, "checkpoint.pt"),
                checkpoint_stride=int(training_config.get("save_freq", 10)),
                bestmodel_path=os.path.join(output_dir, "best_model.pth"),
                print_stride=int(training_config.get("disp_freq", 1)),
                screen_nan=bool(training_config.get("screen_nan", True)),
            )
            seconds = time.perf_counter() - started
            ended_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            num_steps = block_epochs * steps_per_epoch
            timing["blocks"].append({
                "n": block_number,
                "label": f"phase{block_index}.{repeat_index}",
                "energy_weight": float(block["energy_weight"]),
                "epochs": block_epochs,
                "num_steps": num_steps,
                "seconds": round(seconds, 2),
                "s_per_batch": round(seconds / num_steps, 6) if num_steps else None,
                "started": started_utc,
                "ended": ended_utc,
                "fresh_task": fresh_task,
                "ok": True,
            })
            if device.type == "cuda":
                timing["max_gpu_mb"] = round(
                    torch.cuda.max_memory_allocated(device) / 2**20, 1
                )
                torch.cuda.reset_peak_memory_stats(device)
            _write_block_record(blocks_path, timing)

        checkpoint_name = block.get("checkpoint")
        if checkpoint_name:
            task.save_model(os.path.join(output_dir, checkpoint_name))

    if not blocks[-1].get("checkpoint"):
        task.save_model(os.path.join(output_dir, "model.pth"))

    record = copy.deepcopy(config)
    record.pop("_config_path", None)
    timing["finished"] = not smoke
    _write_block_record(blocks_path, timing)
    with open(os.path.join(output_dir, "run_settings.json"), "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(f"saved model and run settings under {output_dir}")
    if smoke:
        print("smoke schedule completed; this is a wiring check, not a trained model")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="DeepMD-style input.json")
    parser.add_argument("--check-only", action="store_true",
                        help="validate data/model and descriptor gradients without training")
    parser.add_argument("--smoke", action="store_true",
                        help="run one epoch per fit block for plumbing/timing checks")
    args = parser.parse_args(argv)
    run(args.config, check_only=args.check_only, smoke=args.smoke)


if __name__ == "__main__":
    main()