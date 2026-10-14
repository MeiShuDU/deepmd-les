#!/usr/bin/env python3
"""Prepare and run the DeepMD-CACE campaign locally or on an HPC GPU.

The smoke path copies a deterministic, small subset of a DeepMD system and
runs both descriptor arms through check-only and one-epoch-per-block training.
The formal path uses the full dataset and the schedule from the selected base
configuration.  No remote provisioning is performed by this module.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "desc_bridging" / "input.json"
DEFAULT_DATASET = ROOT / "DeePMD-kit-FastLearn" / "cace_compare_e2e" / "water"


def _system_files(system: Path) -> list[Path]:
    sets = sorted(system.glob("set.*"))
    if not sets:
        raise FileNotFoundError(f"no set.* directory under {system}")
    return sets


def _read_system(system: Path) -> tuple[np.ndarray, dict[str, np.ndarray | str]]:
    type_ids = np.loadtxt(system / "type.raw", dtype=np.int64).reshape(-1)
    type_map = (system / "type_map.raw").read_text(encoding="utf-8")
    arrays: dict[str, list[np.ndarray]] = {"coord": [], "box": [], "energy": [], "force": []}
    for set_dir in _system_files(system):
        for key in arrays:
            path = set_dir / f"{key}.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            arrays[key].append(np.load(path))
    merged = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    merged["type"] = type_ids
    merged["type_map"] = type_map
    return type_ids, merged


def _write_system(system: Path, arrays: dict[str, np.ndarray | str], indices: np.ndarray) -> None:
    system.mkdir(parents=True, exist_ok=True)
    (system / "type.raw").write_text(
        "\n".join(str(value) for value in np.asarray(arrays["type"]).reshape(-1)) + "\n",
        encoding="utf-8",
    )
    (system / "type_map.raw").write_text(str(arrays["type_map"]), encoding="utf-8")
    set_dir = system / "set.000"
    set_dir.mkdir()
    for key in ("coord", "box", "energy", "force"):
        np.save(set_dir / f"{key}.npy", np.asarray(arrays[key])[indices])


def make_smoke_dataset(source: Path, destination: Path, frames: int, valid_frames: int, seed: int) -> None:
    """Create a tiny deterministic train/valid dataset from one DeepMD system."""
    if (source / "train").is_dir():
        source = source / "train"
    _, arrays = _read_system(source)
    nframes = int(np.asarray(arrays["energy"]).shape[0])
    if nframes < frames + valid_frames:
        raise ValueError(
            f"{source} has {nframes} frames; need at least {frames + valid_frames}"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(nframes)
    destination.mkdir(parents=True, exist_ok=True)
    _write_system(destination / "train", arrays, order[:frames])
    _write_system(destination / "valid", arrays, order[frames : frames + valid_frames])
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "seed": seed,
                "train_frames": frames,
                "valid_frames": valid_frames,
                "source_frames": nframes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _config(base: Path, dataset: Path, output: Path, arm: str, device: str) -> dict:
    config = json.loads(base.read_text(encoding="utf-8"))
    config["model"]["arm"] = arm
    config["model"]["long_range"]["enabled"] = arm == "lr"
    config["training"]["training_data"]["systems"] = [str(dataset / "train")]
    config["training"]["validation_data"]["systems"] = [str(dataset / "valid")]
    config["training"]["device"] = device
    config["training"]["output_dir"] = str(output)
    return config


def _write_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _run(config: Path, *, check_only: bool = False, smoke: bool = False,
         env: dict[str, str]) -> None:
    command = [sys.executable, "-m", "deepmd_cace", str(config)]
    if check_only:
        command.append("--check-only")
    if smoke:
        command.append("--smoke")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _dataset_root(source: Path) -> Path:
    """Return a dataset root containing the expected train/valid systems."""
    if (source / "train").is_dir() and (source / "valid").is_dir():
        return source
    if source.name == "train" and (source.parent / "valid").is_dir():
        return source.parent
    raise FileNotFoundError(
        f"dataset must contain train/ and valid/ systems: {source}"
    )


def _git_revision() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "campaign_runs")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    parser.add_argument("--arm", choices=("sr", "lr", "both"), default="both")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--valid-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    base = args.base_config.resolve()
    source = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arms = ("sr", "lr") if args.arm == "both" else (args.arm,)

    if args.mode == "smoke":
        dataset = output / "smoke_dataset"
        if dataset.exists():
            shutil.rmtree(dataset)
        make_smoke_dataset(source, dataset, args.frames, args.valid_frames, args.seed)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "desc_bridging") + os.pathsep + env.get("PYTHONPATH", "")
        if args.device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        dataset = _dataset_root(source)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "desc_bridging") + os.pathsep + env.get("PYTHONPATH", "")

    for arm in arms:
        run_dir = output / f"{args.mode}-{arm}"
        if run_dir.exists() and args.mode == "smoke":
            shutil.rmtree(run_dir)
        config = _config(base, dataset, run_dir, arm, args.device)
        config_path = run_dir / "input.json"
        _write_config(config, config_path)
        if args.mode == "smoke":
            _run(config_path, check_only=True, env=env)
        _run(config_path, smoke=args.mode == "smoke", env=env)
        manifest = {
            "mode": args.mode,
            "arm": arm,
            "device": args.device,
            "seed": args.seed,
            "source_dataset": str(source),
            "effective_dataset": str(dataset),
            "config": str(config_path),
            "git_revision": _git_revision(),
        }
        (run_dir / "campaign_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"completed {args.mode} {arm}: {run_dir}")


if __name__ == "__main__":
    main()
