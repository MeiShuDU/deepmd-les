#!/usr/bin/env python3
"""Plot CACE and DeepMD validation learning curves for the water interface campaign."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


STEP_RE = re.compile(r"^##### Step:\s*(\d+)")
ENERGY_RE = re.compile(r"^val_e/atom_rmse:\s*([0-9.eE+-]+)")
FORCE_RE = re.compile(r"^val_f_rmse:\s*([0-9.eE+-]+)")
LCURVE_RE = re.compile(
    r"^\s*(\d+)\s+\S+\s+\S+\s+([0-9.eE+-]+|nan)\s+\S+\s+([0-9.eE+-]+|nan)"
)


COLORS = {
    "cace-sr": "#4C72B0",
    "cace-lr": "#C44E52",
    "deepmd": "#55A868",
    "deepmd-les": "#DD8452",
}


def finite(value: str) -> float:
    return float(value)


def cace_points(log_path: Path, steps_per_epoch: int = 450):
    """Read CACE metrics and turn each block-local epoch into a global step."""
    records = []
    current_step = None
    energy = None
    force = None
    for line in log_path.read_text().splitlines():
        match = STEP_RE.match(line)
        if match:
            if current_step is not None and energy is not None and force is not None:
                records.append((current_step, energy, force))
            current_step = int(match.group(1))
            energy = force = None
            continue
        match = ENERGY_RE.match(line)
        if match and current_step is not None:
            energy = finite(match.group(1))
            continue
        match = FORCE_RE.match(line)
        if match and current_step is not None:
            force = finite(match.group(1))
    if current_step is not None and energy is not None and force is not None:
        records.append((current_step, energy, force))

    points = []
    offset = 0
    previous_step = None
    for local_step, energy, force in records:
        if previous_step is not None and local_step <= previous_step:
            offset += (previous_step + 1) * steps_per_epoch
        points.append((offset + local_step * steps_per_epoch, energy, force))
        previous_step = local_step
    return points


def deepmd_points(arm_dir: Path):
    """Read all segment curves and concatenate them using segments.json offsets."""
    schedule = json.loads((arm_dir / "segments.json").read_text())
    segments = {item["n"]: item for item in schedule["segments"]}
    points = []
    offset = 0
    for number in sorted(segments):
        curve_path = arm_dir / f"s{number}" / "lcurve.out"
        if not curve_path.exists():
            break
        segment = segments[number]
        for line in curve_path.read_text().splitlines():
            match = LCURVE_RE.match(line)
            if not match:
                continue
            local_step = int(match.group(1))
            energy = finite(match.group(2))
            force = finite(match.group(3))
            points.append((offset + local_step, energy, force))
        offset += segment["num_steps"]
    return points


def discover_cace(cace_root: Path):
    curves = {}
    for log_path in sorted(cace_root.glob("*/train.log")):
        branch = log_path.parent.name.rsplit("_s", 1)[0]
        curves[log_path.parent.name] = (branch, cace_points(log_path))
    return curves


def discover_deepmd(deepmd_root: Path):
    curves = {}
    for arm_dir in sorted(deepmd_root.glob("*_*")):
        if not (arm_dir / "segments.json").exists():
            continue
        branch = arm_dir.name.rsplit("_s", 1)[0]
        curves[arm_dir.name] = (branch, deepmd_points(arm_dir))
    return curves


def plot_curves(cace_root: Path, deepmd_root: Path, output: Path) -> None:
    curves = {}
    curves.update(discover_cace(cace_root))
    curves.update(discover_deepmd(deepmd_root))

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    metric_names = [("energy", "Validation energy RMSE / atom (eV)"),
                    ("force", "Validation force RMSE (eV/A)")]
    for branch_name, (branch, points) in curves.items():
        if not points:
            continue
        x = [point[0] for point in points]
        y_energy = [point[1] for point in points]
        y_force = [point[2] for point in points]
        replicate = branch_name.rsplit("_s", 1)[-1]
        linestyle = "-" if replicate == "A" else "--"
        label = branch_name.replace("_s", " ")
        axes[0].plot(x, y_energy, color=COLORS[branch], linestyle=linestyle,
                     linewidth=1.4, label=label)
        axes[1].plot(x, y_force, color=COLORS[branch], linestyle=linestyle,
                     linewidth=1.4, label=label)

    for axis, (_, ylabel) in zip(axes, metric_names):
        axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.22)
        axis.legend(ncol=3, frameon=False)
    axes[1].set_xlabel("Global training step (CACE epoch converted at 450 steps/epoch)")
    fig.suptitle("Water-interface validation learning curves")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"saved {output}")
    for name, (branch, points) in curves.items():
        if points:
            print(f"{name}: {len(points)} points, steps {points[0][0]}..{points[-1][0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("learning_curves.png"))
    args = parser.parse_args()
    plot_curves(args.campaign / "cace" / "runs", args.campaign / "deepmd" / "runs",
                args.output)


if __name__ == "__main__":
    main()
