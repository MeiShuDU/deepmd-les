#!/usr/bin/env python3
"""Inspect a candidate DeepMD or XYZ dataset before an HPC campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def inspect_system(system: Path) -> dict:
    type_ids = np.loadtxt(system / "type.raw", dtype=int).reshape(-1)
    type_map = (system / "type_map.raw").read_text(encoding="utf-8").split()
    sets = sorted(system.glob("set.*"))
    if not sets:
        raise ValueError(f"no set.* directories under {system}")
    counts = []
    has_box = has_energy = has_force = True
    frames = 0
    for set_dir in sets:
        coord = np.load(set_dir / "coord.npy")
        energy = np.load(set_dir / "energy.npy")
        force = np.load(set_dir / "force.npy")
        box = np.load(set_dir / "box.npy")
        nframes = int(energy.shape[0])
        frames += nframes
        counts.append((set_dir.name, nframes, tuple(coord.shape), tuple(force.shape), tuple(box.shape)))
        has_box &= box.shape[0] == nframes
        has_energy &= energy.shape[0] == nframes
        has_force &= force.shape[0] == nframes
    return {
        "system": str(system),
        "frames": frames,
        "atoms": int(type_ids.size),
        "type_map": type_map,
        "type_counts": {str(i): int((type_ids == i).sum()) for i in sorted(set(type_ids))},
        "sets": counts,
        "labels_aligned": bool(has_box and has_energy and has_force),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", type=Path)
    args = parser.parse_args()
    info = inspect_system(args.system.resolve())
    for key, value in info.items():
        print(f"{key}: {value}")
    if not info["labels_aligned"]:
        raise SystemExit("label arrays are not frame-aligned")


if __name__ == "__main__":
    main()
