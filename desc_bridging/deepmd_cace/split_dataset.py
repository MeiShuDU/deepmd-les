"""Split a DeepMD system directory into native train and validation systems."""

import argparse
import json
from pathlib import Path

from .data import split_system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="DeepMD system containing set.*/")
    parser.add_argument("--output", default="./dataset")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-frames", type=int,
                        help="randomly select this many source frames before splitting")
    parser.add_argument("--train-frames", type=int,
                        help="number of selected frames to write to train")
    args = parser.parse_args()
    train_count, valid_count, train_indices, valid_indices = split_system(
        args.source,
        f"{args.output}/train",
        f"{args.output}/valid",
        valid_fraction=args.valid_fraction,
        seed=args.seed,
        total_frames=args.total_frames,
        train_frames=args.train_frames,
        return_indices=True,
    )
    manifest = {
        "source": str(Path(args.source).resolve()),
        "seed": args.seed,
        "total_frames": train_count + valid_count,
        "train_frames": train_count,
        "valid_frames": valid_count,
        "train_source_indices": train_indices.tolist(),
        "valid_source_indices": valid_indices.tolist(),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote train={train_count}, valid={valid_count} frames under {args.output}")


if __name__ == "__main__":
    main()
