"""Split a DeepMD system directory into native train and validation systems."""

import argparse

from .data import split_system


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="DeepMD system containing set.*/")
    parser.add_argument("--output", default="./dataset")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    train_count, valid_count = split_system(
        args.source,
        f"{args.output}/train",
        f"{args.output}/valid",
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )
    print(f"wrote train={train_count}, valid={valid_count} frames under {args.output}")


if __name__ == "__main__":
    main()