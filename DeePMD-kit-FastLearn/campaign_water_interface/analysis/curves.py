"""Every arm's validation trace, as one CSV.

The eight train.log files are two different programs' output - cace's published
fit script and the sea runner - but they share the metric names, because the sea
arms run cace's own ``TrainingTask`` and ``Metrics``. Both log each metric twice,
once bare and once behind a logger's timestamp; matching only at the start of a
line keeps the bare copy and drops the other, so no de-duplication state is needed.

Block membership is taken from ``##### Step: 0 `` rather than from the block
banners, because the two programs print those at opposite ends: the cace script
prints ``block N <label>: ... s for M batches`` after the block, the sea runner
prints ``block N (energy weight W) repeat ...`` before it. ``##### Step: 0`` is
printed at the start of every block by the shared cace training loop, so it is the
one marker both logs agree on.

The energy weight is carried through because the schedule's whole shape is the
weight: 40 epochs at 0.01 five times, then 100 epochs each at 1, 10, 1000. A curve
plot that ignores it makes the loss jumps look like instability.
"""
import csv
import re
from pathlib import Path

import arms as arms_module

OUT = Path(__file__).resolve().parent / "curves.csv"

BLOCK_START = re.compile(r"^##### Step: 0 ")
EPOCH = re.compile(r"^Epoch (\d+), Train Loss: ([-\d.eE+]+), Val Loss: ([-\d.eE+]+)")
VAL_E = re.compile(r"^val_e/atom_rmse:\s*([-\d.eE+]+)")
VAL_F = re.compile(r"^val_f_rmse:\s*([-\d.eE+]+)")
#  sea:  block 3 (energy weight 0.01) repeat 3/5, 40 epochs, fresh task: True
SEA_BLOCK = re.compile(r"^block (\d+) \(energy weight ([-\d.eE+]+)\)")
#  cace:   block 1 phase0.0: 6088.1 s for 18000 batches (338.2 ms/batch)
CACE_BLOCK = re.compile(r"^\s+block (\d+) (\S+):\s*[-\d.]+ s for (\d+) batches \(([-\d.]+) ms/batch\)")
#  the phase-0 blocks are labelled phase0.N and carry the loss weight the script
#  builds its first energy loss with (fit-cace-nnp.py: loss_weight=0.01)
PHASE0_WEIGHT = 0.01


def _cace_weight(label):
    match = re.search(r"w_e=([-\d.eE+]+)", label)
    if match:
        return float(match.group(1))
    if label.startswith("phase0"):
        return PHASE0_WEIGHT
    return float("nan")


def parse_log(path):
    """Rows for one arm. Returns (rows, block_info)."""
    rows = []
    block_info = {}          # block number -> {"label":..., "weight":..., "batches":..., "ms_per_batch":...}
    block = 0
    current = None

    def flush():
        if current and current["val_e_atom_rmse"] is not None:
            rows.append(current)

    with open(path, encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if BLOCK_START.match(line):
                flush()
                current = None
                block += 1
                continue

            match = SEA_BLOCK.match(line)
            if match:
                block_info.setdefault(int(match.group(1)), {})["weight"] = float(match.group(2))
                continue

            match = CACE_BLOCK.match(line)
            if match:
                info = block_info.setdefault(int(match.group(1)), {})
                info["label"] = match.group(2)
                info["weight"] = _cace_weight(match.group(2))
                info["batches"] = int(match.group(3))
                info["ms_per_batch"] = float(match.group(4))
                continue

            match = EPOCH.match(line)
            if match:
                flush()
                current = {
                    "block": block,
                    "epoch": int(match.group(1)),
                    "train_loss": float(match.group(2)),
                    "val_loss": float(match.group(3)),
                    "val_e_atom_rmse": None,
                    "val_f_rmse": None,
                }
                continue

            if current is None:
                continue
            match = VAL_E.match(line)
            if match:
                current["val_e_atom_rmse"] = float(match.group(1))
                continue
            match = VAL_F.match(line)
            if match:
                current["val_f_rmse"] = float(match.group(1))
    flush()

    # global epoch: the per-block counter restarts, the cumulative one does not
    seen = 0
    for row in rows:
        if row["epoch"] == 1:
            block_start = seen + 1
        seen = block_start + row["epoch"] - 1
        row["global_epoch"] = seen
    return rows, block_info


def main():
    rows = []
    print(f"{'arm':12s} {'blocks':>6s} {'epochs':>6s} {'weights':<28s} {'last val_e/atom':>15s} {'last val_f':>11s}")
    for arm in arms_module.ARMS:
        parsed, blocks = parse_log(arm.train_log)
        weights = ",".join(f"{blocks[k].get('weight', float('nan')):g}" for k in sorted(blocks))
        for row in parsed:
            rows.append({
                "family": arm.family,
                "topology": arm.topology,
                "rep": arm.rep,
                "arm": arm.arm,
                "arm_id": arm.arm_id,
                "block": row["block"],
                "energy_weight": blocks.get(row["block"], {}).get("weight", float("nan")),
                "global_epoch": row["global_epoch"],
                "block_epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "val_loss": row["val_loss"],
                "val_e_atom_rmse": row["val_e_atom_rmse"],
                "val_f_rmse": row["val_f_rmse"],
            })
        last = parsed[-1]
        print(f"{arm.arm_id:12s} {len(blocks):6d} {len(parsed):6d} {weights:<28s} "
              f"{last['val_e_atom_rmse']:15.6e} {last['val_f_rmse']:11.5f}")

    with open(OUT, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
