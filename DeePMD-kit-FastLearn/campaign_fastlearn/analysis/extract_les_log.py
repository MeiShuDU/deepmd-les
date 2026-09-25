"""Parse the LES verbose logs into one tidy TSV for plotting.

`les_params.verbose: true, log_freq: 100` makes the LES module emit, every 100
training steps:

    Training steps :N
    latent_charges mean: <m>, std: <s>
    \tQ_MEAN\tQ_STD
    1\t<tensor([+0.3604], ...)>\t<tensor([0.0019], ...)>     # keyed by atomic
    8\t<tensor([-0.9098], ...)>\t<tensor([0.0031], ...)>     # number: 1 = H, 8 = O
    E_lr = tensor([<f0>, <f1>], ...)                          # one value per frame
    Q layer params (norm summary):
      atomwise.outnet.0.linear.weight: |w|=3.260910e+00 (train)
      ...

The per-type Q_MEAN is the O/H charge the net is producing (the diagnostic for
whether the charge channel is learning anything), and the |w| norms show whether
the charge net's weights are moving. `latent_charges mean/std` is over all atoms
in the batch; the per-type table is the informative one.

Writes analysis/data/les_metrics.tsv with one row per (arm, logged step).
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "data", "les")
OUT = os.path.join(HERE, "data", "les_metrics.tsv")

RE_STEPS = re.compile(r"Training steps :(\d+)")
RE_LATENT = re.compile(r"latent_charges mean: ([-\d.eE+]+), std: ([-\d.eE+]+)")
RE_TYPE = re.compile(
    r"^INFO:les\.les:(\d+)\ttensor\(\[([-\d.eE+]+)\].*?\ttensor\(\[([-\d.eE+]+)\]"
)
RE_ELR = re.compile(r"E_lr = tensor\(\[([-\d.eE+, ]+)\]")
RE_W = re.compile(r"INFO:les\.les:\s+(\S+?): \|w\|=([\d.eE+-]+) \(")
RE_CONT = re.compile(r"^\s+\S")


def logical_lines(fh):
    """Yield one line per log record, re-joining torch's wrapped tensor reprs.

    torch's repr linewidth is 80, so a tensor repr that reaches 80 columns is
    split and its tail is written indented on the next physical line. The learned
    arms hit this on the per-type O row and only there: its mean carries a minus
    sign, which is the character that pushes that repr to 80 columns while the
    positive H row stays at 79. `RE_TYPE` is anchored with `^`, so the wrapped O
    record was dropped outright - `qO_mean` came out empty for every learned arm
    and populated only for freeze-charge, whose charges are buffers, so they print
    with no `grad_fn` and never reach the width at all.

    Every record starts with the `INFO:les.les:` prefix; a wrapped tail starts
    with whitespace instead. Re-joining on that distinction restores one line per
    record before any regex runs.
    """
    buf = None
    for line in fh:
        if buf is not None and RE_CONT.match(line):
            buf = buf.rstrip("\n") + " " + line.lstrip()
        else:
            if buf is not None:
                yield buf
            buf = line
    if buf is not None:
        yield buf


def parse(path):
    rows = []
    cur = None
    with open(path, errors="replace") as fh:
        for line in logical_lines(fh):
            m = RE_STEPS.search(line)
            if m:
                if cur:
                    rows.append(cur)
                cur = {"step": int(m.group(1)), "e_lr_mean": None}
                continue
            if cur is None:
                continue
            m = RE_LATENT.search(line)
            if m:
                cur["q_mean"], cur["q_std"] = float(m.group(1)), float(m.group(2))
                continue
            m = RE_TYPE.match(line)
            if m:
                z = m.group(1)
                tag = {"1": "H", "8": "O"}.get(z, f"Z{z}")
                cur[f"q{tag}_mean"] = float(m.group(2))
                cur[f"q{tag}_std"] = float(m.group(3))
                continue
            m = RE_ELR.search(line)
            if m:
                vals = [float(v) for v in m.group(1).split(",") if v.strip()]
                cur["e_lr_mean"] = sum(vals) / len(vals)
                cur["e_lr_nframes"] = len(vals)
                continue
            m = RE_W.search(line)
            if m:
                key = m.group(1).replace(".linear.", "_").replace("atomwise.", "")
                cur[f"w_{key}"] = float(m.group(2))
        if cur:
            rows.append(cur)
    return rows


def main():
    all_rows, cols = [], ["arm", "step"]
    for path in sorted(glob.glob(os.path.join(LOGS, "*", "les.log"))):
        arm = os.path.basename(os.path.dirname(path))
        rows = parse(path)
        if not rows:
            continue  # plain SR arms log nothing here
        for r in rows:
            r["arm"] = arm
            for k in r:
                if k not in cols:
                    cols.append(k)
        all_rows.extend(rows)
        print(f"{arm:32s} {len(rows):4d} logged steps  "
              f"e_lr_mean {rows[0].get('e_lr_mean'):+.3f} -> {rows[-1].get('e_lr_mean'):+.3f}")
    with open(OUT, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in all_rows:
            fh.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
    print(f"\nwrote {OUT}  ({len(all_rows)} rows, {len(cols)} cols)")


if __name__ == "__main__":
    main()
