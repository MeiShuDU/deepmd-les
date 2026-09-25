"""How much of a trained LES model's total force is the long-range term?

The cace arms report `f_lr_over_f_tot` (the share of the total force norm carried
by the Ewald head's forces), which is a mechanism number, not an accuracy number.
This measures the same quantity on the deepmd side so the two can be laid side by
side, over the whole 80-frame validation split rather than a couple of frames.

`lr_weight` scales `E_lr` *before* the coordinate gradient that produces the
long-range force (`hybridles_model.py`: the scaling sits above the
`torch.autograd.grad([E_lr_total], [coord])` call), so `lr_weight = 0` removes both
the long-range energy and its force exactly, with the short-range net untouched.
Nothing is re-trained, so this is the decomposition the trained model actually
uses:

    E_LR = E(w=1) - E(w=0)      the model's own long-range energy
    F_LR = F(w=1) - F(w=0)      its long-range force

Writes a TSV for the notebook. Run from the repo root.
"""
import argparse
import os

import numpy as np

from deepmd.common import j_loader
from deepmd.infer.deep_pot import DeepPot

CAMPAIGN = ("/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/"
            "campaign_fastlearn")
DEFAULT_RUNS = os.path.join(CAMPAIGN, "deepmd", "runs")
DEFAULT_OUT = os.path.join(CAMPAIGN, "analysis", "data", "lr_mechanism_les.tsv")
COLS = ["arm", "step", "nframes", "e_sr_mean", "e_lr_mean", "e_lr_std",
        "f_tot_rms", "f_sr_rms", "f_lr_rms", "f_lr_over_f_tot", "e_self_mean",
        "f_self_rms", "f_self_over_f_tot", "self_over_e_lr", "sum_q2_mean",
        "rsi_trained"]


def measure(rundir, step, nframes):
    arm = os.path.basename(rundir.rstrip("/"))
    ckpt = os.path.join(rundir, f"model.ckpt-{step}.pt")
    if not os.path.exists(ckpt):
        print(f"skip {arm}: no {os.path.basename(ckpt)}")
        return None
    jdata = j_loader(os.path.join(rundir, "input.yaml"))
    systems = jdata["training"]["validation_data"]["systems"]
    system = systems[0] if isinstance(systems, list) else systems
    if not os.path.isabs(system):
        system = os.path.normpath(os.path.join(rundir, system))
    coord = np.load(os.path.join(system, "set.000", "coord.npy"))
    box = np.load(os.path.join(system, "set.000", "box.npy"))
    types = np.loadtxt(os.path.join(system, "type.raw")).astype(int)
    if nframes:
        coord, box = coord[:nframes], box[:nframes]
    nf = coord.shape[0]

    dp = DeepPot(ckpt, no_jit=True)
    model = dp.deep_eval.dp.model["Default"]
    les = model.atomic_model.les_model
    ewald = les.ewald
    trained_weight = model.lr_weight
    trained_flag = ewald.remove_self_interaction

    def ev():
        e, f, _ = dp.eval(coord, box, types, atomic=False)
        return np.asarray(e).reshape(nf), np.asarray(f).reshape(nf, -1, 3)

    # A: as trained          B: long-range removed     C: self term removed
    # B and C are each differenced against A, so both must be taken at the
    # trained lr_weight; only B moves lr_weight, only C moves the kernel flag.
    model.lr_weight = 1.0
    e_tot, f_tot = ev()
    ewald.remove_self_interaction = True
    e_nosi, f_nosi = ev()
    ewald.remove_self_interaction = trained_flag
    model.lr_weight = 0.0
    e_sr, f_sr = ev()
    model.lr_weight = trained_weight
    del dp

    e_lr = e_tot - e_sr           # the model's own long-range energy
    e_self = e_tot - e_nosi       # the term the flag controls
    f_lr = f_tot - f_sr
    rms = lambda a: float(np.sqrt((a ** 2).mean()))
    f_self_rms = rms(f_tot - f_nosi)
    # rsi=False keeps +norm_factor * sum(q^2) / (sigma * (2*pi)^1.5), so the flag
    # flip prices that term and back-solves sum(q^2) from it.
    coef = getattr(ewald, 'norm_factor', 0.0) / (
        getattr(ewald, 'sigma', 1.0) * (2 * np.pi) ** 1.5)
    row = {
        "arm": arm, "step": step, "nframes": nf,
        "e_sr_mean": float(e_sr.mean()), "e_lr_mean": float(e_lr.mean()),
        "e_lr_std": float(e_lr.std()),
        "f_tot_rms": rms(f_tot), "f_sr_rms": rms(f_sr), "f_lr_rms": rms(f_lr),
        "f_lr_over_f_tot": rms(f_lr) / rms(f_tot),
        "e_self_mean": float(e_self.mean()),
        "f_self_rms": f_self_rms,
        "f_self_over_f_tot": f_self_rms / rms(f_tot),
        "self_over_e_lr": float(e_self.mean() / e_lr.mean()) if e_lr.mean() else None,
        "sum_q2_mean": float(e_self.mean() / coef) if coef else None,
        "rsi_trained": int(bool(trained_flag)),
    }
    def fmt(v, spec):
        return "n/a" if v is None else format(v, spec)

    print(f"{arm:34s} step={step} n={nf}  E_SR {row['e_sr_mean']:+11.4f}  "
          f"E_LR {row['e_lr_mean']:+8.4f} +- {row['e_lr_std']:.4f}  "
          f"E_self {row['e_self_mean']:+8.4f} = "
          f"{fmt(row['self_over_e_lr'], '5.2f')}x E_LR  "
          f"|F| tot {row['f_tot_rms']:.4f} LR {row['f_lr_rms']:.4f}  "
          f"LR share {row['f_lr_over_f_tot']:.4f}  "
          f"self-force share {row['f_self_over_f_tot']:.4f}  "
          f"sum q^2 {fmt(row['sum_q2_mean'], '.2f')}  (rsi={row['rsi_trained']})",
          flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="*", default=None)
    ap.add_argument("--runs-root", default=DEFAULT_RUNS)
    ap.add_argument("--step", type=int, default=80000)
    ap.add_argument("--nframes", type=int, default=0,
                    help="0 = all frames of the validation system (default)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    arms = args.arms or ["deepmd-les_sA", "deepmd-les_sB",
                         "deepmd-les-claim-neutral_sA",
                         "deepmd-les-claim-neutral_sB",
                         "deepmd-les-freeze-charge_sA",
                         "deepmd-les-freeze-charge_sB"]
    rows = []
    for arm in arms:
        r = measure(os.path.join(args.runs_root, arm), args.step, args.nframes)
        if r:
            rows.append(r)
    if args.out and rows:
        def cell(v):
            if isinstance(v, str):
                return v
            if v is None:
                return ''
            if isinstance(v, bool):
                return str(int(v))
            if isinstance(v, int):
                return str(v)
            return f"{v:.8e}"

        with open(args.out, "w") as fh:
            fh.write("\t".join(COLS) + "\n")
            for r in rows:
                fh.write("\t".join(cell(r[c]) for c in COLS) + "\n")
        print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
