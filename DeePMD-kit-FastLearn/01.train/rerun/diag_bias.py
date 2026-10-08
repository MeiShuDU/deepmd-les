"""Dump the signed per-frame per-atom energy error for the late checkpoints.

Why
---
The exact `rmse_e` the document reports is the L2 norm of the per-frame energy
error over a whole split:

    rmse_e = sqrt( mean_i ( (E_pred,i - E_label,i) / natoms )^2 )

Squared, that is mean(e^2) = mean(e)^2 + var(e), i.e.

    rmse_e^2 = bias^2 + spread^2

where bias = mean(e) is a systematic offset shared by every frame and
spread = std(e) is how scattered the per-frame errors are. The two move for
completely different reasons, so a plot is worth more than the scalar.

This script writes the raw per-frame vector for a whole split, one (run, step)
per process (loading several models into one CUDA context trips an allocator
assert on this host).

The split argument exists to separate a wandering model energy zero from a
fixed train/valid mean-energy mismatch: if the bias is the model's energy zero,
the train and valid per-frame vectors differ only by a constant (the split
mean-energy difference), so their biases move together. Run it for both splits
and compare.

Usage: python diag_bias.py <split: train|valid> <run_dir_name> [step ...]
Writes/appends: bias_frames_<split>.tsv
"""
import os
import sys

import numpy as np

from deepmd.common import j_loader
from deepmd.infer.deep_pot import DeepPot
from deepmd.utils.data import DeepmdData

from runname import DEFAULT_DECAY, parse_run

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "extended")
KEY = {"train": "training_data", "valid": "validation_data"}


def main(split, name, steps):
    rundir = os.path.join(EXT, name)
    jdata = j_loader(os.path.join(rundir, "input.yaml"))
    systems = jdata["training"][KEY[split]]["systems"]
    if isinstance(systems, str):
        systems = [systems]
    meta = parse_run(name)
    model, rep = meta["model"], meta["rep"]
    suffix = "" if meta["decay"] == DEFAULT_DECAY else f"_d{meta['decay']}"
    out = os.path.join(HERE, f"bias_frames_{split}{suffix}.tsv")

    append = os.path.exists(out)
    lines = []
    print(f"{name} [{split}]")
    print(f"{'step':>6} {'rmse':>11} {'bias':>11} {'spread':>11} "
          f"{'nframes':>8} {'bias^2/rmse^2':>13}")
    for step in steps:
        dp = DeepPot(os.path.join(rundir, f"model.ckpt-{step}.pt"), no_jit=True)
        tmap = dp.get_type_map()
        err = []
        for system in systems:
            data = DeepmdData(system, set_prefix="set", shuffle_test=False,
                              type_map=tmap, sort_atoms=False)
            data.add("energy", 1, atomic=False, must=False, high_prec=True)
            td = data.get_test()
            natoms = len(td["type"][0])
            nframes = td["box"].shape[0]
            e, = dp.eval(td["coord"].reshape(nframes, -1), td["box"],
                         td["type"][0])[:1]
            err.append((np.asarray(e).reshape(nframes)
                        - np.asarray(td["energy"]).reshape(nframes)) / natoms)
        err = np.concatenate(err)
        bias, spread = err.mean(), err.std()
        rmse = np.sqrt((err ** 2).mean())
        print(f"{step:>6} {rmse:>11.4e} {bias:>+11.4e} {spread:>11.4e} "
              f"{err.size:>8} {bias * bias / (rmse * rmse):>13.3f}")
        for i, d in enumerate(err):
            lines.append(f"{name}\t{model}\t{rep}\t{step}\t{i}\t{d:.8e}")

    with open(out, "a" if append else "w") as f:
        if not append:
            f.write("run\tmodel\trep\tstep\tframe\tperatom_de\n")
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]] or [8000, 9000, 10000])
