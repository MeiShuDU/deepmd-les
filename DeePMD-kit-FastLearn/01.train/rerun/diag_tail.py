"""Reconcile the logged validation curve with the exact full-set values.

The `rmse_e_val` column in lcurve_*.out is an RMSE over 3 randomly drawn frames
(validation_data batch_size 1, numb_btch 3), and each point comes from a
*different* model state. A saved checkpoint, evaluated over all 80 frames,
gives the exact value for one state. Comparing a window of logged points
against one checkpoint's exact value is therefore invalid, and it looks like a
contradiction: for run_hybrid_fixed_sB the logged values over steps 8000-10000
have median 7.08e-4 while the step-10000 exact value is 1.5468e-3.

This script shows there is no contradiction. For each checkpoint it reports the
exact full-set value, the sampling distribution a 3-frame draw would have, and
the value actually logged at that step. Each logged value is an ordinary draw
from its own checkpoint's distribution; only the lumped window misleads.

Usage: python diag_tail.py <run_dir_name> [step ...]
"""
import os
import sys

import numpy as np
import pandas as pd

from deepmd.common import j_loader
from deepmd.infer.deep_pot import DeepPot
from deepmd.utils.data import DeepmdData

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "extended")
LC_COLS = ["step", "rmse_val", "rmse_trn", "rmse_e_val", "rmse_e_trn",
           "rmse_f_val", "rmse_f_trn", "lr"]


def main(name, steps):
    rundir = os.path.join(EXT, name)
    jdata = j_loader(os.path.join(rundir, "input.yaml"))
    systems = jdata["training"]["validation_data"]["systems"]
    if isinstance(systems, str):
        systems = [systems]
    tag = name[len("run_"):]
    logged = pd.read_csv(os.path.join(rundir, f"lcurve_{tag}.out"),
                         sep=r"\s+", skiprows=2, names=LC_COLS).set_index("step")

    rng = np.random.default_rng(0)
    print(f"{name}: the logged curve against each checkpoint's own distribution")
    print(f"{'step':>6} {'exact':>11} {'3frm med':>11} {'3frm p5':>11} "
          f"{'3frm p95':>11} {'med/exact':>10} {'logged':>11}")
    for step in steps:
        dp = DeepPot(os.path.join(rundir, f"model.ckpt-{step}.pt"), no_jit=True)
        tmap = dp.get_type_map()
        data = DeepmdData(systems[0], set_prefix="set", shuffle_test=False,
                          type_map=tmap, sort_atoms=False)
        data.add("energy", 1, atomic=False, must=False, high_prec=True)
        td = data.get_test()
        natoms = len(td["type"][0])
        nframes = td["box"].shape[0]
        e, _ = dp.eval(td["coord"].reshape(nframes, -1), td["box"], td["type"][0])[:2]
        per_frame = np.asarray(e).reshape(nframes) - np.asarray(td["energy"]).reshape(nframes)
        per_frame = per_frame / natoms
        exact = np.sqrt((per_frame ** 2).mean())
        # a 3-frame draw averages 3 sampled per-frame squared errors
        draws = rng.integers(0, nframes, size=(20000, 3))
        three = np.sqrt((per_frame[draws] ** 2).mean(axis=1))
        lg = logged.loc[step, "rmse_e_val"] if step in logged.index else np.nan
        print(f"{step:>6} {exact:>11.4e} {np.median(three):>11.4e} "
              f"{np.percentile(three, 5):>11.4e} {np.percentile(three, 95):>11.4e} "
              f"{np.median(three) / exact:>10.3f} {lg:>11.4e}")


if __name__ == "__main__":
    main(sys.argv[1], [int(s) for s in sys.argv[2:]] or [8000, 9000, 10000])
