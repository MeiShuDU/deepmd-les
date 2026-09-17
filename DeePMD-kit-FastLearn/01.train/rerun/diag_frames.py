"""Diagnose the full-set evaluation: batched-vs-single-frame equivalence and
the per-frame distribution of energy error.

Two questions this answers:
  1. Sanity: does evaluating all 80 frames in one DeepPot.eval call give the same
     per-frame energy as evaluating each frame alone? (Guards against a
     cross-frame leak in the LES/Ewald path.)
  2. Why the windowed lcurve mean sits ~35% below the full-set RMSE: are per-frame
     energy errors heavy-tailed, so that a 3-frame draw usually misses the bad
     frames and the mean of many small RMSEs underestimates the pooled RMSE?

Usage: python diag_frames.py <run_dir_name>
"""
import os
import sys

import numpy as np

from deepmd.common import j_loader
from deepmd.infer.deep_pot import DeepPot
from deepmd.utils.data import DeepmdData

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "extended")
CKPT = "model.ckpt-10000.pt"


def main(name):
    rundir = os.path.join(EXT, name)
    jdata = j_loader(os.path.join(rundir, "input.yaml"))
    systems = jdata["training"]["validation_data"]["systems"]
    if isinstance(systems, str):
        systems = [systems]

    dp = DeepPot(os.path.join(rundir, CKPT), no_jit=True)
    tmap = dp.get_type_map()

    data = DeepmdData(
        systems[0], set_prefix="set", shuffle_test=False, type_map=tmap, sort_atoms=False
    )
    data.add("energy", 1, atomic=False, must=False, high_prec=True)
    data.add("force", 3, atomic=True, must=False, high_prec=False)
    td = data.get_test()
    natoms = len(td["type"][0])
    nframes = td["box"].shape[0]

    coord = td["coord"].reshape(nframes, -1)
    box = td["box"]
    atype = td["type"][0]

    # ---- batched ----
    e_batch, f_batch = dp.eval(coord, box, atype)[:2]
    e_batch = np.asarray(e_batch).reshape(nframes)
    f_batch = np.asarray(f_batch).reshape(nframes, -1)

    # ---- one frame at a time ----
    e_one = np.zeros(nframes)
    f_one = np.zeros_like(f_batch)
    for i in range(nframes):
        ei, fi = dp.eval(coord[i : i + 1], box[i : i + 1], atype)[:2]
        e_one[i] = np.asarray(ei).reshape(-1)[0]
        f_one[i] = np.asarray(fi).reshape(-1)

    de = np.abs(e_batch - e_one)
    df = np.abs(f_batch - f_one)
    print(f"batched vs per-frame: max|dE|={de.max():.3e} eV  max|dF|={df.max():.3e} eV/A")

    # ---- per-frame error stats ----
    e_ref = np.asarray(td["energy"]).reshape(nframes)
    f_ref = np.asarray(td["force"]).reshape(nframes, -1)
    per_frame_e = (e_batch - e_ref) / natoms  # per-atom energy error
    per_frame_f_rmse = np.sqrt((f_batch - f_ref) ** 2).mean(axis=1)

    order = np.argsort(-np.abs(per_frame_e))
    print(f"\nper-atom energy error: rmse={np.sqrt((per_frame_e**2).mean()):.3e}  "
          f"max|.|={np.abs(per_frame_e).max():.3e}")
    print("worst 8 frames (idx, per-atom dE):")
    for i in order[:8]:
        print(f"  {i:3d}  {per_frame_e[i]:+.3e}")

    # drop the worst 4 frames and recompute, to show tail sensitivity
    keep = order[4:]
    print(f"rmse over all {nframes} frames      : {np.sqrt((per_frame_e**2).mean()):.3e}")
    print(f"rmse over best {len(keep)} frames     : "
          f"{np.sqrt((per_frame_e[keep]**2).mean()):.3e}")

    # ---- can 3-frame subsampling reproduce the lcurve tail values? ----
    # Training logs the RMSE over only `numb_btch`=3 random frames per display
    # step. Simulate that: many 3-frame draws, then compare the whole simulated
    # distribution against the observed lcurve tail.
    lc = os.path.join(rundir, f"lcurve_{name[len('run_'):]}.out")
    tail = np.loadtxt(lc, skiprows=2, usecols=(0, 3))
    per_step = tail[tail[:, 0] >= 8000][:, 1]

    rng = np.random.default_rng(0)
    draws = rng.integers(0, nframes, size=(20000, 3))
    three = np.sqrt((per_frame_e[draws] ** 2).mean(axis=1))
    pooled = np.sqrt((per_frame_e**2).mean())
    print(f"\nE[rmse over 3 random frames] = {three.mean():.3e} "
          f"(pooled over 80 = {pooled:.3e})")
    print(f"  -> a 3-frame draw averages {100 * three.mean() / pooled:.0f}% "
          f"of the pooled RMSE")
    pct = np.percentile(three, [1, 5, 25, 50, 75, 95, 99])
    print("  simulated 3-frame RMSE percentiles (1,5,25,50,75,95,99):")
    print("   " + "  ".join(f"{p:.2e}" for p in pct))
    print(f"  observed lcurve tail (8000-10000): "
          f"min={np.min(per_step):.2e} median={np.median(per_step):.2e} "
          f"max={np.max(per_step):.2e}")
    # If the logged value were effectively a SINGLE frame (not 3), its median
    # would track median|per-frame error| rather than the pooled RMSE.
    print(f"\n|per-frame per-atom dE|: median={np.median(np.abs(per_frame_e)):.3e} "
          f"mean={np.abs(per_frame_e).mean():.3e} rms={pooled:.3e}")
    print(f"  ratio pooled/median = {pooled / np.median(np.abs(per_frame_e)):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
