"""One forward on one trained checkpoint, printed at full precision.

Run this on the pod (pod's les/deepmd tree) and locally (local tree) on the SAME
checkpoint and the SAME frames, then compare the numbers. The two `les` revisions
differ only in default-off features by inspection; this makes it a measurement.

CPU on both sides, so the comparison is not confounded by GPU/CPU library paths.
Set DP_INTERFACE_PREC=low before running: these checkpoints were trained in fp32.
"""
import os
import sys

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else (
    "/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_water_interface/"
    "deepmd/runs/deepmd-les-cace-sched_sA/s8")
VALID = ("/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/"
         "campaign_water_interface/data/water-interface/valid")
NFRAMES = 2

import deepmd.env  # noqa: E402  (after DP_INTERFACE_PREC is set by the shell)
from deepmd.infer.deep_pot import DeepPot  # noqa: E402

ckpt = os.path.join(RUN, "model.ckpt-45000.pt")
coord = np.load(os.path.join(VALID, "set.000", "coord.npy"))[:NFRAMES]
box = np.load(os.path.join(VALID, "set.000", "box.npy"))[:NFRAMES]
types = np.loadtxt(os.path.join(VALID, "type.raw")).astype(int)

dp = DeepPot(ckpt, no_jit=True)
e, f, v = dp.eval(coord, box, types, atomic=False)
e = np.asarray(e).reshape(-1)
f = np.asarray(f).reshape(NFRAMES, -1, 3)
v = np.asarray(v).reshape(NFRAMES, -1)
print("DP_INTERFACE_PREC =", os.environ.get("DP_INTERFACE_PREC"))
print("les module file  =", __import__("les").__file__)
print("E            =", [repr(float(x)) for x in e])
print("F[0,0]       =", [repr(float(x)) for x in f[0, 0]])
print("F[1,-1]      =", [repr(float(x)) for x in f[1, -1]])
print("|F|          =", repr(float(np.linalg.norm(f))))
print("V sum        =", repr(float(v.sum())))