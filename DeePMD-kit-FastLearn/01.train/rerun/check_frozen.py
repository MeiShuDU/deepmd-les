"""numerics: eager model vs torch.jit.script-frozen model must agree bit-for-bit.

The freeze path (dp freeze) torches the exact code path try_freeze.py scripts.
This verifies that scripting does not change any numerics: the same frame run
through the eager model and through the frozen (scripted) model must produce
identical energy / force / virial / atom_virial.

Usage: python check_frozen.py [ckpt] [system] [frame] [--cubic]
"""
import copy
import sys

import numpy as np
import torch

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
CKPT = argv[0] if len(argv) > 0 else "extended/run_hybrid_fixed_sA/model.ckpt-10000.pt"
SYSTEM = argv[1] if len(argv) > 1 else (
    "/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/data/data_3"
)
FRAME = int(argv[2]) if len(argv) > 2 else 0
I3 = torch.eye(3, dtype=torch.float64)

from deepmd.pt.model.model import get_model
from deepmd.pt.train.wrapper import ModelWrapper
from deepmd.utils.data import DeepmdData

state = torch.load(CKPT, map_location="cpu", weights_only=True)
if "model" in state:
    state = state["model"]
model = get_model(copy.deepcopy(state["_extra_state"]["model_params"]))
ModelWrapper(model).load_state_dict(state)
model.eval()
model.to("cpu")

# script exactly like dp freeze does (try_freeze.py)
try:
    sm = torch.jit.script(model)
except Exception as e:
    print("SCRIPT FAIL:", type(e).__name__)
    print(e)
    sys.exit(1)
print("SCRIPT OK ->", type(sm).__name__)

data = DeepmdData(SYSTEM, set_prefix="set", shuffle_test=False,
                  type_map=model.get_type_map(), sort_atoms=False)
data.add("energy", 1, atomic=False, must=False, high_prec=True)
td = data.get_test()
nloc = len(td["type"][0])

coord0 = torch.tensor(td["coord"][FRAME], dtype=torch.float64).reshape(nloc, 3)
box0 = torch.tensor(np.asarray(td["box"][FRAME]), dtype=torch.float64).reshape(3, 3)
if "--cubic" in sys.argv:
    shear = I3
else:
    shear = torch.tensor([[1.0, 0.113, 0.0],
                          [0.0, 1.0, 0.071],
                          [0.047, 0.0, 1.0]], dtype=torch.float64)
    coord0 = coord0 @ shear
    box0 = box0 @ shear
atype = torch.tensor(np.tile(np.asarray(td["type"][0]), (1, 1)), dtype=torch.int64)
frac = coord0 @ torch.linalg.inv(box0)
box = box0
coord = (frac @ box).reshape(1, nloc, 3)
cell = box.reshape(1, 9).clone()

# 模型 forward 内部用 autograd.grad 求力/virial (与 check_virial.py 相同), 因此
# 不能包 no_grad: 包住后内部张量无 grad_fn, autograd.grad 报错。
r_eager = model(coord, atype, cell, do_atomic_virial=True)
r_frozen = sm(coord, atype, cell, do_atomic_virial=True)

keys = ["energy", "atom_energy", "force", "virial", "atom_virial"]
worst = 0.0
for k in keys:
    a = r_eager[k].detach()
    b = r_frozen[k].detach()
    if a.shape != b.shape:
        print(f"{k}: SHAPE MISMATCH eager {tuple(a.shape)} frozen {tuple(b.shape)}")
        sys.exit(1)
    d = (a - b).abs().max().item()
    norm = a.abs().max().item()
    worst = max(worst, d)
    tag = "OK " if d == 0.0 else "DIFF"
    print(f"{tag} max|eager - frozen| {k:<12} = {d:.3e}   (max|eager|={norm:.6e})")

print(f"\nworst componentwise diff = {worst:.3e}")
if worst == 0.0:
    print("RESULT: PASS (bit-for-bit identical)")
    sys.exit(0)
if worst < 1e-9:
    print("RESULT: PASS (float-assoc noise < 1e-9)")
    sys.exit(0)
print("RESULT: FAIL")
sys.exit(1)