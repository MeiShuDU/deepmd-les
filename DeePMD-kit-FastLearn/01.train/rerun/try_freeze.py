"""Try to script the hybrid model, as `dp freeze` does."""
import copy
import sys

import torch

CKPT = sys.argv[1] if len(sys.argv) > 1 else (
    "extended/run_hybrid_fixed_sA/model.ckpt-10000.pt"
)

from deepmd.pt.model.model import get_model
from deepmd.pt.train.wrapper import ModelWrapper

st = torch.load(CKPT, map_location="cpu", weights_only=True)
if "model" in st:
    st = st["model"]
m = get_model(copy.deepcopy(st["_extra_state"]["model_params"]))
ModelWrapper(m).load_state_dict(st)
m.to("cpu")
m.eval()

try:
    sm = torch.jit.script(m)
except Exception as e:
    print("SCRIPT FAIL:", type(e).__name__)
    print(e)
    sys.exit(1)

print("SCRIPT OK ->", type(sm).__name__)
torch.jit.save(sm, "/tmp/frozen_hybrid.pt")
print("SAVE OK")
back = torch.jit.load("/tmp/frozen_hybrid.pt")
print("LOAD OK ->", type(back).__name__)
