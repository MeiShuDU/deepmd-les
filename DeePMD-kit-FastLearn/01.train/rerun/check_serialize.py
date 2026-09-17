"""E2E check: a hybrid_ener checkpoint must survive serialize()/deserialize().

Path under test (dp convert-back style, no state_dict in between):

    model = get_model(model_params); load_state_dict(ckpt)
    data = model.serialize()
    model2 = BaseModel.deserialize(data)
    -> model2 must be a HybridLESModel whose energy/force/virial match model

Run on CPU (one hybrid model per CUDA context is not supported on this host).

Usage: python check_serialize.py [ckpt] [system]
"""
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "extended")
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    EXT, "run_hybrid_fixed_sA", "model.ckpt-10000.pt"
)
SYSTEM = sys.argv[2] if len(sys.argv) > 2 else (
    "/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/data/data_3"
)
NF = 4

from deepmd.pt.model.model import get_model
from deepmd.pt.model.model.model import BaseModel
from deepmd.pt.train.wrapper import ModelWrapper
from deepmd.utils.data import DeepmdData


def predict(model, coord, atype, box):
    model.eval()
    ret = model(
        coord.clone(),
        atype.clone(),
        box.clone(),
        do_atomic_virial=True,
    )
    return {k: v.detach().clone() for k, v in ret.items() if v is not None}


def main():
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    params = state["_extra_state"]["model_params"]

    model = get_model(copy.deepcopy(params))
    ModelWrapper(model).load_state_dict(state)
    model.to("cpu")  # get_model() builds on the default device; this host allows one CUDA model per process

    data = DeepmdData(SYSTEM, set_prefix="set", shuffle_test=False,
                      type_map=model.get_type_map(), sort_atoms=False)
    data.add("energy", 1, atomic=False, must=False, high_prec=True)
    td = data.get_test()
    coord = torch.tensor(td["coord"][:NF], dtype=torch.float64)
    atype0 = np.asarray(td["type"][0])
    nloc = atype0.size
    atype = torch.tensor(np.tile(atype0, (coord.shape[0], 1)), dtype=torch.int64)
    box = torch.tensor(np.asarray(td["box"][:NF]), dtype=torch.float64)
    coord = coord.reshape(coord.shape[0], nloc, 3)

    ref = predict(model, coord, atype, box)

    # ---- serialize -> deserialize ----
    serialized = model.serialize()
    print("serialize() keys:", sorted(serialized.keys()))
    print("  type                 :", serialized.get("type"))
    print("  les_params           :", serialized.get("les_params"))
    les_vars = serialized.get("@variables", {}).get("les_model", {})
    print("  les_model tensors    :", len(les_vars))

    try:
        model2 = BaseModel.deserialize(copy.deepcopy(serialized))
    except Exception as e:
        print(f"RESULT: FAIL - deserialize raised {type(e).__name__}: {e}")
        return 1
    print("deserialize() class     :", type(model2).__name__)
    if hasattr(model2, "to"):
        model2.to("cpu")
    if type(model2).__name__ != "HybridLESModel":
        print("RESULT: FAIL - wrong class after round-trip")
        return 1
    if not hasattr(model2.atomic_model, "les_model"):
        print("RESULT: FAIL - restored model has no LES submodule")
        return 1

    got = predict(model2, coord, atype, box)

    ok = True
    for key in sorted(ref):
        a, b = ref[key], got[key]
        if a.shape != b.shape:
            print(f"  {key:12s} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
            ok = False
            continue
        d = (a - b).abs().max().item()
        scale = a.abs().max().item() + 1e-12
        flag = "ok " if d <= 1e-12 * max(1.0, scale) else "BAD"
        print(f"  {key:12s} {flag} max|diff|={d:.3e} max|ref|={scale:.3e}")
        ok = ok and flag == "ok "

    # also confirm the LES weights themselves round-tripped (not just the outputs)
    w1 = model.atomic_model.les_model.state_dict()
    w2 = model2.atomic_model.les_model.state_dict()
    wmax = max((w1[k] - w2[k]).abs().max().item() for k in w1)
    print(f"  LES weight max|diff|  = {wmax:.3e}")
    ok = ok and wmax == 0.0

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
