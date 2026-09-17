"""serialize()/deserialize() round-trip: the restored model must be equivalent.

Builds a hybrid_ener model, calls serialize() on it, rebuilds the model from that
document alone with BaseModel.deserialize(), and compares the outputs and the LES
weights. This is the path used for model export / convert-back, so the serialized
document must carry the LES weights and les_params, not just the deepmd side.

Usage: python check_serialize.py [ckpt]
"""
import copy
import sys

import torch

from _common import build_model, build_water, load_checkpoint
from deepmd.pt.model.model.model import BaseModel


def predict(model, coord, atype, box):
    model.eval()
    ret = model(coord.clone(), atype.clone(), box.clone(), do_atomic_virial=True)
    return {k: v.detach().clone() for k, v in ret.items() if v is not None}


def main():
    coord, atype, box = build_water()
    model = load_checkpoint(sys.argv[1]) if len(sys.argv) > 1 else build_model()
    ref = predict(model, coord, atype, box)

    serialized = model.serialize()
    print("serialize() keys     :", sorted(serialized.keys()))
    print("  type               :", serialized.get("type"))
    print("  les_params         :", serialized.get("les_params"))
    les_vars = serialized.get("@variables", {}).get("les_model", {})
    print("  les_model tensors  :", len(les_vars))

    if len(les_vars) == 0:
        print("RESULT: FAIL - serialized document carries no LES weights")
        return 1
    try:
        model2 = BaseModel.deserialize(copy.deepcopy(serialized))
    except Exception as e:
        print(f"RESULT: FAIL - deserialize raised {type(e).__name__}: {e}")
        return 1
    print("deserialize() class  :", type(model2).__name__)

    if type(model2).__name__ != "HybridLESModel":
        print("RESULT: FAIL - wrong class after round-trip")
        return 1
    if not hasattr(model2.atomic_model, "les_model"):
        print("RESULT: FAIL - restored model has no LES submodule")
        return 1
    if hasattr(model2, "to"):
        model2.to(coord.device)

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
        good = d <= 1e-12 * max(1.0, scale)
        print(f"  {key:12s} {'ok ' if good else 'BAD'} max|diff|={d:.3e} max|ref|={scale:.3e}")
        ok = ok and good

    w1 = model.atomic_model.les_model.state_dict()
    w2 = model2.atomic_model.les_model.state_dict()
    wmax = max((w1[k] - w2[k]).abs().max().item() for k in w1)
    print(f"  LES weight max|diff| = {wmax:.3e}")
    ok = ok and wmax == 0.0

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
