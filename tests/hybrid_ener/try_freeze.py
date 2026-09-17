"""Smoke test for the freeze path: script -> save -> load.

Mirrors the three steps `dp freeze` performs (torch.jit.script the model, then
torch.jit.save, then reload) without invoking the CLI. Use check_frozen.py if you
also want the eager-vs-scripted numerics compared.

Usage: python try_freeze.py [ckpt]
"""
import os
import sys
import tempfile

import torch

from _common import build_model, load_checkpoint


def main():
    if len(sys.argv) > 1:
        model = load_checkpoint(sys.argv[1], device="cpu")
    else:
        model = build_model(device="cpu")
    model.eval()

    try:
        scripted = torch.jit.script(model)
    except Exception as e:
        print("RESULT: FAIL - torch.jit.script raised", type(e).__name__)
        print(e)
        return 1
    print("SCRIPT OK ->", type(scripted).__name__)

    out = os.path.join(tempfile.mkdtemp(prefix="hybrid_freeze_"), "frozen.pth")
    try:
        torch.jit.save(scripted, out)
    except Exception as e:
        print("RESULT: FAIL - torch.jit.save raised", type(e).__name__)
        print(e)
        return 1
    print("SAVE OK ->", out)

    try:
        loaded = torch.jit.load(out)
    except Exception as e:
        print("RESULT: FAIL - torch.jit.load raised", type(e).__name__)
        print(e)
        return 1
    print("LOAD OK ->", type(loaded).__name__)

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
