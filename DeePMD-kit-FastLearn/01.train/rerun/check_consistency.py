import json, sys, copy
import numpy as np
import torch
import yaml

from deepmd.utils.argcheck import normalize
from deepmd.pt.model.model import get_model

cfgs = {
    "hybrid_q": open("input_hybrid_q.yaml").read(),
    "hybrid_fixed": open("input_hybrid_fixed.yaml").read(),
}
torch.manual_seed(0)
np.random.seed(0)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build(yaml_text):
    data = yaml.safe_load(yaml_text)
    full = normalize(data)
    m = get_model(full["model"])
    m.to(DEV).double()
    m.eval()
    return m

def make_system(nH2O=2, L=9.0):
    # O then H,H per molecule, matching type_map ["O","H"]
    coord = np.array([
        [0.0, 0.0, 0.0],
        [0.96, 0.0, 0.0],
        [-0.24, 0.93, 0.0],
    ] * nH2O, dtype=np.float64)
    coord += np.random.randn(*coord.shape) * 0.05
    coord += np.array([L/2, L/2, L/2])
    atype = np.array([0, 1, 1] * nH2O, dtype=np.int64)
    box = np.diag([L, L, L]).reshape(1, 9)
    return (torch.tensor(coord).reshape(1, -1, 3).to(DEV),
            torch.tensor(atype).reshape(1, -1).to(DEV),
            torch.tensor(box).to(DEV))

for name, txt in cfgs.items():
    try:
        model = build(txt)
    except Exception as e:
        print(f"[{name}] build failed: {type(e).__name__}: {e}")
        continue
    coord, atype, box = make_system()
    nloc = coord.shape[1]
    out = model(coord.clone(), atype, box)
    E = out["energy"].sum().item()
    F = out["force"].detach().clone()  # [1,nloc,3]

    # central finite difference
    eps = 1e-5
    Fd = torch.zeros_like(F)
    for i in range(nloc):
        for d in range(3):
            cp = coord.clone(); cp[0, i, d] += eps
            cm = coord.clone(); cm[0, i, d] -= eps
            Ep = model(cp, atype, box)["energy"].sum().item()
            Em = model(cm, atype, box)["energy"].sum().item()
            Fd[0, i, d] = -(Ep - Em) / (2 * eps)
    diff = (F - Fd).abs()
    rel = diff.max().item() / (Fd.abs().max().item() + 1e-12)
    print(f"[{name}] max|F_autograd - F_fd| = {diff.max().item():.3e} eV/A ; "
          f"max|F_fd| = {Fd.abs().max().item():.3e} ; rel = {rel:.3e}")
    print(f"    E={E:.6f} eV, F_autograd[0,0]={F[0,0].tolist()}, F_fd[0,0]={Fd[0,0].tolist()}")
