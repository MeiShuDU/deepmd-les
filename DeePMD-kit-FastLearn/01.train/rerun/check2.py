import numpy as np
import torch
import yaml

from deepmd.utils.argcheck import normalize
from deepmd.pt.model.model import get_model

torch.manual_seed(0); np.random.seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build(path):
    data = yaml.safe_load(open(path).read())
    m = get_model(normalize(data)["model"]); m.to(DEV).double(); m.eval()
    return m

def make_system(nH2O=2, L=30.0):
    coord = np.array([[0.,0.,0.],[0.96,0.,0.],[-0.24,0.93,0.]]*nH2O)
    coord += np.random.randn(*coord.shape)*0.05 + np.array([L/2,L/2,L/2])
    atype = np.array([0,1,1]*nH2O, dtype=np.int64)
    box = np.diag([L,L,L]).reshape(1,9)
    return (torch.tensor(coord).reshape(1,-1,3).to(DEV),
            torch.tensor(atype).reshape(1,-1).to(DEV),
            torch.tensor(box).to(DEV))

def fd_check(name, model, coord, atype, box, eps=1e-5):
    out = model(coord.clone(), atype, box)
    E = out["energy"].sum().item(); F = out["force"].detach().clone()
    nloc = coord.shape[1]; Fd = torch.zeros_like(F)
    for i in range(nloc):
        for d in range(3):
            cp = coord.clone(); cp[0,i,d]+=eps
            cm = coord.clone(); cm[0,i,d]-=eps
            Fd[0,i,d] = -(model(cp,atype,box)["energy"].sum().item()
                          - model(cm,atype,box)["energy"].sum().item())/(2*eps)
    diff=(F-Fd).abs()
    print(f"[{name}] max|F_ag-F_fd|={diff.max().item():.3e}  max|F_fd|={Fd.abs().max().item():.3e}  "
          f"rel={diff.max().item()/(Fd.abs().max().item()+1e-12):.3e}  E={E:.6f}")

coord, atype, box = make_system()
for name,path in [("ordinary","input_ordinary.yaml"),
                  ("hybrid_q","input_hybrid_q.yaml"),
                  ("hybrid_fixed","input_hybrid_fixed.yaml")]:
    try:
        m = build(path)
        fd_check(name, m, coord, atype, box)
    except Exception as e:
        print(f"[{name}] ERROR {type(e).__name__}: {e}")
