import numpy as np, torch, yaml
from deepmd.utils.argcheck import normalize
from deepmd.pt.model.model import get_model
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

torch.manual_seed(0); np.random.seed(0)
DEV = torch.device("cuda")

def build(path):
    m = get_model(normalize(yaml.safe_load(open(path).read()))["model"])
    m.to(DEV).double(); m.eval(); return m

def sys_(nH2O=2, L=30.0):
    coord = np.array([[0.,0.,0.],[0.96,0.,0.],[-0.24,0.93,0.]]*nH2O)
    coord += np.random.randn(*coord.shape)*0.05 + np.array([L/2,L/2,L/2])
    atype = np.array([0,1,1]*nH2O, dtype=np.int64)
    return (torch.tensor(coord).reshape(1,-1,3).to(DEV),
            torch.tensor(atype).reshape(1,-1).to(DEV),
            torch.tensor(np.diag([L,L,L]).reshape(1,9)).to(DEV))

def E_lr_of(model, coord, atype, box):
    rcut, sel = model.get_rcut(), model.get_sel()
    ec, ea, mp, nl = extend_input_and_build_neighbor_list(
        coord, atype, rcut, sel, box=box, mixed_types=model.mixed_types())
    desc = model.atomic_model.descriptor(ec, ea, nl)[0]
    nloc = coord.shape[1]
    out = model.atomic_model.les_model(
        positions=coord[0], cell=box[0].reshape(3,3).unsqueeze(0),
        desc=desc[0], batch=None, compute_energy=True,
        atomic_types=[model.atomic_model.type_map[a] for a in atype[0]])
    return out["E_lr"].sum()

m = build("input_hybrid_q.yaml")
coord0, atype, box = sys_()
coord = coord0.clone().requires_grad_(True)
E = E_lr_of(m, coord, atype, box)
F_ag = -torch.autograd.grad(E, coord)[0]     # [1,nloc,3]

eps = 1e-6; nloc = coord.shape[1]
F_fd = torch.zeros_like(F_ag)
for i in range(nloc):
    for d in range(3):
        cp = coord0.clone(); cp[0,i,d]+=eps
        cm = coord0.clone(); cm[0,i,d]-=eps
        F_fd[0,i,d] = -(E_lr_of(m,cp,atype,box).item()-E_lr_of(m,cm,atype,box).item())/(2*eps)
diff=(F_ag-F_fd).abs()
print(f"E_LR only: max|F_ag-F_fd|={diff.max().item():.3e}  max|F_fd|={F_fd.abs().max().item():.3e}  rel={diff.max().item()/(F_fd.abs().max().item()+1e-12):.3e}")
print("F_ag[0,0]=",F_ag[0,0].tolist()," F_fd[0,0]=",F_fd[0,0].tolist())

# Also: repeat but detach desc, to see the descriptor-response contribution size
def E_lr_nod(model, coord, atype, box):
    rcut, sel = model.get_rcut(), model.get_sel()
    ec, ea, mp, nl = extend_input_and_build_neighbor_list(
        coord, atype, rcut, sel, box=box, mixed_types=model.mixed_types())
    desc = model.atomic_model.descriptor(ec, ea, nl)[0].detach()
    return model.atomic_model.les_model(
        positions=coord[0], cell=box[0].reshape(3,3).unsqueeze(0),
        desc=desc[0], batch=None, compute_energy=True,
        atomic_types=[model.atomic_model.type_map[a] for a in atype[0]])["E_lr"].sum()
E2 = E_lr_nod(m, coord, atype, box)
F_ag2 = -torch.autograd.grad(E2, coord)[0]
print("descriptor-response contribution max:", (F_ag-F_ag2).abs().max().item())
