"""E2E check of the long-range virial against a finite-difference strain derivative.

DeepMD's convention is V = -dE/deps, where eps strains the cell at fixed
fractional coordinates:  h -> h (I + eps),  r -> r (I + eps).

For the hybrid, E = E_SR + E_LR. E_SR depends on the cell only through the
coordinates (already handled by DeepMD's atomic virial); E_LR additionally
depends on the cell explicitly (volume, reciprocal lattice), so its virial needs
the extra term -(dE_LR/dh) h^T on top of the atomic term.

This script verifies:
  1. the reported virial equals -dE/deps to FD accuracy, for all 9 components;
  2. dropping the LES terms (the pre-fix behaviour = short-range virial only)
     does NOT;
  3. atom_virial sums to virial over atoms.

All 9 components matter: eps decomposes into a symmetric strain plus an
infinitesimal rigid rotation, and the rotation part leaves E invariant. A test
that only probes symmetric eps (as an earlier version of this script did) can
therefore never detect a transposed index pairing in the virial.

The cell is sheared off-diagonal by default so that the test is not degenerate:
with a cubic cell (box = a*I) the left/right strain conventions coincide and a
transposed virial still matches the FD.

Usage: python check_virial.py [ckpt] [system] [frame] [--cubic]
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
EPS = 1e-4
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


def energy_and_virial(eps):
    """Total energy and the model's virial at strain eps (fractional coords fixed)."""
    box = box0 @ (I3 + eps)
    coord = (frac @ box).reshape(1, nloc, 3)
    ret = model(coord, atype, box.reshape(1, 9).clone(), do_atomic_virial=True)
    return (
        ret["energy"].sum(),
        ret["virial"].detach().reshape(3, 3),
        ret["atom_virial"].detach(),
        ret["force"].detach(),
    )


# reference: model prediction at zero strain
e0, virial, atom_virial, force = energy_and_virial(torch.zeros(3, 3))

# short-range-only virial (the pre-fix value) for comparison
coord_z = (frac @ box0).reshape(1, nloc, 3)
sr_virial = (
    model.forward_common(coord_z, atype, box0.reshape(1, 9).clone())["energy_derv_c_redu"]
    .detach()
    .squeeze(-2)
    .reshape(3, 3)
)

# finite-difference -dE/deps over ALL 9 components (non-symmetric eps)
fd = torch.zeros(3, 3, dtype=torch.float64)
for a in range(3):
    for b in range(3):
        ep = torch.zeros(3, 3, dtype=torch.float64)
        ep[a, b] = EPS
        em = torch.zeros(3, 3, dtype=torch.float64)
        em[a, b] = -EPS
        d = -(energy_and_virial(ep)[0] - energy_and_virial(em)[0]).item() / (2 * EPS)
        fd[a, b] = d

scale = fd.abs().max().item()
err_tot = (virial - fd).abs().max().item()
err_sr = (sr_virial - fd).abs().max().item()

print(f"ckpt={CKPT} frame={FRAME} nloc={nloc} "
      f"cell={'cubic' if '--cubic' in sys.argv else 'sheared'}")
print(f"max|FD virial|        = {scale:.6e}")
print(f"max|virial - FD|      = {err_tot:.6e}   (full: SR + LR)")
print(f"max|sr_virial - FD|   = {err_sr:.6e}   (pre-fix: SR only)")
print(f"LR contribution       = {(virial - sr_virial).abs().max().item():.6e}")
print(f"antisym(virial)       = {(virial - virial.T).abs().max().item():.3e}")
asum = atom_virial.sum(dim=1).reshape(3, 3)
print(f"max|sum(atom_virial) - virial| = {(asum - virial).abs().max().item():.3e}")

ok = err_tot < 1e-5 * max(1.0, scale) and err_tot < 0.01 * err_sr
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
