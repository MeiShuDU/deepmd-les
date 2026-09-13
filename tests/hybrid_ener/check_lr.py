"""Long-range force check: the LES/Ewald force is -dE_LR/dr.

Isolates the LES contribution from the short-range network: the descriptor is
recomputed, only the LES module is evaluated, and its autograd force is compared
with a central finite difference.

It also measures the descriptor-response part: the difference between the full
autograd force and the force obtained when the descriptor is detached. That term
(dE_LR/dq * dq/ddesc * ddesc/dr) is real and must be present, otherwise the
long-range force is inconsistent with the energy.

Usage: python check_lr.py [ckpt]
"""
import sys

import torch

from _common import build_model, build_water, load_checkpoint
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

EPS = 1e-6


def les_energy(model, coord, atype, box, detach_desc=False):
    """Total long-range energy E_LR for the whole (single-frame) system."""
    rcut, sel = model.get_rcut(), model.get_sel()
    ec, ea, _mapping, nlist = extend_input_and_build_neighbor_list(
        coord, atype, rcut, sel, box=box, mixed_types=model.mixed_types()
    )
    desc = model.atomic_model.descriptor(ec, ea, nlist)[0]
    if detach_desc:
        desc = desc.detach()
    out = model.atomic_model.les_model(
        positions=coord[0],
        cell=box[0].reshape(3, 3).unsqueeze(0),
        desc=desc[0],
        batch=None,
        compute_energy=True,
        atomic_types=[model.atomic_model.type_map[a] for a in atype[0]],
    )
    return out["E_lr"].sum()


def main():
    coord0, atype, box = build_water()
    model = load_checkpoint(sys.argv[1]) if len(sys.argv) > 1 else build_model()

    coord = coord0.clone().requires_grad_(True)
    f_ag = -torch.autograd.grad(les_energy(model, coord, atype, box), coord)[0]

    nloc = coord.shape[1]
    f_fd = torch.zeros_like(f_ag)
    for i in range(nloc):
        for d in range(3):
            cp = coord0.clone()
            cp[0, i, d] += EPS
            cm = coord0.clone()
            cm[0, i, d] -= EPS
            ep = les_energy(model, cp, atype, box).item()
            em = les_energy(model, cm, atype, box).item()
            f_fd[0, i, d] = -(ep - em) / (2 * EPS)

    diff = (f_ag - f_fd).abs().max().item()
    scale = f_fd.abs().max().item()
    ok = diff < 1e-5 * max(1.0, scale)
    print(
        f"max|F_autograd - F_fd| = {diff:.3e}  max|F_fd| = {scale:.3e}  "
        f"rel = {diff / (scale + 1e-12):.3e} -> {'OK' if ok else 'FAIL'}"
    )

    # Descriptor-response term: how much of F_LR vanishes if the descriptor is
    # treated as a constant. It must be non-zero, and it is what the model must
    # keep in the graph to stay energy-force consistent.
    coord2 = coord0.clone().requires_grad_(True)
    f_no_desc_response = -torch.autograd.grad(
        les_energy(model, coord2, atype, box, detach_desc=True), coord2
    )[0]
    resp = (f_ag - f_no_desc_response).abs().max().item()
    print(f"descriptor-response contribution max = {resp:.3e}")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
