"""Force/energy consistency: the analytic force must be -dE/dr.

Runs the toy H2O system through the model and compares the reported force with a
central finite difference of the reported energy, for both LES baselines (free
latent charges, and free plus fixed baseline charges). A mismatch means the
long-range force is not the gradient of the energy the model reports.

Usage: python check_consistency.py [ckpt]

Without a checkpoint the checks run on a random-weight model (enough to exercise
the autograd path). With one, they run on the trained model instead.
"""
import sys

import torch

from _common import build_model, build_water, load_checkpoint

EPS = 1e-5


def fd_forces(model, coord, atype, box):
    """Central-difference force, -dE/dr, one coordinate at a time."""
    nloc = coord.shape[1]
    fd = torch.zeros(1, nloc, 3, dtype=coord.dtype, device=coord.device)
    for i in range(nloc):
        for d in range(3):
            cp = coord.clone()
            cp[0, i, d] += EPS
            cm = coord.clone()
            cm[0, i, d] -= EPS
            ep = model(cp, atype, box)["energy"].sum().item()
            em = model(cm, atype, box)["energy"].sum().item()
            fd[0, i, d] = -(ep - em) / (2 * EPS)
    return fd


def run(name, model, coord, atype, box):
    out = model(coord.clone(), atype, box)
    force = out["force"].detach()
    energy = out["energy"].sum().item()
    fd = fd_forces(model, coord, atype, box)
    diff = (force - fd).abs().max().item()
    scale = fd.abs().max().item()
    ok = diff < 1e-5 * max(1.0, scale)
    print(
        f"[{name}] max|F_autograd - F_fd| = {diff:.3e} eV/A "
        f"(max|F_fd| = {scale:.3e}, rel = {diff / (scale + 1e-12):.3e}); "
        f"E = {energy:.6f} eV -> {'OK' if ok else 'FAIL'}"
    )
    return ok


def main():
    coord, atype, box = build_water()
    cases = []
    if len(sys.argv) > 1:
        cases.append(("checkpoint", load_checkpoint(sys.argv[1])))
    else:
        cases.append(("free_q", build_model(use_fixed_charges=False)))
        cases.append(("fixed_q", build_model(use_fixed_charges=True)))

    ok = True
    for name, model in cases:
        ok &= run(name, model, coord, atype, box)
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
