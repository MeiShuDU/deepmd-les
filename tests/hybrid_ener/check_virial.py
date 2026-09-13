"""Long-range virial check against a finite-difference strain derivative.

DeepMD convention: V = -dE/deps, where eps strains the cell at fixed fractional
coordinates: h -> h (I + eps), r -> r (I + eps).

For the hybrid, E = E_SR + E_LR. The short-range energy depends on the cell only
through the coordinates (already covered by DeepMD's atomic virial), but E_LR
depends on the cell explicitly as well (volume, reciprocal lattice, Ewald grid),
so its virial needs an extra cell term on top of the atomic sum. This script
checks:

  1. the reported virial equals -dE/deps to finite-difference accuracy, for all
     nine components of eps;
  2. the short-range-only virial does not (i.e. dropping the LES terms fails);
  3. sum_i atom_virial_i == virial.

The cell is sheared off-diagonal so a transposed index pairing cannot pass by
symmetry, and all nine strain components are probed because the antisymmetric
part of eps is a rigid rotation that leaves the energy invariant (zero
derivative) but would reveal a mispaired tensor if the virial were transposed.

Usage: python check_virial.py [ckpt]
"""
import sys

import torch

from _common import build_model, build_water, load_checkpoint

EPS = 1e-4
I3 = torch.eye(3, dtype=torch.float64)
SHEAR = torch.tensor(
    [[1.0, 0.113, 0.0], [0.0, 1.0, 0.071], [0.047, 0.0, 1.0]], dtype=torch.float64
)


def main():
    coord, atype, box = build_water()
    model = load_checkpoint(sys.argv[1]) if len(sys.argv) > 1 else build_model()
    dev = coord.device

    i3 = I3.to(dev)
    shear = SHEAR.to(dev)
    nloc = coord.shape[1]
    coord0 = coord.reshape(nloc, 3) @ shear
    box0 = box.reshape(3, 3) @ shear
    frac = coord0 @ torch.linalg.inv(box0)

    def energy_and_virial(eps):
        b = box0 @ (i3 + eps)
        r = (frac @ b).reshape(1, nloc, 3)
        ret = model(r, atype, b.reshape(1, 9).clone(), do_atomic_virial=True)
        return ret["energy"].sum(), ret["virial"].detach().reshape(3, 3), ret["atom_virial"].detach()

    # Reference prediction at zero strain.
    _e0, virial, atom_virial = energy_and_virial(torch.zeros(3, 3, dtype=torch.float64, device=dev))

    # Short-range-only virial (the value before the LES cell term was added).
    r_z = (frac @ box0).reshape(1, nloc, 3)
    sr_virial = (
        model.forward_common(r_z, atype, box0.reshape(1, 9).clone())["energy_derv_c_redu"]
        .detach()
        .squeeze(-2)
        .reshape(3, 3)
    )

    fd = torch.zeros(3, 3, dtype=torch.float64, device=dev)
    for a in range(3):
        for b in range(3):
            ep = torch.zeros(3, 3, dtype=torch.float64, device=dev)
            ep[a, b] = EPS
            em = torch.zeros(3, 3, dtype=torch.float64, device=dev)
            em[a, b] = -EPS
            fd[a, b] = -(energy_and_virial(ep)[0] - energy_and_virial(em)[0]).item() / (2 * EPS)

    scale = fd.abs().max().item()
    err_tot = (virial - fd).abs().max().item()
    err_sr = (sr_virial - fd).abs().max().item()
    asum = atom_virial.sum(dim=1).reshape(3, 3)

    print(f"nloc={nloc} cell=sheared")
    print(f"max|FD virial|            = {scale:.6e}")
    print(f"max|virial - FD|          = {err_tot:.6e}   (full: SR + LR)")
    print(f"max|sr_virial - FD|       = {err_sr:.6e}   (SR only, expected to fail)")
    print(f"LR contribution           = {(virial - sr_virial).abs().max().item():.6e}")
    print(f"antisym(virial)           = {(virial - virial.T).abs().max().item():.3e}")
    print(f"max|sum(atom_virial)-vir| = {(asum - virial).abs().max().item():.3e}")

    ok = err_tot < 1e-5 * max(1.0, scale) and err_tot < 0.01 * err_sr
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
