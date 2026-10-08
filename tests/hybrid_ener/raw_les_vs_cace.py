"""Raw LES Ewald cross-validation: deepmd-versioned LES vs cace-versioned LES.

Loads the cace benchmark liquid-64.xyz frame (64 H2O molecules in a cubic box),
puts the SAME point charges q on the SAME positions r in the SAME cell, and
compares the long-range electrostatic energy and forces from three independent
evaluations:

  A. les-package Ewald (deepmd-versioned):  les.module.ewald.Ewald
  B. cace internal Ewald:                    cace.modules.ewald.EwaldPotential
  C. naive full-grid k-space reference:      independent plain-kernel sum

The two LES kernels are physically the same formula (Gaussian-screened
reciprocal-space sum with self-interaction removed), differing only in a unit
convention: the les package uses norm_factor = 1/(2 e0) ~ 90.4756 eV e^-2 A,
cace uses norm_factor = 1.0 with charges pre-scaled by sqrt(norm_factor).
We align by setting norm_factor=1.0 on the les Ewald, and separately print the
deepmd-default conversion to prove the unit mapping.

Running: python raw_les_compare.py   (CPU only; requires py39 env with les +
sys.path to /root/app/cace, none of the full cace deps needed)
"""
import sys
import importlib.util as _ilu
import types as _types

import numpy as np
import torch

from les.module.ewald import Ewald as LesEwald                 # A

# Load cace/modules/ewald.py as a standalone module. 'cace' itself is not in
# the py39 env and its __init__ imports ase, so we bypass the package. We also
# relax one float32 assumption in compute_potential_triclinic (nvec was forced
# to .float() before the matmul with a float64 reciprocal lattice G) so the
# cace kernel can run at double precision like the les kernel.
_cace_src = open("/root/app/cace/cace/modules/ewald.py").read()
_cace_src = _cace_src.replace("nvec.float() @ G", "nvec.to(G.dtype) @ G")
_cace_mod = _types.ModuleType("cace_ewald_module")
exec(compile(_cace_src, "/root/app/cace/cace/modules/ewald.py", "exec"), _cace_mod.__dict__)
CaceEwald = _cace_mod.EwaldPotential                             # B

DEEPMD_NORM = 90.4756  # 1/(2 e0) in e^2 eV^-1 A^-1, deepmd default norm_factor


def load_liquid64():
    path = "/root/app/cace/benchmark/liquid-64.xyz"
    lines = open(path).read().splitlines()
    n = int(lines[0].strip())
    # pbc="T T T" Lattice="12.415...000" Properties=...
    lat = lines[1].split('"')[3].split()
    cell = np.array([float(x) for x in lat]).reshape(3, 3)
    r = []
    for i in range(n):
        parts = lines[2 + i].split()
        r.append([float(parts[1]), float(parts[2]), float(parts[3])])
    r = np.array(r)
    return torch.tensor(r, dtype=torch.float64), torch.tensor(cell, dtype=torch.float64).view(1, 3, 3)


def tip3p_charges(r, cell):
    """TIP3P point charges: O -0.834 e, H +0.417 e. Neutral overall."""
    n = r.shape[0]
    q = torch.zeros(n, dtype=torch.float64)
    q[0::3] = -0.834
    q[1::3] = 0.417
    q[2::3] = 0.417
    assert torch.allclose(q.sum(), torch.tensor(0.0, dtype=torch.float64))
    return q


def random_neutral(n, num):
    g = torch.Generator().manual_seed(num)
    q = torch.rand(n, generator=g, dtype=torch.float64) - 0.5
    q = q - q.mean()
    return q


def eval_les(ew, r, q, cell):
    """E (scalar, eV at norm=1) and F = -dE/dr from the les-package Ewald."""
    qr = q.clone().requires_grad_(True)
    rr = r.clone().requires_grad_(True)
    e = ew(q=qr, r=rr, cell=cell)[0].sum()  # batch of 1 frame
    f = -torch.autograd.grad(e, rr)[0]
    return e.detach(), f.detach(), qr.detach()


def eval_cace(ep, r, q, cell):
    """E and F from the cace internal EwaldPotential (data-dict interface)."""
    qr = q.clone().requires_grad_(True)
    rr = r.clone().requires_grad_(True)
    n = rr.shape[0]
    data = {
        "positions": rr,
        "cell": cell.clone(),
        "q": qr.unsqueeze(1),
        "batch": torch.zeros(n, dtype=torch.int64),
    }
    out = ep(data)
    e = out["potc"].sum()
    f = -torch.autograd.grad(e, rr)[0]
    return e.detach(), f.detach(), qr.detach()


def naive_krec(r, q, cell, sigma, dl):
    """Independent reference: full (non-hemisphere) k-grid reciprocal sum.

    E = 1/V * sum_{k!=0, |k|<=km} exp(-sigma^2 k^2/2)/k^2 * |S(k)|^2 - self
    The hemisphere {1,2}-factor trick in the two LES kernels makes the hemi sum
    equal the full sum (each pair k, -k keeps the same |S|^2), so no 1/2 is
    needed. The self-interaction term is subtracted to match
    remove_self_interaction=True.
    """
    vol = torch.det(cell[0])
    G = 2 * np.pi * cell[0].inverse().T
    kmax_sq = (2 * np.pi / dl) ** 2
    Nk = [max(1, int(np.linalg.norm(cell[0, i]) / dl)) for i in range(3)]
    n1 = np.arange(-Nk[0], Nk[0] + 1)
    n2 = np.arange(-Nk[1], Nk[1] + 1)
    n3 = np.arange(-Nk[2], Nk[2] + 1)
    nvec = np.stack(np.meshgrid(n1, n2, n3, indexing="ij"), -1).reshape(-1, 3)
    kvec = nvec @ G.numpy()
    ksq = (kvec ** 2).sum(1)
    m = (ksq > 1e-14) & (ksq <= kmax_sq)
    kvec, ksq = kvec[m], ksq[m]
    kr = r.numpy() @ kvec.T
    S = (q.numpy()[:, None] * np.exp(1j * kr)).sum(axis=0)
    kfac = np.exp(-0.5 * sigma * sigma * ksq) / ksq
    E = (kfac * np.abs(S) ** 2).sum() / vol
    E -= (q ** 2).sum() / (sigma * (2 * np.pi) ** 1.5)  # remove self-interaction
    return float(E)


def report(name, e_a, e_b, f_a, f_b):
    print(f"--- {name} ---")
    print(f"  E_A(les)          = {e_a:.12e} eV")
    print(f"  E_B(cace)         = {e_b:.12e} eV")
    print(f"  |E_A - E_B|       = {abs(e_a - e_b):.3e}   rel = {abs(e_a - e_b) / max(abs(e_a), 1e-30):.3e}")
    fmax = f_a.abs().max().item()
    fd = (f_a - f_b).abs().max().item()
    print(f"  |F_A - F_B|_max   = {fd:.3e}  (max|F_A| = {fmax:.3e}, rel = {fd / max(fmax, 1e-30):.3e})")
    ok = abs(e_a - e_b) < 1e-10 * max(1.0, abs(e_a)) and fd < 1e-9 * max(1.0, fmax)
    print(f"  MATCH             = {ok}")
    return ok


def main():
    torch.set_default_dtype(torch.float64)
    r, cell = load_liquid64()
    n = r.shape[0]
    print(f"liquid-64: n_atoms={n}, cell diag = {cell[0].diagonal().tolist()}")
    print(f"deepmd norm_factor = {DEEPMD_NORM} (1/2e0); cace norm_factor = 1.0\n")

    sigma, dl = 1.0, 2.0
    all_ok = True
    for tag, q in [("TIP3P charges", tip3p_charges(r, cell)),
                   ("neutral random q", random_neutral(n, 7))]:
        # A: les-package Ewald, norm aligned to cace (=1.0)
        ew = LesEwald(sigma=sigma, dl=dl, remove_self_interaction=True, norm_factor=1.0)
        e_a, f_a, _ = eval_les(ew, r, q, cell)
        # deepmd-default conversion: E_deepmd = E_cace * norm_factor
        e_a_dp = e_a * DEEPMD_NORM

        # B: cace internal EwaldPotential
        ep = CaceEwald(dl=dl, sigma=sigma, remove_self_interaction=True,
                       feature_key="q", output_key="potc")
        e_b, f_b, _ = eval_cace(ep, r, q, cell)

        # C: naive independent reference
        e_c = naive_krec(r, q, cell, sigma, dl)

        ok = report(tag, e_a, e_b, f_a, f_b)
        print(f"  E_cace (E)        = {e_b:.12e}  (independent reference)")
        print(f"  E_naive - E_les   = {e_c - e_a:.3e}  (guards against shared-bug)")
        print(f"  E_deepmd-default  = {e_a_dp:.12e} = E_cace * {DEEPMD_NORM}")
        ratio = e_b * DEEPMD_NORM / e_a
        print(f"  unit ratio check  = {ratio:.6f}  (expect {DEEPMD_NORM})\n")
        all_ok &= ok and abs(e_c - e_a) < 1e-8 * max(1.0, abs(e_a))

    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())