"""Ewald-kernel reference checks for the LES module.

The charge layer is useless if the Ewald kernel underneath it is not the physical
Coulomb interaction, so this file pins the kernel to analytic answers rather than
to any trained model. It needs no training data and no checkpoint - every claim
below is a property of the kernel, not of fitted values.

Three properties are checked:

1. Normalisation. An isolated +1/-1 pair in a large box must return the physical
   Coulomb energy 14.3996 q1 q2 / r (eV, with r in A). This is the check that
   catches a misapplied ``norm_factor``: les advertises
   ``norm_factor = 1/(2 eps_0) = 90.4756``, which is 2*pi times the physical
   constant, so the ``/ 2 pi`` has to actually happen somewhere and a silent
   factor of 2*pi is exactly the kind of bug this catches.
2. Bilinearity. E_lr must be bilinear in the charges, so scaling a neutral pair
   by L scales the energy by L^2. A kernel carrying a spurious per-charge term
   would break this.
3. Net-charge behaviour. A frame whose charges do not sum to zero picks up an
   extra term proportional to Q^2 - the kernel has no jellium background. This is
   documented rather than asserted away, because it is the physical reason
   ``claim_neutral`` exists: an unconstrained local_charge that lets Q drift is
   optimised against a shifted energy.

The box is deliberately huge (100 A) in the pair tests so periodic images are
negligible and the analytic answer is unambiguous. That matters: on a real 13 A
water box the exact sum over images differs from a minimum-image Coulomb sum by
tens of percent, and the Ewald result is the physical one - so a naive
minimum-image comparison would "find" a bug that is not there.

Usage: python check_ewald_reference.py
"""
import sys

import torch

from les import Les

COULOMB = 14.3996  # eV*A, the physical e^2 / (4 pi eps0)
BIG_BOX = 100.0
DTYPE = torch.float64


def make_les(**kwargs) -> Les:
    args = {"dim_descrpt": 8, "ntypes": 2, "element_numbers": [8, 1],
            "sigma": 1.0, "dl": 1.5, "verbose": False}
    args.update(kwargs)
    return Les(les_arguments=args).to(device="cpu", dtype=DTYPE)


def ewald_energy(charges, positions, box=BIG_BOX) -> float:
    """E_lr for explicit charges given straight to the kernel (no charge layer)."""
    les = make_les()
    n = len(charges)
    q = torch.tensor(charges, dtype=DTYPE)
    pos = torch.tensor(positions, dtype=DTYPE)
    cell = (torch.eye(3, dtype=DTYPE) * box).unsqueeze(0)
    batch = torch.zeros(n, dtype=torch.int64)
    out = les(positions=pos, cell=cell, batch=batch, latent_charges=q,
              compute_energy=True)
    return float(out["E_lr"].sum())


def pair(q1, q2, d):
    return ewald_energy([q1, q2], [[0.0, 0.0, 0.0], [d, 0.0, 0.0]])


def lone(q):
    return ewald_energy([q], [[0.0, 0.0, 0.0]])


def test_normalisation():
    """A neutral pair must reproduce 14.3996 q1 q2 / r in a large box."""
    rows = []
    ok = True
    for d in (3.0, 5.0, 10.0):
        got = pair(1.0, -1.0, d)
        want = COULOMB * (-1.0) / d
        rel = abs(got / want - 1.0)
        # 1% covers the residual image tail; a 2*pi or factor-2 error is ~528%.
        good = rel < 0.01
        ok = ok and good
        rows.append(f"d={d:4.1f}: {got:+.6e} vs {want:+.6e} (rel {rel:.2e})"
                    f"{'' if good else ' FAIL'}")
    return ok, "; ".join(rows)


def test_bilinearity():
    """E_lr is bilinear in the charges, on top of the separate Q^2 term.

    Scaling a NEUTRAL pair by L must scale E by L^2. Scaling only ONE charge of a
    pair leaves a net charge, so that case is bilinear part + Q^2 term - checked
    explicitly so the two effects are not mistaken for a broken kernel.
    """
    d = 5.0
    e1 = pair(1.0, -1.0, d)
    e5 = pair(0.5, -0.5, d)
    ok = abs(e5 / e1 - 0.25) < 1e-9
    e_partial = pair(1.0, -0.5, d)
    want_partial = COULOMB * (-0.5) / d + 0.25 * lone(1.0)
    rel = abs(e_partial / want_partial - 1.0)
    ok = ok and rel < 0.01
    return ok, (f"neutral x0.5: {e5:+.6e} = 0.25 x {e1:+.6e}; "
                f"one-charge-halved: {e_partial:+.6e} vs bilinear+Q^2 "
                f"{want_partial:+.6e} (rel {rel:.2e})")


def test_net_charge():
    """A non-neutral frame carries an extra Q^2 term (no jellium background)."""
    e1 = lone(1.0)
    e2 = lone(2.0)
    e05 = lone(0.5)
    # Q^2 scaling: E(Q) / E(1) == Q^2.
    r2 = e2 / e1
    r05 = e05 / e1
    ok = abs(r2 - 4.0) < 1e-9 and abs(r05 - 0.25) < 1e-9
    # And it is genuinely a *shift*: a +1/+1 pair is the physical repulsion plus
    # the Q=+2 term (4 * e1), not the physical repulsion alone.
    e_pair = pair(1.0, 1.0, 5.0)
    rest = e_pair - COULOMB * 1.0 / 5.0
    ok = ok and abs(rest / e2 - 1.0) < 0.01
    return ok, (f"lone Q=1 -> {e1:+.6e} eV; E(2)/E(1)={r2:.6f} (want 4), "
                f"E(.5)/E(1)={r05:.6f} (want .25); +1/+1 pair {e_pair:+.6e} = "
                f"physical {COULOMB / 5.0:+.6e} + Q^2 term {rest:+.6e} "
                f"(~= E(2)={e2:+.6e}) - so a drifting total charge shifts the "
                f"energy, which is why claim_neutral is not cosmetic")


def test_neutral_pair_is_unaffected_by_the_q2_term():
    """The Q^2 term vanishes for a neutral frame, so it cannot touch the fit."""
    e = pair(1.0, -1.0, 5.0)
    want = COULOMB * (-1.0) / 5.0
    ok = abs(e / want - 1.0) < 0.01
    return ok, f"neutral pair {e:+.6e} vs {want:+.6e} (no Q^2 term present)"


def main() -> int:
    tests = [
        ("normalisation (neutral pair = 14.3996 q1q2/r)", test_normalisation),
        ("bilinearity in the charge product", test_bilinearity),
        ("net-charge Q^2 term", test_net_charge),
        ("neutral frame carries no Q^2 term",
         test_neutral_pair_is_unaffected_by_the_q2_term),
    ]
    failed = 0
    for name, fn in tests:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 - surface every failure
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        if not ok:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("RESULT:", "PASS" if failed == 0 else f"FAIL ({failed} test(s))")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
