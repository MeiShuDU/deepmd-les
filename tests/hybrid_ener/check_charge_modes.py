"""Charge-layer checks for the LES module: local / freeze.

The charge layer decides where the latent charges q come from, and the two
options are mutually exclusive:

- ``local_charge=True``   q = Atomwise(descriptor)   (per atom, NN)
- ``freeze_charge=[...]`` q = per-type constant       (no SGD: classical Ewald)

``assign_charge`` (a trainable per-atom charge vector) existed and was removed:
a charge indexed by an atom's slot in the tensor is a lookup by atom ID, not a
function of the configuration, so it cannot transfer to a new frame.

``freeze_charge`` is the classical-Ewald end of the spectrum: the charges are
given up front and never updated, so the framework is a plain Ewald sum on top of
a fixed charge table.

Two per-element baselines may be attached to local mode:

- ``use_fixed_charges``   oxidation-number baseline (FixedCharges, factor 0.5)
- ``initial_guess``       user-supplied per-type charges

Both are added on top of the NN output on every forward. freeze mode rejects
both, because the charges are already given.

``claim_total_charge=S`` (or ``claim_neutral`` for S=0) projects the charges of
every frame onto the ``sum(q) = S`` plane. The projection is linear and
idempotent, so it holds exactly at every step while gradients still flow.

Everything runs on CPU with float64. No training data or checkpoint is needed:
the checks are properties of the code path, not of fitted values.

Usage: python check_charge_modes.py
"""
import sys

import torch

from les import Les

# type_map is ["O", "H"] throughout: index 0 = O, index 1 = H.
NTYPES = 2
ELEMENT_NUMBERS = [8, 1]  # O, H
# FixedCharges(typical_charge, normalization_factor=0.5): O -> -2*0.5, H -> +1*0.5
O_BASELINE, H_BASELINE = -1.0, 0.5

DEVICE = torch.device("cpu")
DTYPE = torch.float64

# One frame: O, H, H.
TYPE_INDEX_1 = torch.tensor([0, 1, 1], dtype=torch.int64)
DESC_DIM = 8


def make_les(**kwargs) -> Les:
    """A bare Les module with the water type map already wired in."""
    args = {
        "dim_descrpt": DESC_DIM,
        "ntypes": NTYPES,
        "element_numbers": ELEMENT_NUMBERS,
        "sigma": 1.0,
        "dl": 1.5,
        "verbose": False,
    }
    args.update(kwargs)
    les = Les(les_arguments=args)
    return les.to(device=DEVICE, dtype=DTYPE)


def one_frame(nframes: int = 1, atoms: int = 3):
    """A small periodic system of `nframes` frames of `atoms` atoms each."""
    torch.manual_seed(0)
    positions = torch.randn(nframes * atoms, 3, dtype=DTYPE) * 3.0
    cell = (torch.eye(3, dtype=DTYPE) * 12.0).unsqueeze(0).repeat(nframes, 1, 1)
    batch = torch.arange(nframes, dtype=torch.int64).repeat_interleave(atoms)
    type_index = TYPE_INDEX_1.repeat(nframes)
    atomic_numbers = torch.tensor(ELEMENT_NUMBERS, dtype=torch.int64)[type_index]
    desc = torch.randn(nframes * atoms, DESC_DIM, dtype=DTYPE)
    return dict(
        positions=positions,
        cell=cell,
        batch=batch,
        type_index=type_index,
        atomic_numbers=atomic_numbers,
        desc=desc,
    )


def latent_of(les: Les, **kwargs) -> torch.Tensor:
    """Run the module and return latent charges as [n_atoms, 1]."""
    out = les(compute_energy=False, **kwargs)
    q = out["latent_charges"]
    assert q is not None
    return q


def close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-12) -> bool:
    return torch.allclose(a.reshape(-1), b.reshape(-1), atol=tol, rtol=0.0)


def per_atom(per_type_values) -> torch.Tensor:
    """Expand a per-type value list to a per-atom column using TYPE_INDEX_1."""
    t = torch.tensor(per_type_values, dtype=DTYPE)
    return t[TYPE_INDEX_1].reshape(-1, 1)


# --------------------------------------------------------------------------- #
# charge sources
# --------------------------------------------------------------------------- #
def test_freeze_values():
    """freeze_charge gives exactly the per-type constants, per atom."""
    les = make_les(freeze_charge=[-0.82, 0.41])
    f = one_frame()
    q = latent_of(les, **f)
    ok = close(q, per_atom([-0.82, 0.41]))
    # The framework degrades to a classical Ewald sum: nothing is trainable.
    n_train = sum(p.numel() for p in les.parameters() if p.requires_grad)
    ok = ok and n_train == 0
    return ok, f"q={q.reshape(-1).tolist()} trainable_params={n_train}"


def test_local_baselines():
    """In local mode the baselines are added on top of the NN output."""
    f = one_frame()
    torch.manual_seed(11)
    base = make_les(use_atomwise=True)
    q_nn = latent_of(base, **f)

    torch.manual_seed(11)
    les_fixed = make_les(use_atomwise=True, use_fixed_charges=True)
    d_fixed = latent_of(les_fixed, **f) - q_nn

    guess = [-0.82, 0.41]
    torch.manual_seed(11)
    les_guess = make_les(use_atomwise=True, initial_guess=guess)
    d_guess = latent_of(les_guess, **f) - q_nn

    torch.manual_seed(11)
    les_both = make_les(
        use_atomwise=True, use_fixed_charges=True, initial_guess=guess
    )
    d_both = latent_of(les_both, **f) - q_nn

    ok = (
        close(d_fixed, per_atom([O_BASELINE, H_BASELINE]))
        and close(d_guess, per_atom(guess))
        and close(d_both, d_fixed + d_guess)
        # The NN output itself is unchanged by adding a baseline (no re-scaling).
        and not close(q_nn, torch.zeros_like(q_nn))
    )
    return ok, (
        f"d_fixed={d_fixed.reshape(-1).tolist()} d_guess={d_guess.reshape(-1).tolist()} "
        f"d_both=d_fixed+d_guess -> {close(d_both, d_fixed + d_guess)}"
    )


# --------------------------------------------------------------------------- #
# total-charge constraint
# --------------------------------------------------------------------------- #
def test_claim_total_charge():
    """claim_total_charge/claim_neutral make every frame sum to exactly S."""
    nframes, atoms = 3, 3
    f = one_frame(nframes=nframes, atoms=atoms)

    # The unconstrained local charges, from the same weights as the projected
    # ones: they must NOT already sum to S, else the projection proves nothing.
    torch.manual_seed(7)
    raw_sums = latent_of(make_les(use_atomwise=True), **f).reshape(
        nframes, atoms).sum(dim=1)

    results = {}
    for label, kwargs, want in [
        ("neutral", {"claim_neutral": True}, 0.0),
        ("S=1", {"claim_total_charge": 1}, 1.0),
        ("S=-0.5", {"claim_total_charge": -0.5}, -0.5),
    ]:
        torch.manual_seed(7)
        les = make_les(use_atomwise=True, **kwargs)
        sums = latent_of(les, **f).reshape(nframes, atoms).sum(dim=1)
        results[label] = (
            sums.tolist(),
            close(sums, torch.full((nframes,), want, dtype=DTYPE)),
        )
    ok = all(v[1] for v in results.values()) and not close(
        raw_sums, torch.full((nframes,), 0.0, dtype=DTYPE))
    return ok, "; ".join(
        f"{k}: sums={[round(x, 10) for x in v[0]]}" for k, v in results.items()
    ) + f"; unconstrained sums={[round(x, 6) for x in raw_sums.tolist()]}"


def test_claim_gradient_flows():
    """The projected charges still receive gradients (option 1, not a penalty)."""
    torch.manual_seed(7)
    les = make_les(use_atomwise=True, claim_neutral=True)
    f = one_frame(nframes=1)
    q = latent_of(les, **f)
    loss = q.pow(2).sum()
    loss.backward()
    params = [p for p in les.parameters() if p.requires_grad]
    grads = [p.grad for p in params if p.grad is not None]
    n_nonzero = sum(1 for g in grads if g.abs().sum().item() > 0)
    ok = (
        loss.item() > 0.0
        and len(grads) == len(params)
        and n_nonzero > 0
        and all(bool(torch.isfinite(g).all()) for g in grads)
    )
    return ok, (
        f"loss={loss.item():.6e} params={len(params)} with_grad={len(grads)} "
        f"nonzero_grad={n_nonzero}"
    )


# --------------------------------------------------------------------------- #
# configuration errors
# --------------------------------------------------------------------------- #
def test_errors():
    """Invalid combinations are rejected at construction time."""
    cases = [
        ("local+freeze", {"use_atomwise": True, "freeze_charge": [0.0, 0.0]}),
        ("freeze+fixed", {"freeze_charge": [0.0, 0.0], "use_fixed_charges": True}),
        ("freeze+guess", {"freeze_charge": [0.0, 0.0], "initial_guess": [0.0, 0.0]}),
        ("neutral+both", {"claim_neutral": True, "claim_total_charge": 1}),
        # ntypes is injected by HybridLESAtomicModel from type_map and is needed
        # by the per-type tables (freeze / initial_guess / use_fixed_charges).
        (
            "guess_no_ntypes",
            {"use_atomwise": True, "initial_guess": [0.1, 0.2], "ntypes": 0},
        ),
        ("wrong_table_len", {"use_atomwise": True, "initial_guess": [0.1, 0.2, 0.3]}),
    ]
    outcomes = []
    for name, kwargs in cases:
        try:
            make_les(**kwargs)
        except (ValueError, AssertionError, TypeError) as exc:
            outcomes.append((name, True, type(exc).__name__))
        else:
            outcomes.append((name, False, "no error raised"))
    ok = all(o[1] for o in outcomes)
    return ok, "; ".join(
        f"{n}:{'ok' if good else 'MISSING'} ({why})" for n, good, why in outcomes
    )


# --------------------------------------------------------------------------- #
# scriptability
# --------------------------------------------------------------------------- #
def build_modes():
    """One Les per charge-layer configuration, plus a label."""
    guess = [-0.82, 0.41]
    return [
        ("local", dict(use_atomwise=True)),
        ("local+fixed", dict(use_atomwise=True, use_fixed_charges=True)),
        ("local+guess", dict(use_atomwise=True, initial_guess=guess)),
        (
            "local+fixed+guess",
            dict(use_atomwise=True, use_fixed_charges=True, initial_guess=guess),
        ),
        ("freeze", dict(freeze_charge=[-0.82, 0.41])),
        ("neutral", dict(use_atomwise=True, claim_neutral=True)),
        ("claim_S=1", dict(use_atomwise=True, claim_total_charge=1)),
    ]


def test_script():
    """Every mode scripts under torch.jit.script and matches eager bit-for-bit."""
    f = one_frame(nframes=2)
    rows = []
    for label, kwargs in build_modes():
        torch.manual_seed(5)
        les = make_les(**kwargs)
        eager = les(compute_energy=True, **f)
        q_e = eager["latent_charges"]
        e_e = eager["E_lr"]
        scripted = torch.jit.script(les)
        out = scripted(
            positions=f["positions"],
            cell=f["cell"],
            desc=f["desc"],
            batch=f["batch"],
            atomic_numbers=f["atomic_numbers"],
            type_index=f["type_index"],
            compute_energy=True,
        )
        q_d = (out["latent_charges"] - q_e).abs().max().item()
        e_d = (out["E_lr"] - e_e).abs().max().item()
        rows.append((label, q_d, e_d))
    ok = all(qd == 0.0 and ed == 0.0 for _l, qd, ed in rows)
    return ok, "; ".join(f"{l}: dq={qd:.1e} dE={ed:.1e}" for l, qd, ed in rows)


def main() -> int:
    tests = [
        ("freeze values", test_freeze_values),
        ("local baselines", test_local_baselines),
        ("claim total charge", test_claim_total_charge),
        ("claim gradient", test_claim_gradient_flows),
        ("config errors", test_errors),
        ("torchscript parity", test_script),
    ]
    failed = 0
    for name, fn in tests:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 - surface every failure, keep going
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        if not ok:
            failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print("RESULT:", "PASS" if failed == 0 else f"FAIL ({failed} test(s))")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
