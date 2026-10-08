"""End-to-end charge-layer checks through the real hybrid_ener DeepMD model.

check_charge_modes.py exercises the bare ``Les`` module. This script drives the
whole path that training uses: get_model() -> HybridLESModel -> HybridLESAtomicModel
-> Les (with ``ntypes`` / ``element_numbers`` / ``type_index`` injected by the
atomic model), so a wiring mistake in the deepmd-side plumbing shows up here.

Three properties are checked, none of which needs training data:

1. Projection equivalence. A freeze_charge table that does not sum to S, with
   claim_neutral on, must give exactly the same long-range energy as the same
   table pre-shifted so it already sums to S. The projection is a uniform shift
   ``q - (Q - S)/N``, so both paths must produce identical charges; the energies
   are compared bit-for-bit.
2. Energy-force consistency. For every charge-layer mode, the force returned by
   ``forward`` is compared against a central finite difference of the total
   energy (short-range + long-range). This catches a broken autograd path, a
   dtype mismatch, or a charge layer that bypasses the graph.
3. state_dict hygiene. The per-type tables (freeze_charge, initial_guess) are
   non-persistent buffers, so they stay out of the state_dict and existing
   checkpoints still load strictly; freeze mode leaves nothing trainable at all.

Usage: python check_charge_modes_e2e.py
"""
import copy
import sys

import numpy as np
import torch

from _common import BASE_CONFIG, build_water, resolve_device
from deepmd.pt.model.model import get_model
from deepmd.utils.argcheck import normalize

EPS = 1e-6

# The charge layer is a two-way choice, so an inherited config must not carry a
# stale selector alongside the one under test (BASE_CONFIG sets use_atomwise).
MODE_KEYS = ("use_atomwise", "local_charge", "freeze_charge")
BASELINE_KEYS = ("use_fixed_charges", "use_fixed_atomic_charges", "initial_guess")
CONSTRAINT_KEYS = ("claim_neutral", "claim_total_charge")


def build(les_params, device="cpu", seed=0):
    """A hybrid_ener model with BASE_CONFIG's les_params plus the override.

    Any charge-layer selector inherited from BASE_CONFIG is dropped first, so the
    caller names exactly the mode it wants to test.
    """
    cfg = copy.deepcopy(BASE_CONFIG)
    base = cfg["model"]["les_params"]
    for key in MODE_KEYS + BASELINE_KEYS + CONSTRAINT_KEYS:
        base.pop(key, None)
    base.update(les_params)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = get_model(normalize(cfg)["model"])
    return model.to(resolve_device(device)).double().eval()


def total_energy(model, coord, atype, box):
    out = model(coord.detach().clone(), atype, box=box)
    return out["energy"].sum()


def test_projection_equivalence():
    """claim_neutral == pre-shifting the freeze_charge table by -(Q-S)/N."""
    # build_water(n_molecules=1) is one frame of O,H,H, so with a per-type table
    # [O, H] the frame total is Q = table[0] + 2*table[1].
    table = [0.0, 0.5]
    frame_types = [0, 1, 1]
    n_atoms = len(frame_types)
    Q = sum(table[t] for t in frame_types)
    # claim_neutral (S=0) shifts every charge by -(Q - S)/N.
    shift = Q / n_atoms

    coord, atype, box = build_water(n_molecules=1, box_len=9.0, device="cpu")
    assert atype.reshape(-1).tolist() == frame_types
    model_a = build({"freeze_charge": table, "claim_neutral": True})
    model_b = build({"freeze_charge": [v - shift for v in table]})
    e_a = total_energy(model_a, coord, atype, box).item()
    e_b = total_energy(model_b, coord, atype, box).item()
    diff = abs(e_a - e_b)
    ok = diff < 1e-12 * max(1.0, abs(e_a))
    return ok, f"E(claim_neutral)={e_a!r} E(pre-shifted)={e_b!r} |diff|={diff:.3e}"


def les_params_for(label, guess):
    if label == "local":
        return {"use_atomwise": True}
    if label == "local+fixed":
        return {"use_atomwise": True, "use_fixed_charges": True}
    if label == "local+neutral":
        return {"use_atomwise": True, "claim_neutral": True}
    if label == "freeze":
        return {"freeze_charge": [-0.82, 0.41]}
    raise KeyError(label)


def test_energy_force_consistency():
    """The force from forward() equals -dE/dr of the total energy."""
    guess = [-0.82, 0.41]
    labels = ["local", "local+fixed", "local+neutral", "freeze"]
    coord0, atype, box = build_water(n_molecules=2, box_len=9.0, device="cpu")
    rows = []
    all_ok = True
    for label in labels:
        model = build(les_params_for(label, guess))
        f_ag = model(coord0.clone(), atype, box=box)["force"]
        nloc = coord0.shape[1]
        f_fd = torch.zeros_like(f_ag)
        for i in range(nloc):
            for d in range(3):
                cp = coord0.clone()
                cp[0, i, d] += EPS
                ep = total_energy(model, cp, atype, box).item()
                cm = coord0.clone()
                cm[0, i, d] -= EPS
                em = total_energy(model, cm, atype, box).item()
                f_fd[0, i, d] = -(ep - em) / (2 * EPS)
        diff = (f_ag - f_fd).abs().max().item()
        scale = f_fd.abs().max().item()
        ok = np.isfinite(diff) and diff < 1e-4 * max(1.0, scale)
        all_ok = all_ok and ok
        rows.append((label, diff, scale, ok))
    detail = "; ".join(
        f"{l}: max|dF|={d:.2e}/{s:.2e}{'' if o else ' FAIL'}"
        for l, d, s, o in rows
    )
    return all_ok, detail


def test_state_dict_hygiene():
    """The per-type tables stay out of the state_dict; freeze trains nothing."""
    keys_of = lambda m: set(m.atomic_model.les_model.state_dict().keys())
    trainable_of = lambda m: [
        n for n, p in m.atomic_model.les_model.named_parameters() if p.requires_grad
    ]

    base = build({"use_atomwise": True})
    frozen = build({"freeze_charge": [-0.82, 0.41]})
    guessed = build({"use_atomwise": True, "initial_guess": [-0.82, 0.41]})

    ok = (
        # freeze_charge is a fixed table, not a parameter: the framework is a
        # classical Ewald sum and has nothing left to train.
        trainable_of(frozen) == []
        # The guess table is a baseline, not a parameter: local mode still trains
        # the NN only.
        and trainable_of(guessed) == trainable_of(base)
        and trainable_of(base) != []
        # Non-persistent buffers: the tables must not change the key set, else
        # old checkpoints would stop loading strictly.
        and "freeze_charge_table" not in keys_of(frozen)
        and "initial_guess_table" not in keys_of(guessed)
        and "element_numbers" not in keys_of(base)
    )
    return ok, (
        f"trainable(freeze)={trainable_of(frozen)} "
        f"trainable(base)==trainable(guess)="
        f"{trainable_of(guessed) == trainable_of(base)} (n={len(trainable_of(base))})"
    )


def main() -> int:
    tests = [
        ("projection equivalence", test_projection_equivalence),
        ("energy-force consistency", test_energy_force_consistency),
        ("state_dict hygiene", test_state_dict_hygiene),
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
