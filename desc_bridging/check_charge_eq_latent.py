"""End-to-end checks for ``charge_eq_latent`` on real campaign frames.

This drives the path training uses - ``build_model`` on the campaign's own
trained configuration, over real frames of the ``campaign_water_interface`` slab
- rather than the bare module, so a wiring mistake in ``model.py`` or
``config.py`` shows up here. The trained ``sea-lr_sA`` checkpoint is rebuilt with
the equalization switched on; the module adds no parameters, so the same state
dict loads strictly and the same charges are equalized.

What is checked, and why each one is worth a test:

1. ``coulomb_matrix`` is the kernel's field matrix. The closed form this module
   assembles in ``O(n^2)`` memory is compared entry by entry against
   ``EwaldPotential``'s own ``compute_field=True`` output, on a real frame's
   geometry, in both kernel branches and both self-interaction settings. Without
   this the rest of the file would only be checking that the module is
   self-consistent with a matrix that might be wrong.
2. The energy identity. For the same charges, ``0.5 sum_c q_c^T A q_c`` must equal
   the energy the kernel reports. It does, except in the one configuration where
   the kernel counts the self term once per channel; that case is asserted to
   differ by exactly ``(n_out - 1) * sum_c sum_i q_ic^2 / (sigma (2 pi)^1.5)``, so
   the exception is pinned rather than tolerated.
3. The constraint ``sum_c 1^T q_c == total_charge``, including a non-zero request.
4. Stationarity: ``(A + 2w I) q_c + multiplier`` must be the same constant vector
   on every channel, which is the whole content of the single charge constraint.
5. ``w -> infinity`` recovers the plain latent-charge term, so the module is a
   strict generalization of ``EwaldPotential`` and not a different model.
6. The long-range force is the exact autograd derivative of the reported energy,
   against a central finite difference. The equalization solves a linear system
   inside the graph, so this is the test that the solve is differentiable at all.

The sweep at the end prints ``||q_eq - q_r||`` against ``w`` so the default has a
measured basis rather than a guessed one.

One dependency quirk worth stating up front, because it decides how this file
reads the energy: in the campaign's ``combine_potentials`` path,
``cace.models.combined.CombinePotential.forward`` scales each potential's output
by ``weight`` **in place** (``v_now *= potential_key['weight']``, where the tensor
still belongs to the shared ``data`` dict the sub-models returned). So after a
combined forward, ``data['ewald_potential']`` holds ``weight * E_lr``, not the
energy the long-range module reported. That does not touch the training loss
(``CACE_energy``/``CACE_forces`` are the correctly weighted sums), but it does
mean a diagnostic that reads the raw long-range key off a combined model sees it
scaled by the weight. This file therefore captures the energy at the module
boundary rather than from the returned dict, and prints the ratio it observes so
the scaling is visible rather than silently divided out.

Usage: PYTHONPATH=desc_bridging python desc_bridging/check_charge_eq_latent.py
"""
import copy
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from cace.modules.ewald import EwaldPotential  # noqa: E402
from deepmd.pt.model.descriptor.se_a import DescrptSeA  # noqa: E402
from deepmd_cace.charge_eq_latent import (  # noqa: E402
    ChargeEqLatent,
    coulomb_matrix,
    equalize_charges,
)
from deepmd_cace.data import load_split  # noqa: E402
from deepmd_cace.model import build_model  # noqa: E402

CAMPAIGN = REPO / "DeePMD-kit-FastLearn" / "campaign_water_interface"
VALID_SYSTEM = CAMPAIGN / "data" / "water-interface" / "valid"
SEA_LR_CHECKPOINT = CAMPAIGN / "deepmd" / "runs" / "sea-lr_sA" / "best_model.pth"
CUTOFF = 5.5
TYPE_MAP = ["O", "H"]
N_FRAMES = 3
WEIGHT = 0.1


def frames_as_dicts(limit):
    loader, _ = load_split(
        [str(VALID_SYSTEM)], TYPE_MAP, CUTOFF, 1,
        shuffle=False, collect_stats=False, atomic_energies=None,
    )
    frames = []
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        frames.append(batch.to_dict())
    return frames


def _to_device(batch, device):
    data = {key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()}
    data["positions"] = data["positions"].detach().clone().requires_grad_(True)
    return data


def frame_box(data, frame=0):
    return data["cell"].view(-1, 3, 3)[frame]


def kernel_energy(kernel, r, q, cell):
    """What ``EwaldPotential`` reports for these charges, via the branch it picks."""
    diagonal = cell.diagonal()
    if bool((diagonal.abs() < 1e-6).all()):
        return float(kernel.compute_potential_realspace(r, q)[0].sum())
    return float(kernel.compute_potential_triclinic(r, q, cell)[0].sum())


def forward_capturing_long_range(model, module, data, training=True):
    """Run ``model`` and return its outputs plus the module's own long-range energy.

    The energy is read off the module's forward hook, i.e. before the enclosing
    ``CombinePotential`` touches it, for the in-place-weight reason in the module
    docstring. It is returned un-detached, still attached to the graph, so the
    finite-difference check can differentiate it. ``outputs`` is the ordinary
    model return, kept so the charge outputs and the in-place-scaled value can
    both be inspected.
    """
    captured = {}

    def remember(_module, _inputs, output):
        # cloned, because the tensor handed back is the very one CombinePotential
        # scales in place a moment later; the clone is value-frozen and still
        # carries the graph, which is what the finite difference below needs
        captured["energy"] = output[module.ewald_key].clone()

    handle = module.register_forward_hook(remember)
    try:
        outputs = model(data, training=training)
    finally:
        handle.remove()
    return outputs, captured["energy"].reshape(-1)[0]


# ---------------------------------------------------------------------------
# 1 and 2: the matrix, and the identity that ties the energy to it
# ---------------------------------------------------------------------------
def check_matrix_and_identity(frames):
    """Pin the closed-form Coulomb matrix to the kernel, on real geometry."""
    print("1/2. coulomb_matrix against the kernel's own field matrix, and the energy identity")
    # a real frame's geometry, truncated to a size the kernel's eye trick can
    # still hold: 60 atoms is ample for an entry-by-entry comparison
    data = frames[0]
    n_atoms = 60
    r = data["positions"][:n_atoms]
    cell = frame_box(data)
    print(f"   {n_atoms} atoms of a real campaign frame, box "
          f"{[round(v, 2) for v in cell.diagonal().tolist()]}")

    for remove_self in (False, True):
        kernel = EwaldPotential(dl=2.0, sigma=1.0, remove_self_interaction=remove_self,
                                compute_field=True)
        ours = coulomb_matrix(kernel, r, cell)
        reference = kernel.compute_potential_triclinic(
            r, torch.eye(n_atoms), cell, compute_field=True)[1]
        gap = float((ours - reference).abs().max())
        assert gap <= 1e-6 * float(reference.abs().max()), (
            f"remove_self_interaction={remove_self}: matrix differs by {gap}")
        print(f"   remove_self_interaction={remove_self}: max|closed form - kernel| = {gap:.2e} "
              f"on a scale of {float(reference.abs().max()):.4f}")
        assert float((ours - ours.T).abs().max()) <= 1e-6 * float(ours.abs().max()), \
            "the Coulomb matrix must be symmetric"

        # the identity, and the one configuration where the kernel breaks it
        for n_out in (1, 4):
            torch.manual_seed(n_out)
            q = torch.randn(n_atoms, n_out) * 0.3
            reported = kernel_energy(kernel, r, q, cell)
            quadratic = float(0.5 * sum(q[:, c] @ ours @ q[:, c] for c in range(n_out)))
            # the kernel subtracts the whole charge's self term once per channel,
            # so its energy is low by (n_out - 1) of them
            expected_gap = (-(n_out - 1) * float((q ** 2).sum())
                            / (kernel.sigma * kernel.twopi ** 1.5))
            if remove_self and n_out > 1:
                gap = reported - quadratic
                assert abs(gap - expected_gap) <= 1e-3 * abs(expected_gap), (
                    f"n_out={n_out}: gap {gap} but the channel-counted self term is "
                    f"{expected_gap}")
                print(f"      n_out={n_out}: kernel over-counts the self term by {gap:.2e} "
                      f"= (n_out-1) sum q^2 / (sigma (2pi)^1.5), as predicted")
            else:
                assert abs(reported - quadratic) <= 1e-5 * max(1.0, abs(reported)), (
                    f"remove_self_interaction={remove_self} n_out={n_out}: "
                    f"0.5 q^T A q {quadratic} != kernel {reported}")
                print(f"      n_out={n_out}: 0.5 sum_c q_c^T A q_c == kernel energy "
                      f"({reported:+.6f})")

    # the aperiodic branch, which the campaign never takes but a slab in vacuum does
    kernel = EwaldPotential(dl=2.0, sigma=1.0, remove_self_interaction=True, compute_field=True)
    ours = coulomb_matrix(kernel, r, torch.zeros(3, 3))
    reference = kernel.compute_potential_realspace(r, torch.eye(n_atoms), compute_field=True)[1]
    gap = float((ours - reference).abs().max())
    assert gap <= 1e-6 * float(reference.abs().max()), f"realspace matrix differs by {gap}"
    print(f"   aperiodic branch: max|closed form - kernel| = {gap:.2e}")

    # and the point of the closed form: it scales, the kernel's own path does not
    print(f"   the kernel's own path materializes [n, n, n_k]; this one keeps [n, n]")
    print()


# ---------------------------------------------------------------------------
# 3 to 6: through the trained model, on real frames
# ---------------------------------------------------------------------------
def check_trained_model(frames, device):
    print("3-6. the trained sea-lr checkpoint, rebuilt with the equalization on")
    if not SEA_LR_CHECKPOINT.is_file():
        raise SystemExit(f"missing campaign checkpoint {SEA_LR_CHECKPOINT}")
    blob = torch.load(SEA_LR_CHECKPOINT, map_location="cpu", weights_only=False)
    assert blob["format"] == "deepmd-cace-state-dict-1", blob["format"]
    config = copy.deepcopy(blob["config"])
    config["model"]["long_range"]["charge_eq_latent"] = {
        "enabled": True, "regularization_weight": WEIGHT,
    }
    n_out = config["model"]["long_range"]["charge_net"].get("n_out", 1)
    remove_self = config["model"]["long_range"]["ewald"].get("remove_self_interaction", True)

    def rebuild(model_config):
        descriptor = DescrptSeA(**{
            key: value for key, value in model_config["descriptor"].items() if key != "type"
        })
        model = build_model(model_config, descriptor, device)
        model.load_state_dict(blob["state_dict"], strict=True)
        return model.eval()

    model = rebuild(config["model"])
    plain = rebuild({**config["model"], "long_range": {
        k: v for k, v in config["model"]["long_range"].items() if k != "charge_eq_latent"}})
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == sum(p.numel() for p in plain.parameters()), \
        "enabling the equalization must not change the parameter count"
    print(f"   n_out={n_out}, remove_self_interaction={remove_self}, {n_params} parameters; "
          f"the same state dict loads with the equalization off")
    module = next(m for m in model.modules() if isinstance(m, ChargeEqLatent))
    combined = bool(config["model"]["long_range"].get("combine_potentials", False))
    combine_weight = float(config["model"]["long_range"].get("weight", 1.0)) if combined else 1.0

    for index, batch in enumerate(frames):
        data = _to_device(batch, device)
        # the leaf handed in, kept because the model replaces data["positions"]
        # with a copy during the forward; the values are identical, but only this
        # tensor is the one autograd below can differentiate against
        leaf = data["positions"]
        with torch.enable_grad():
            outputs, energy = forward_capturing_long_range(model, module, data)
        energy_value = float(energy.detach())
        q_ref, q_eq = outputs["q"].detach(), outputs["q_eq"].detach()
        box = frame_box(data)
        mask = data["batch"] == 0

        # the in-place weight scaling, pinned so it is visible and not silently
        # divided out; the identity below uses the module's own number
        reported_key = float(outputs["ewald_potential"].reshape(-1)[0])
        assert abs(reported_key - combine_weight * energy_value) <= 1e-5 * max(
            1.0, abs(reported_key)), (
            f"frame {index}: returned ewald_potential {reported_key} is not "
            f"{combine_weight} x the module's {energy_value}")

        # 2, again, now on the module's own reported number
        with torch.no_grad():
            A_mat = module._coulomb_matrix(leaf.detach()[mask], box)
        quadratic = 0.5 * sum(q_eq[:, c] @ A_mat @ q_eq[:, c] for c in range(n_out))
        assert abs(energy_value - float(quadratic)) <= 1e-5 * max(1.0, abs(float(quadratic))), (
            f"frame {index}: reported {energy_value} != 0.5 sum_c q_c^T A q_c "
            f"{float(quadratic)}")
        # and against the kernel on the equalized charges, except in the one
        # configuration where the kernel over-counts the self term
        kernel = EwaldPotential(dl=module.ep.dl, sigma=module.ep.sigma,
                                remove_self_interaction=remove_self)
        reported = kernel_energy(kernel, leaf.detach()[mask], q_eq, box)
        if remove_self and n_out > 1:
            expected_gap = (-(n_out - 1) * float((q_eq ** 2).sum())
                            / (kernel.sigma * kernel.twopi ** 1.5))
            assert abs((reported - energy_value) - expected_gap) <= 1e-3 * abs(expected_gap), (
                f"frame {index}: kernel gap is not the predicted self-term over-count")
        else:
            assert abs(reported - energy_value) <= 1e-5 * max(1.0, abs(reported)), (
                f"frame {index}: module {energy_value} != kernel {reported}")

        # 3: the constraint, on the sum over atoms *and* channels
        totals = q_eq.sum()
        assert abs(float(totals) - module.total_charge) <= 1e-4, (
            f"frame {index}: equalized total charge {float(totals)} != "
            f"{module.total_charge}")

        # 4: stationarity, i.e. the residual is one constant vector over channels
        residual = ((A_mat + 2 * WEIGHT * torch.eye(A_mat.shape[0])) @ q_eq[mask]
                    - 2 * WEIGHT * q_ref[mask])
        spread = float((residual - residual[:, :1]).abs().max())
        assert spread <= 1e-3 * float(residual.abs().max()) + 1e-6, (
            f"frame {index}: stationarity residual varies across channels by {spread}")

        # 5: the limit w -> infinity. The trust region pins the charges, but the
        # constraint does not disappear: what survives is a uniform shift
        # s = (sum(q_r) - Q) / (n_atoms * n_channels), independent of w, so the
        # limit is q_r - s (broadcast) and not the raw latent charges. The channel
        # count is in the denominator because the constraint sums over channels as
        # well as atoms. Asserting against that limit tests both the trust region
        # and the constraint at once; asserting against q_r would only fail,
        # because the charge head's proposal carries a net charge.
        shift = (float(q_ref.sum()) - module.total_charge) / q_ref.numel()
        limited = q_ref - shift
        limit_energy = float(0.5 * sum(limited[:, c] @ A_mat @ limited[:, c]
                                       for c in range(n_out)))
        strict_q, _ = equalize_charges(A_mat, q_ref, 1e8, module.total_charge)
        strict_energy = float(0.5 * sum(strict_q[:, c] @ A_mat @ strict_q[:, c]
                                        for c in range(n_out)))
        assert abs(strict_energy - limit_energy) <= 1e-4 * max(1.0, abs(limit_energy)), (
            f"frame {index}: w->inf energy {strict_energy} != constrained limit "
            f"{limit_energy}")

        # 6: the reported force is the autograd derivative of the reported energy
        positions = leaf
        autograd = -torch.autograd.grad(energy, positions, retain_graph=False)[0].detach()
        step, worst = 1e-2, 0.0
        for atom in (0, positions.shape[0] // 2, positions.shape[0] - 1):
            for axis in range(3):
                moved = []
                for sign in (1.0, -1.0):
                    probe = dict(data)
                    shifted = positions.detach().clone()
                    shifted[atom, axis] += sign * step
                    probe["positions"] = shifted.requires_grad_(True)
                    with torch.enable_grad():
                        moved.append(float(forward_capturing_long_range(
                            model, module, probe, training=True)[1].detach()))
                worst = max(worst, abs(-(moved[0] - moved[1]) / (2 * step)
                                       - float(autograd[atom, axis])))
        scale = float(autograd.abs().max())
        assert worst <= 1e-2 * max(1e-9, scale) + 1e-5, (
            f"frame {index}: autograd force differs from finite difference by {worst} "
            f"against a force scale of {scale}")

        print(f"   frame {index}: E_lr {float(energy):+.6f}  kernel {reported:+.6f}  "
              f"sum q_eq {float(totals):+.1e}  |F_lr|max {scale:.4f}  "
              f"fd_gap {worst:.2e}  w->inf gap {abs(strict_energy - limit_energy):.1e}"
              f"  returned/raw {reported_key / float(energy):.4f}")
    print()


# ---------------------------------------------------------------------------
# the request, and the default
# ---------------------------------------------------------------------------
def check_scale(frames, device):
    print("the total charge, and the w scale the default is chosen from")
    blob = torch.load(SEA_LR_CHECKPOINT, map_location="cpu", weights_only=False)
    config = copy.deepcopy(blob["config"])
    config["model"]["long_range"]["charge_eq_latent"] = {
        "enabled": True, "regularization_weight": WEIGHT,
    }
    descriptor = DescrptSeA(**{
        key: value for key, value in config["model"]["descriptor"].items() if key != "type"})
    model = build_model(config["model"], descriptor, device)
    model.load_state_dict(blob["state_dict"], strict=True)
    module = next(m for m in model.modules() if isinstance(m, ChargeEqLatent))

    data = _to_device(frames[0], device)
    mask = data["batch"] == 0
    A_mat = module._coulomb_matrix(data["positions"][mask], frame_box(data))
    with torch.enable_grad():
        q_ref = model(data, training=False)["q"].detach()[mask]

    for total in (0.0, 0.01, -0.05):
        q_eq, multiplier = equalize_charges(A_mat, q_ref, WEIGHT, total)
        print(f"   total_charge {total:+.3f}: achieved {float(q_eq.sum()):+.3e}  "
              f"multiplier {float(multiplier):+.6f}  "
              f"rms(q_eq - q_r) {float((q_eq - q_ref).pow(2).mean().sqrt()):.4e}")
    print()

    reference = float(0.5 * sum(q_ref[:, c] @ A_mat @ q_ref[:, c]
                               for c in range(q_ref.shape[1])))
    scale = float(q_ref.pow(2).mean().sqrt())
    print(f"   {'w':>9} {'rms(q_eq-q_r)':>14} {'rms(q_r)':>11} {'movement':>9} "
          f"{'E_lr':>11} {'E_lr(q_r)':>11}")
    for trial in (100.0, 10.0, 1.0, WEIGHT, 0.01, 1e-3):
        q_eq, _ = equalize_charges(A_mat, q_ref, trial, 0.0)
        shift = float((q_eq - q_ref).pow(2).mean().sqrt())
        energy = float(0.5 * sum(q_eq[:, c] @ A_mat @ q_eq[:, c]
                                 for c in range(q_eq.shape[1])))
        print(f"   {trial:>9.3g} {shift:>14.4e} {scale:>11.4e} {shift / scale:>9.3f} "
              f"{energy:>11.6f} {reference:>11.6f}")
    print()


def main():
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float32)
    frames = frames_as_dicts(N_FRAMES)
    print(f"{len(frames)} real validation frames from {VALID_SYSTEM.name}, "
          f"{frames[0]['positions'].shape[0]} atoms each\n")
    check_matrix_and_identity(frames)
    check_trained_model(frames, device)
    check_scale(frames, device)
    print("all charge_eq_latent checks passed")


if __name__ == "__main__":
    main()
