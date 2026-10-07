import pytest
import torch
import logging

from les import Les
from les.module.charge_eq import project_zero_mean
from les.module.ewald import Ewald


def test_charge_eq_debug_logs_reference_and_projected_charge_stats(caplog):
    dtype = torch.float64
    model = Les(
        {
            "dim_descrpt": 4,
            "local_charge": True,
            "charge_eq": True,
            "charge_eq_debug": True,
            "log_freq": 1,
            "n_hidden": [8],
            "n_layers": 2,
        }
    ).to(dtype=dtype)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [2.0, 0.0, 0.0], [2.9, 0.0, 0.0]],
        dtype=dtype,
    )
    descriptor = torch.randn(4, 4, dtype=dtype)
    atomic_numbers = torch.tensor([8, 8, 1, 1])

    with caplog.at_level(logging.DEBUG, logger="les.les"):
        model(positions, desc=descriptor, atomic_numbers=atomic_numbers)

    messages = caplog.text
    assert "DEBUG    les.les:les.py" in messages
    assert "charge_eq_debug q_r by element" in messages
    assert "charge_eq_debug q by element" in messages
    assert "q_r\tZ=8\tmean=" in messages
    assert "q\tZ=1\tmean=" in messages
    assert "dtype=" not in messages
    assert "device=" not in messages


def test_verbose_charge_stats_log_per_element_variance(caplog):
    dtype = torch.float64
    model = Les(
        {
            "dim_descrpt": 4,
            "local_charge": True,
            "verbose": True,
            "log_freq": 1,
            "n_hidden": [8],
            "n_layers": 2,
        }
    ).to(dtype=dtype)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [2.0, 0.0, 0.0], [2.9, 0.0, 0.0]],
        dtype=dtype,
    )
    descriptor = torch.randn(4, 4, dtype=dtype)
    atomic_numbers = torch.tensor([8, 8, 1, 1])

    with caplog.at_level(logging.INFO, logger="les.les"):
        model(positions, desc=descriptor, atomic_numbers=atomic_numbers)

    assert "latent_charges by element: Z\tmean\tvariance" in caplog.text
    assert "Z=8\tmean=" in caplog.text
    assert "Z=1\tmean=" in caplog.text
    assert "latent_charges mean:" not in caplog.text
    assert "std:" not in caplog.text
    assert "grad_fn" not in caplog.text
    assert "dtype=" not in caplog.text
    assert "device=" not in caplog.text


def test_parameter_and_graph_summaries_are_independently_debug_gated(caplog):
    dtype = torch.float64
    model = Les(
        {
            "dim_descrpt": 4,
            "local_charge": True,
            "charge_eq": True,
            "q_nn_debug": True,
            "e_lr_graph_debug": True,
            "log_freq": 1,
            "n_hidden": [8],
            "n_layers": 2,
        }
    ).to(dtype=dtype)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=dtype
    )
    descriptor = torch.randn(2, 4, dtype=dtype)

    with caplog.at_level(logging.DEBUG, logger="les.les"):
        model(positions, desc=descriptor)

    assert "Q NN params (norm summary)" in caplog.text
    assert "E_lr graph:" in caplog.text
    assert "CatBackward" in caplog.text


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("remove_self_interaction", [False, True])
def test_coulomb_matrix_matches_ewald_energy(periodic, remove_self_interaction):
    dtype = torch.float64
    positions = torch.tensor(
        [[0.2, 0.3, 0.4], [1.1, 0.5, 0.7], [0.6, 1.2, 0.9]], dtype=dtype
    )
    charges = torch.tensor([0.3, -0.1, -0.2], dtype=dtype)
    cell = (
        torch.tensor(
            [[5.0, 0.3, 0.1], [0.0, 4.8, 0.2], [0.1, 0.0, 5.2]], dtype=dtype
        )
        if periodic
        else None
    )
    ewald = Ewald(
        sigma=0.8, dl=1.5, remove_self_interaction=remove_self_interaction
    )

    matrix = ewald.compute_coulomb_matrix(positions, cell)
    if periodic:
        energy = ewald.compute_potential_triclinic(positions, charges, cell)["pot"]
    else:
        energy = ewald.compute_potential_realspace(positions, charges)["pot"]

    matrix_energy = 0.5 * charges @ matrix @ charges
    torch.testing.assert_close(matrix_energy.reshape_as(energy), energy)


def test_charge_equilibrium_projects_and_backpropagates():
    dtype = torch.float64
    positions = torch.tensor(
        [[0.2, 0.3, 0.4], [1.1, 0.5, 0.7], [0.6, 1.2, 0.9]], dtype=dtype
    )
    descriptor = torch.randn(3, 4, dtype=dtype, requires_grad=True)
    cell = torch.eye(3, dtype=dtype).unsqueeze(0) * 5.0
    model = Les(
        {
            "dim_descrpt": 4,
            "local_charge": True,
            "charge_eq": True,
            "regularization_weight": 0.1,
            "n_hidden": [8],
            "n_layers": 2,
        }
    ).to(dtype=dtype)

    output = model(positions, cell=cell, desc=descriptor)
    torch.testing.assert_close(
        output["latent_charges"].sum(), torch.tensor(0.0, dtype=dtype), atol=1e-10, rtol=0
    )
    expected_energy = 0.5 * torch.dot(
        output["latent_charges"].reshape(-1),
        torch.mv(
            model.ewald.compute_coulomb_matrix(positions, cell[0]),
            output["latent_charges"].reshape(-1),
        ),
    )
    torch.testing.assert_close(output["E_lr"].reshape(()), expected_energy)

    output["E_lr"].sum().backward()
    assert descriptor.grad is not None
    assert torch.isfinite(descriptor.grad).all()


def test_charge_equilibrium_supports_nonzero_total_charge():
    q_r = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    matrix = torch.eye(3, dtype=torch.float64)

    charges = project_zero_mean(q_r, matrix, 0.1, total_charge=1.0)

    torch.testing.assert_close(charges.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_charge_equilibrium_handles_multiple_frames_independently():
    dtype = torch.float64
    positions = torch.tensor(
        [
            [0.2, 0.3, 0.4],
            [1.1, 0.5, 0.7],
            [0.6, 1.2, 0.9],
            [1.4, 0.8, 1.1],
        ],
        dtype=dtype,
    )
    descriptor = torch.randn(4, 4, dtype=dtype, requires_grad=True)
    batch = torch.tensor([0, 0, 1, 1])
    cell = torch.stack([torch.eye(3, dtype=dtype) * 5.0,
                        torch.eye(3, dtype=dtype) * 5.5])
    model = Les(
        {
            "dim_descrpt": 4,
            "local_charge": True,
            "charge_eq": True,
            "regularization_weight": 0.1,
            "n_hidden": [8],
            "n_layers": 2,
        }
    ).to(dtype=dtype)

    output = model(positions, cell=cell, desc=descriptor, batch=batch)

    torch.testing.assert_close(
        output["latent_charges"][:2].sum(), torch.tensor(0.0, dtype=dtype), atol=1e-10, rtol=0
    )
    torch.testing.assert_close(
        output["latent_charges"][2:].sum(), torch.tensor(0.0, dtype=dtype), atol=1e-10, rtol=0
    )
    assert output["E_lr"].shape == (2,)
    output["E_lr"].sum().backward()
    assert descriptor.grad is not None
    assert torch.isfinite(descriptor.grad).all()


# ---------------------------------------------------------------------------
# Regression: the k-space contraction in compute_coulomb_matrix.
#
# The matrix used to be assembled by forming the phase difference explicitly as
# an [n, n, M] tensor and taking cos of it. That is the definition, and it is
# correct, but it is also unusable at the size the model actually sees: the water
# interface slab is 1566 atoms with M = 11146 k-vectors, so one such tensor is
# 218 GB and the old path materialised two of them - 407 GiB of float64. It fits
# no single accelerator, and on a 15 GB host it already peaked at 14 GB by n = 300
# and failed outright at n = 600. The tests above all use 3-4 atoms, which is why
# none of them noticed.
#
# The fix contracts k inside two GEMMs, using
#   cos(k.(r_i - r_j)) = cos(k.r_i)cos(k.r_j) + sin(k.r_i)sin(k.r_j)
# so A = (cos*p) @ cos^T + (sin*p) @ sin^T with O(n*M) intermediates. The two
# tests below pin both halves: that the contracted matrix is still the matrix the
# definition gives (at real M, which is what makes it a test of this change
# rather than of small-k arithmetic), and that it is computable at production n.
# ---------------------------------------------------------------------------

# The water-interface slab: the cell the campaign trains on. M depends only on
# the cell and dl, never on n or on the positions, so this reproduces the
# production k-space exactly at any size.
PRODUCTION_CELL = torch.diag(torch.tensor([25.6, 25.6, 65.0], dtype=torch.float64))


def _reference_coulomb_matrix(r, cell, sigma, dl, norm_factor, remove_self_interaction):
    """The [n, n, M] double-sum definition, kept as the oracle for the contraction.

    Mirrors compute_coulomb_matrix's k-space construction step for step, and
    differs only in the last two lines. Returns (matrix, M).
    """
    dtype = r.dtype
    n = r.shape[0]
    volume = torch.linalg.det(cell)
    reciprocal = 2.0 * torch.pi * torch.linalg.inv(cell).T
    norms = torch.linalg.norm(cell, dim=1)
    nk = [max(1, int(norms[axis].item() / dl)) for axis in range(3)]
    axes = [torch.arange(-k, k + 1) for k in nk]
    nvec = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    nvec = nvec.to(reciprocal.dtype)
    kvec = nvec @ reciprocal
    k_sq = torch.sum(kvec**2, dim=1)
    keep = (k_sq > 0) & (k_sq <= (2.0 * torch.pi / dl) ** 2)
    nvec, kvec, k_sq = nvec[keep], kvec[keep], k_sq[keep]

    first_non_zero = torch.argmax((nvec != 0).to(torch.int), dim=1)
    sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze(1)
    hemisphere = (sign > 0) | (nvec == 0).all(dim=1)
    nvec, kvec, k_sq = nvec[hemisphere], kvec[hemisphere], k_sq[hemisphere]
    factors = torch.where((nvec == 0).all(dim=1), 1.0, 2.0)
    prefactor = (
        factors * 2.0 * torch.exp(-(sigma**2 / 2.0) * k_sq) / k_sq / volume * norm_factor
    )

    phase = r @ kvec.T
    phase_difference = phase[:, None, :] - phase[None, :, :]
    matrix = torch.einsum("m,ijm->ij", prefactor, torch.cos(phase_difference))

    if remove_self_interaction:
        diagonal = 2.0 * norm_factor / (sigma * (2.0 * torch.pi) ** 1.5)
        matrix = matrix - torch.eye(n, dtype=dtype) * diagonal
    return matrix, int(k_sq.shape[0])


def _production_regime_positions(n, dtype):
    """n points in the production cell. Seeded, so the comparison is deterministic."""
    torch.manual_seed(0)
    return (
        torch.rand(n, 3, dtype=torch.float64)
        * torch.tensor([25.0, 25.0, 64.0], dtype=torch.float64)
        + torch.tensor([0.3, 0.3, 0.5], dtype=torch.float64)
    ).to(dtype)


@pytest.mark.parametrize(
    "dtype,tolerance",
    [
        # Measured on this cell: entry-wise 1.6e-14 (float64) and 3.7e-06
        # (float32) relative to max|A| = 7.50 / 3.56. Both are ~3-5x headroom
        # over the measured figure, which is tight enough to fail a real
        # algebraic error and loose enough not to fail on a BLAS reordering.
        (torch.float64, 5e-14),
        (torch.float32, 2e-05),
    ],
)
def test_coulomb_matrix_contraction_matches_definition(dtype, tolerance):
    """The contracted matrix equals the [n, n, M] definition at production M."""
    ewald = Ewald(sigma=1.0, dl=2.0, remove_self_interaction=True)
    r = _production_regime_positions(32, dtype)
    cell = PRODUCTION_CELL.to(dtype)

    reference, n_k = _reference_coulomb_matrix(
        r, cell, ewald.sigma, ewald.dl, ewald.norm_factor, ewald.remove_self_interaction
    )
    # Assert the regime, not the number: if the cell or dl ever moves, the oracle
    # above quietly becomes a small-k test and stops covering the bug it exists for.
    assert n_k > 10000, f"expected the production k-space (M=11146), measured M={n_k}"

    matrix = ewald.compute_coulomb_matrix(r, cell)
    assert matrix.dtype == dtype

    torch.testing.assert_close(
        matrix, reference, rtol=0, atol=tolerance * reference.abs().max().item()
    )

    # The matrix is only a means to an end, so also check the end: 0.5 q^T A q
    # must be the Ewald energy the rest of the pipeline computes directly.
    torch.manual_seed(0)
    charges = torch.randn(32, dtype=torch.float64).to(dtype) * 0.3
    energy = ewald.compute_potential_triclinic(r, charges, cell)["pot"].reshape(())
    torch.testing.assert_close(
        0.5 * charges @ matrix @ charges, energy, rtol=0, atol=tolerance * abs(energy.item())
    )


def test_coulomb_matrix_at_production_size():
    """1566 atoms, M = 11146: the size the pre-contraction code could not run.

    This is a memory regression, not an accuracy test. It asserts only what the
    old implementation could never reach - that the matrix exists, is finite, is
    symmetric, and carries the right energy - because the two 218 GB intermediates
    made all four of those untestable. Peak RSS for this test alone: 1106 MiB.
    """
    ewald = Ewald(sigma=1.0, dl=2.0, remove_self_interaction=True)
    n = 1566
    r = _production_regime_positions(n, torch.float64)

    matrix = ewald.compute_coulomb_matrix(r, PRODUCTION_CELL)

    assert matrix.shape == (n, n)
    assert torch.isfinite(matrix).all()
    # Measured here: max|A - A^T| = 4.5e-15 * max|A| (10.73), and the energy below
    # lands 1.6e-15 from the Ewald value. 1e-12 is ~200-600x headroom over both,
    # enough for a BLAS that accumulates in another order and far too tight for an
    # algebraic error, which shows up at O(1) - the cos-only mutant scores 6.2.
    scale = matrix.abs().max().item()
    torch.testing.assert_close(matrix, matrix.T, rtol=0, atol=1e-12 * scale)

    torch.manual_seed(0)
    charges = torch.randn(n, dtype=torch.float64) * 0.3
    energy = ewald.compute_potential_triclinic(r, charges, PRODUCTION_CELL)["pot"].reshape(())
    torch.testing.assert_close(
        0.5 * charges @ matrix @ charges, energy, rtol=0, atol=1e-12 * abs(energy.item())
    )