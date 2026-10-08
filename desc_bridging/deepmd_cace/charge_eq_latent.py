"""Latent-charge equalization for the DeepMD-CACE long-range stack.

``ChargeEqLatent`` sits exactly where ``cace.modules.ewald.EwaldPotential`` sits
in the long-range branch: it reads the latent charges ``q`` produced by the
charge head, rebuilds the Ewald Coulomb matrix ``A`` through CACE's own kernel,
and replaces those charges by the equilibrium solution of

    min_q  1/2 q^T A q + w ||q - q_r||^2     subject to  1^T q = Q_total

so the network's charge proposal ``q_r`` is relaxed towards an
electrostatically self-consistent set, with ``w`` (``regularization_weight``)
setting how far the relaxation may move away from the proposal. The long-range
energy handed to the loss is then ``0.5 q^T A q`` of the *equalized* charges,
under the same ``ewald_potential`` key, so every downstream stage
(``CombinePotential``, ``FeatureAdd``, ``Forces``) is unchanged.

Two limits bracket the behaviour:

* ``w -> infinity``: the trust region pins the charges, and the module reduces to
  the plain latent-charge Ewald term up to the one thing the constraint still
  imposes. The proposal carries a net charge, so the limit is ``q_r`` shifted
  uniformly by ``(sum q_r - Q) / (n_atoms * n_channels)`` - a shift independent of
  ``w``, which is why the limit is not ``q_r`` itself. ``check_charge_eq_latent.py``
  asserts against that shifted set rather than against ``q_r``.
* ``w -> 0``: the constraint alone survives and the charges relax to the
  minimum-Coulomb-energy distribution at fixed total charge. ``A`` is positive
  definite for a periodic cell once self-interaction is removed, so this limit is
  well posed for every ``w > 0``; ``w <= 0`` is rejected.

Relationship to ``cace.modules.charge_eq.ChargeEq``
---------------------------------------------------
That module is the classic Electronegativity Equalization solver: its charges come
from per-element ``chi``/``J`` parameters, no latent charge is involved, and it
carries a trainable ``J_raw`` hardness table. Here the network's latent charges
are the input and there are no per-element parameters at all, so the equalized
charges remain a function of the geometry alone. The Coulomb matrix is built the
same way in both: ``EwaldPotential`` evaluated on ``q = eye(N)`` with
``compute_field=True`` returns ``A_ij`` directly, the field at atom ``i`` due to a
unit charge at ``j``.

One place this module deliberately does not reproduce the kernel
-----------------------------------------------------------------
With ``remove_self_interaction=True`` and a charge head of ``n_out > 1``, the
kernel's own ``pot`` subtracts ``sum(q**2)`` over *all* channels from *every*
channel, so the self term is counted ``n_out`` times; the energy it reports is
then not the quadratic form of the field matrix it also reports (the gap is
exactly ``(n_out - 1) * sum_c sum_i q_ic^2 / (sigma (2 pi)^1.5)``, verified in
``check_charge_eq_latent.py``). This module reports ``0.5 sum_c q_c^T A q_c``
instead, which is the self-consistent quadratic form and the only quantity whose
gradient is the force it emits. The two agree exactly whenever
``remove_self_interaction`` is false, or ``n_out`` is 1.
"""

import torch
import torch.nn as nn

from cace.modules.ewald import EwaldPotential

__all__ = ["ChargeEqLatent", "coulomb_matrix", "equalize_charges"]


def coulomb_matrix(kernel, r, cell):
    """``A_ij``: the potential at ``i`` from a unit charge at ``j``, one frame.

    The kernel hands this back when it is called with ``q = eye(n)`` and
    ``compute_field=True``, but that materializes an ``[n, n, n_k]`` intermediate
    (and an ``[n, n, n]`` one in real space): twenty terabytes at the 1566-atom
    campaign slab. Both branches below are the same sums with the identity
    contracted out analytically, which costs ``O(n^2)`` memory instead. Nothing
    about the physics changes - ``check_charge_eq_latent.py`` pins every entry of
    this matrix against the kernel's own field output on a system small enough
    for that path to run.

    The reciprocal-space branch mirrors ``EwaldPotential.compute_potential_triclinic``
    line for line - same ``Nk`` from ``dl``, same ``k_sq`` cutoff, same
    hemisphere reduction with its factor of 2, same ``2 kfac`` field scaling, same
    self-interaction correction - and the real-space branch mirrors
    ``compute_potential_realspace``. Reading the grid off the kernel's own
    attributes (``dl``, ``sigma``, ``k_sq_max``, ``twopi``) keeps the two in step
    for everything except the summation itself.
    """
    n_atoms = r.shape[0]
    kernel_type = r.dtype
    cell_now = cell.reshape(3, 3)
    diagonal = cell_now.diagonal(dim1=-2, dim2=-1)
    if bool((diagonal.abs() < 1e-6).all()):
        difference = r.unsqueeze(0) - r.unsqueeze(1)
        distance = difference.norm(dim=-1)
        convergence = torch.special.erf(distance / kernel.sigma / (2.0 ** 0.5))
        A_mat = convergence / (distance + 1e-6) / kernel.twopi
        if not kernel.remove_self_interaction:
            A_mat = A_mat + torch.diag(
                torch.full((n_atoms,), 2.0 / (kernel.sigma * kernel.twopi ** 1.5),
                           device=r.device, dtype=kernel_type)
            )
        return A_mat
    if bool((diagonal > 0).all()):
        return _reciprocal_coulomb_matrix(kernel, r, cell_now)
    raise ValueError("Either all box dimensions must be positive or aperiodic box must be provided.")


def _reciprocal_coulomb_matrix(kernel, r, cell):
    """The reciprocal-space half of :func:`coulomb_matrix`, in closed form.

    ``A_ij = sum_m 2 factors_m kfac_m / V cos(k_m . (r_i - r_j))``, assembled as
    two ``[n, n_k] @ [n_k, n]`` products through the angle-addition identity rather
    than by ever forming the ``[n, n, n_k]`` tensor. The ``2`` is the kernel's own
    field scaling (``sk_field = 2 kfac conj(S_k)``); the energy it gives back is
    ``0.5 sum_c q_c^T A q_c``, which is exactly ``compute_potential_triclinic``'s
    ``pot`` for those charges.
    """
    # the kernel does this matmul in float32 unconditionally, which is what makes
    # its own triclinic path float32-only; casting the operands explicitly keeps
    # the float32 result bit-identical and lets float64 run at all
    reciprocal = 2.0 * torch.pi * torch.linalg.inv(cell).T
    n_grid = [max(1, int(norm.item() / kernel.dl)) for norm in torch.norm(cell, dim=1)]
    axes = [torch.arange(-n, n + 1, device=r.device) for n in n_grid]
    nvec = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    kvec = (nvec.to(torch.float32) @ reciprocal.to(torch.float32)).to(reciprocal.dtype)

    k_sq = kvec.pow(2).sum(dim=1)
    keep = (k_sq > 0) & (k_sq <= kernel.k_sq_max)
    kvec, k_sq, nvec = kvec[keep], k_sq[keep], nvec[keep]

    first_non_zero = torch.argmax((nvec != 0).to(torch.int), dim=1)
    sign = nvec.gather(1, first_non_zero.unsqueeze(1)).squeeze(1)
    hemisphere = (sign > 0) | (nvec == 0).all(dim=1)
    kvec, k_sq, nvec = kvec[hemisphere], k_sq[hemisphere], nvec[hemisphere]
    factors = torch.where((nvec == 0).all(dim=1), 1.0, 2.0)

    kfac = torch.exp(-kernel.sigma_sq_half * k_sq) / k_sq
    weight = 2.0 * factors * kfac / torch.det(cell)

    angles = r @ kvec.T
    cosine, sine = torch.cos(angles), torch.sin(angles)
    A_mat = (cosine * weight) @ cosine.T + (sine * weight) @ sine.T
    if kernel.remove_self_interaction:
        A_mat = A_mat - torch.diag(
            torch.full((r.shape[0],), 2.0 / (kernel.sigma * kernel.twopi ** 1.5),
                       device=r.device, dtype=r.dtype)
        )
    return A_mat


def equalize_charges(A_mat, q_ref, weight, total_charge=0.0):
    """The constrained equalization of one frame, by a KKT linear solve.

    ``A_mat`` is the ``[n, n]`` Coulomb matrix, ``q_ref`` the ``[n, k]`` latent
    charges it should stay near (``k`` = the charge head's ``n_out``), ``weight``
    the trust-region strength ``w`` and ``total_charge`` the value the summed
    charge ``sum_c 1^T q_c`` is held at. Returns ``(q_eq, multiplier)``.

    Stationarity of ``0.5 sum_c q_c^T A q_c + w sum_c ||q_c - q_r,c||^2`` under
    the single constraint ``sum_c 1^T q_c = Q`` is

        (A + 2w I) q_c = 2w q_r,c - multiplier * 1

    so one factorization of the shared ``A + 2w I`` serves every channel: the
    right-hand side is stacked as ``[2w q_ref | 1]`` and ``torch.linalg.solve``
    returns ``x_c`` and ``u = (A + 2w I)^{-1} 1`` together, leaving
    ``q_c = x_c - multiplier * u`` with the multiplier fixed by the constraint.
    Channels therefore stay independent except through that one shift, which is
    what makes the returned charges a function of the geometry and of the total
    charge alone - there is no per-element hardness parameter anywhere.

    ``A`` is positive definite once self-interaction is removed, so ``A + 2w I``
    is positive definite for every ``w > 0``; ``w <= 0`` is rejected rather than
    silently regularized. The solve is differentiable, so the long-range force is
    exact autograd rather than a finite difference.
    """
    if weight <= 0:
        raise ValueError(f"regularization_weight must be positive, got {weight}")
    n_atoms, n_channels = q_ref.shape
    device, dtype = A_mat.device, A_mat.dtype
    q_ref = q_ref.to(dtype=dtype)
    if not isinstance(total_charge, torch.Tensor):
        total_charge = torch.as_tensor(float(total_charge), device=device, dtype=dtype)
    total_charge = total_charge.reshape([]).to(device=device, dtype=dtype)

    # cat rather than writing into a preallocated buffer: an index assignment into
    # a tensor built from an autograd-connected block is an in-place op on a view,
    # and the graph it would have to keep is easier to state as concatenations.
    matrix = A_mat + 2.0 * weight * torch.eye(n_atoms, device=device, dtype=dtype)
    ones = torch.ones(n_atoms, 1, device=device, dtype=dtype)
    solution = torch.linalg.solve(matrix, torch.cat([2.0 * weight * q_ref, ones], dim=1))
    near_proposal, unit_shift = solution[:, :n_channels], solution[:, n_channels]
    multiplier = (near_proposal.sum() - total_charge) / (n_channels * unit_shift.sum())
    return near_proposal - multiplier * unit_shift.unsqueeze(1), multiplier


class ChargeEqLatent(nn.Module):
    """Drop-in replacement for ``EwaldPotential`` that equalizes latent charges.

    Kernel options (``dl``, ``sigma``, ``exponent``, ``remove_self_interaction``)
    are forwarded to a private ``EwaldPotential``, which is used only as a kernel:
    its ``forward`` is never called, so ``external_field`` and
    ``charge_neutral_lambda`` have no effect here and are rejected by the
    configuration validator rather than silently ignored.

    Emits ``output_key`` (the equalized charges, ``[n_atoms, n_out]``) and
    ``ewald_key`` (the long-range energy, one entry per frame under
    ``aggregation_mode="sum"``, matching ``EwaldPotential``'s shape for that
    mode). ``feature_key`` is read and left untouched, so the raw latent charges
    stay available to the trainer and to scoring.

    The module holds no parameters or buffers, so a plain long-range checkpoint
    loads into it and back without a state-dict change; only the energy the same
    charges produce changes.
    """

    def __init__(
        self,
        dl: float = 2.0,
        sigma: float = 1.0,
        exponent: int = 1,
        feature_key: str = "q",
        output_key: str = "q_eq",
        ewald_key: str = "ewald_potential",
        aggregation_mode: str = "sum",
        regularization_weight: float = 0.1,
        total_charge: float = 0.0,
        total_charge_key: str = "system_charge",
        remove_self_interaction: bool = True,
    ):
        super().__init__()
        self.feature_key = feature_key
        self.output_key = output_key
        self.ewald_key = ewald_key
        self.aggregation_mode = aggregation_mode
        self.regularization_weight = float(regularization_weight)
        self.total_charge = float(total_charge)
        self.total_charge_key = total_charge_key
        self.exponent = exponent
        self.model_outputs = [output_key, ewald_key]
        self.required_derivatives = ["cell"]

        if exponent != 1:
            raise ValueError("charge_eq_latent implements the electrostatic kernel, exponent must be 1")
        if self.regularization_weight <= 0:
            raise ValueError("regularization_weight must be positive")

        # compute_field=False here on purpose: the parent NeuralNetworkPotential
        # collects model_outputs from every submodule, and compute_field=True would
        # add '<feature>_field' to that list and make extract_outputs raise for a
        # key this module never writes. The field is requested per call instead.
        self.ep = EwaldPotential(
            dl=dl,
            sigma=sigma,
            feature_key=feature_key,
            output_key=ewald_key,
            aggregation_mode=aggregation_mode,
            remove_self_interaction=remove_self_interaction,
            compute_field=False,
        )

    def _coulomb_matrix(self, r, cell):
        """``A`` for one frame, from the kernel held on this module."""
        return coulomb_matrix(self.ep, r, cell)

    def _frame_totals(self, data, n_frames):
        """The constrained total charge of each frame, from the batch or a scalar."""
        if self.total_charge_key in data and data[self.total_charge_key] is not None:
            values = data[self.total_charge_key]
            if torch.is_tensor(values):
                return values.to(data["positions"].device).reshape(-1)
            return torch.full((n_frames,), float(values), device=data["positions"].device)
        return torch.full((n_frames,), self.total_charge, device=data["positions"].device)

    def forward(self, data, **kwargs):
        if data.get("batch") is None:
            batch_now = torch.zeros(
                data["positions"].shape[0], dtype=torch.int64, device=data["positions"].device
            )
        else:
            batch_now = data["batch"]

        # one cell per frame, shaped as the kernel reads it: a single-frame batch
        # arrives with a bare (3, 3) cell, a multi-frame one with (n_frames, 3, 3)
        cells = data["cell"].view(-1, 3, 3)
        r = data["positions"]
        q_ref = data[self.feature_key]
        if q_ref.dim() == 1:
            q_ref = q_ref.unsqueeze(1)
        assert r.shape[1] == 3, "r dimension error"
        assert r.shape[0] == q_ref.shape[0], "q dimension error"

        frames = torch.unique(batch_now)
        totals = self._frame_totals(data, len(frames))

        equalized, energies = [], []
        for position, frame in enumerate(frames):
            mask = batch_now == frame
            A_mat = self._coulomb_matrix(r[mask], cells[frame])
            q_eq, _ = equalize_charges(
                A_mat, q_ref[mask], self.regularization_weight, totals[position]
            )
            equalized.append(q_eq)
            # 0.5 sum_c q_c^T A q_c, which is exactly what the kernel's own
            # reciprocal- and real-space functions report for these charges; shaped
            # [1] like one EwaldPotential frame result so the aggregation below is
            # the kernel's own and the two modules stay shape-compatible
            energies.append((0.5 * (q_eq * (A_mat @ q_eq)).sum()).reshape(1))

        data[self.output_key] = torch.cat(equalized, dim=0)
        stacked = torch.stack(energies, dim=0)
        data[self.ewald_key] = stacked.sum(axis=1) if self.aggregation_mode == "sum" else stacked
        return data
