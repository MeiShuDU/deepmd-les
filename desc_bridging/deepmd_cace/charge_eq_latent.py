"""Latent-charge equalization for the DeepMD-CACE long-range stack.

``ChargeEqLatent`` sits exactly where ``cace.modules.ewald.EwaldPotential`` sits
in the long-range branch: it reads the latent charges ``q`` produced by the
charge head, obtains the Ewald Coulomb operator ``A`` using the same kernel
convention, and replaces those charges by the equilibrium solution of

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
* ``w -> 0``: a minimum exists only if the Coulomb objective has positive
  curvature along feasible charge directions. Removing the self term can make
  that objective indefinite. The solver checks curvature rather than assuming
  that nonnegative hardness guarantees convexity.

Relationship to ``cace.modules.charge_eq.ChargeEq``
---------------------------------------------------
That module is the classic Electronegativity Equalization solver: its charges come
from per-element ``chi``/``J`` parameters, no latent charge is involved, and it
carries a trainable ``J_raw`` hardness table. Here the network's latent charges
are the input. Optional inverse hardness is predicted from local invariant
features, so the equalized charges remain functions of geometry and boundary
conditions. The Ewald adapter supplies ``compute_coulomb_matrix`` so the solver
can reuse the operator for constraints, equalization, and energy.
Its compatibility contraction agrees with ``EwaldPotential`` evaluated on
``q = eye(N)`` with ``compute_field=True``, without that call's large intermediate.

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

from .ewald_matrix import EwaldMatrixPotential, coulomb_matrix

__all__ = ["ChargeEqLatent", "coulomb_matrix", "equalize_charges"]


def equalize_charges(A_mat, q_ref, weight, total_charge=0.0, constraint_matrix=None,
                     constraint_values=None, kappa_inv=None):
    """Minimize the quadratic with ``C q = values``, returning charges and duals.

    Fixed hardness is ``2*weight`` for compatibility with the original penalty.
    For multiple legacy channels, constraints act on their collective charge.
    Validation is detached; the solve, including geometry-dependent constraint
    rows and hardness, remains differentiable through force-loss gradients.
    """
    if kappa_inv is None and (not torch.isfinite(torch.tensor(weight)) or weight < 0):
        raise ValueError(f"regularization_weight must be finite and nonnegative, got {weight}")
    n_atoms, n_channels = q_ref.shape
    device, dtype = A_mat.device, A_mat.dtype
    q_ref = q_ref.to(dtype=dtype)
    if not isinstance(total_charge, torch.Tensor):
        total_charge = torch.as_tensor(float(total_charge), device=device, dtype=dtype)
    total_charge = total_charge.reshape([]).to(device=device, dtype=dtype)

    hardness = (torch.full((n_atoms,), 2.0 * weight, device=device, dtype=dtype)
                if kappa_inv is None else kappa_inv.to(device=device, dtype=dtype).reshape(n_atoms))
    if not torch.isfinite(hardness).all() or (hardness < 0).any():
        raise ValueError("kappa_inv must be finite and nonnegative")
    # The quadratic objective is defined by the symmetric part of A.
    A_mat = (A_mat + A_mat.T) * 0.5
    matrix = A_mat + torch.diag(hardness)
    if constraint_matrix is None:
        constraint_matrix = torch.ones(1, n_atoms, device=device, dtype=dtype)
        constraint_values = total_charge.reshape(1)
    else:
        constraint_matrix = constraint_matrix.to(device=device, dtype=dtype)
        constraint_values = torch.as_tensor(constraint_values, device=device, dtype=dtype).reshape(-1)
        if constraint_matrix.ndim != 2 or constraint_matrix.shape[1] != n_atoms:
            raise ValueError("constraint_matrix must have shape [n_constraints, n_atoms]")
        if constraint_matrix.shape[0] != constraint_values.numel():
            raise ValueError("constraint_values must match constraint_matrix rows")
        if constraint_matrix.shape[0] == 0:
            raise ValueError("at least one active charge constraint is required")
    if not all(torch.isfinite(x).all() for x in (matrix, q_ref, constraint_matrix, constraint_values)):
        raise ValueError("charge matrix, proposal, constraints and targets must be finite")
    # Normalize rows to avoid potential and charge units skewing the KKT solve.
    row_norm = constraint_matrix.norm(dim=1)
    if (row_norm == 0).any():
        raise ValueError("active charge constraints are linearly dependent (zero row)")
    constraint_matrix = constraint_matrix / row_norm[:, None]
    if torch.linalg.matrix_rank(constraint_matrix.detach()) != constraint_matrix.shape[0]:
        raise ValueError("active charge constraints are linearly dependent or ill-conditioned")
    values = constraint_values / row_norm
    m = constraint_matrix.shape[0]
    with torch.no_grad():
        _, info = torch.linalg.cholesky_ex(matrix)
    if int(info) == 0:
        # One n-by-n factorization serves all channels and constraint rows.
        result = torch.linalg.solve(matrix, torch.cat([
            hardness[:, None] * q_ref, constraint_matrix.T], dim=1))
        unconstrained, response = result[:, :n_channels], result[:, n_channels:]
        dual = torch.linalg.solve(n_channels * constraint_matrix @ response,
                                 constraint_matrix @ unconstrained.sum(dim=1) - values)
        q_eq = unconstrained - (response @ dual)[:, None]
        return q_eq, dual / row_norm
    # A full saddle-point solve also supports a singular H whose constrained
    # subspace has a unique minimum (including the zero-hardness limit).
    h = torch.block_diag(*([matrix] * n_channels))
    c = constraint_matrix.repeat(1, n_channels)
    m = c.shape[0]
    kkt = torch.cat([torch.cat([h, c.T], dim=1),
                     torch.cat([c, h.new_zeros(m, m)], dim=1)], dim=0)
    # Nonnegative hardness alone does not ensure convexity of a self-subtracted
    # Coulomb kernel. Check curvature only along feasible charge directions.
    with torch.no_grad():
        _, _, vh = torch.linalg.svd(c, full_matrices=True)
        null = vh[m:].T
        if null.shape[1]:
            projected = null.T @ h @ null
            if torch.linalg.eigvalsh(projected).min() <= 0:
                raise ValueError("charge objective is not strictly convex on the feasible subspace")
    rhs = torch.cat([(hardness[:, None] * q_ref).T.reshape(-1), values])
    solution = torch.linalg.solve(kkt, rhs)
    return solution[:n_atoms * n_channels].reshape(n_channels, n_atoms).T, solution[n_atoms * n_channels:] / row_norm


class ChargeEqLatent(nn.Module):
    """Drop-in replacement for ``EwaldPotential`` that equalizes latent charges.

    Kernel options (``dl``, ``sigma``, ``exponent``, ``remove_self_interaction``)
    are forwarded to a private ``EwaldMatrixPotential`` adapter:
    its ``forward`` is never called, so ``external_field`` and
    ``charge_neutral_lambda`` have no effect here and are rejected by the
    configuration validator rather than silently ignored.

    Emits ``output_key`` (the equalized charges, ``[n_atoms, n_out]``) and
    ``ewald_key`` (the long-range energy, one entry per frame under
    ``aggregation_mode="sum"``, matching ``EwaldPotential``'s shape for that
    mode). ``feature_key`` is read and left untouched, so the raw latent charges
    stay available to the trainer and to scoring.

    Fixed hardness adds no parameters. Learned hardness uses a separate local
    feature head, saved with the model state dictionary.
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
        constraints_key: str = "charge_eq_constraints",
        systems=None,
        learnable_kappa=False,
        kappa_in=None,
        slab_axis=None,
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
        self.constraints_key = constraints_key
        self.systems = systems or []
        self.slab_axis = slab_axis
        self.kappa_head = None
        if learnable_kappa:
            if kappa_in is None or regularization_weight <= 0:
                raise ValueError("learnable_kappa requires descriptor feature width and positive initial regularization_weight")
            self.kappa_head = nn.Sequential(nn.Linear(kappa_in, 16), nn.SiLU(), nn.Linear(16, 1))
            nn.init.zeros_(self.kappa_head[-1].weight)
            initial = torch.tensor(2.0 * regularization_weight, dtype=torch.float64)
            nn.init.constant_(self.kappa_head[-1].bias,
                              float(initial + torch.log(-torch.expm1(-initial))))
        self.exponent = exponent
        self.model_outputs = [output_key, ewald_key]
        if self.kappa_head is not None:
            self.model_outputs.append("kappa_inv")
        self.required_derivatives = ["cell"]

        if exponent != 1:
            raise ValueError("charge_eq_latent implements the electrostatic kernel, exponent must be 1")
        if not torch.isfinite(torch.tensor(self.regularization_weight)) or self.regularization_weight < 0:
            raise ValueError("regularization_weight must be finite and nonnegative")
        if slab_axis is not None and (isinstance(slab_axis, bool) or slab_axis not in (0, 1, 2)):
            raise ValueError("slab_axis must be 0, 1, 2 or null")

        # compute_field=False here on purpose: the parent NeuralNetworkPotential
        # collects model_outputs from every submodule, and compute_field=True would
        # add '<feature>_field' to that list and make extract_outputs raise for a
        # key this module never writes. The matrix API supplies the operator.
        self.ep = EwaldMatrixPotential(
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
        matrix = self.ep.compute_coulomb_matrix(r, cell)
        if self.slab_axis is not None:
            cell = cell.reshape(3, 3)
            volume = torch.linalg.det(cell).abs()
            if not torch.isfinite(volume) or volume <= 0:
                raise ValueError("slab correction requires a finite periodic cell")
            axis = self.slab_axis
            others = [i for i in range(3) if i != axis]
            if (cell[axis, others].abs() > 1e-10).any() or (cell[others, axis].abs() > 1e-10).any():
                raise ValueError("slab_axis must be aligned with a Cartesian cell axis")
            # CACE's 1/(2*pi*r) prefactor rescales 4*pi/V to 2/V.
            z = r[:, axis]
            matrix = matrix + (2.0 / volume) * torch.outer(z, z)
        return matrix

    def _frame_totals(self, data, n_frames):
        """The constrained total charge of each frame, from the batch or a scalar."""
        if self.total_charge_key in data and data[self.total_charge_key] is not None:
            values = data[self.total_charge_key]
            if torch.is_tensor(values):
                values = values.to(data["positions"].device).reshape(-1)
                if values.numel() != n_frames:
                    raise ValueError("total_charge_key requires one value per frame")
                return values
            return torch.full((n_frames,), float(values), device=data["positions"].device)
        return torch.full((n_frames,), self.total_charge, device=data["positions"].device)

    def _frame_constraints(self, data, frame, mask, q_ref, A_mat, total):
        spec = data.get(self.constraints_key)
        if self.systems:
            ids = data.get("charge_eq_system")
            if ids is None:
                raise ValueError("per-system equalization requires charge_eq_system metadata")
            system_id = int(ids.reshape(-1)[frame])
            if not 0 <= system_id < len(self.systems):
                raise ValueError("charge_eq_system is outside the configured systems")
            spec = self.systems[system_id]
            if int(mask.sum()) != spec["n_atoms"]:
                raise ValueError("atom count differs from fixed-group system definition")
        if spec is None:
            return None, total
        if isinstance(spec, (list, tuple)):
            spec = spec[frame]
        if not isinstance(spec, dict):
            raise ValueError(f"{self.constraints_key} entries must be mappings")
        rows, values = [], []
        n_atoms = int(mask.sum())
        for item in spec.get("constraints", []):
            kind = str(item.get("type", "")).upper()
            group = item.get("indices")
            if group is None:
                raise ValueError("constraint entries require zero-based 'indices'")
            indices = torch.as_tensor(group, device=mask.device, dtype=torch.long)
            if indices.numel() == 0 or int(indices.min()) < 0 or int(indices.max()) >= n_atoms:
                raise ValueError("constraint indices are out of range")
            row = torch.zeros(n_atoms, device=A_mat.device, dtype=A_mat.dtype)
            if kind == "CONQ":
                row.scatter_(0, indices, 1.0)
                value = item.get("total_charge")
            elif kind == "CONP":
                group2 = item.get("indices2")
                if group2 is None:
                    raise ValueError("CONP requires 'indices2'")
                indices2 = torch.as_tensor(group2, device=mask.device, dtype=torch.long)
                if indices2.numel() == 0 or int(indices2.min()) < 0 or int(indices2.max()) >= n_atoms:
                    raise ValueError("CONP indices2 are out of range")
                row.scatter_add_(0, indices, torch.full((indices.numel(),), 1.0 / indices.numel(), device=A_mat.device, dtype=A_mat.dtype))
                row.scatter_add_(0, indices2, torch.full((indices2.numel(),), -1.0 / indices2.numel(), device=A_mat.device, dtype=A_mat.dtype))
                row = row @ A_mat
                value = item.get("delta_V")
            else:
                raise ValueError(f"unsupported charge constraint type: {kind}")
            if value is None or not torch.isfinite(torch.as_tensor(value)):
                raise ValueError(f"missing or non-finite target for {kind}")
            rows.append(row)
            values.append(float(value))
        if not rows:
            return None, total
        return torch.stack(rows), torch.as_tensor(values, device=A_mat.device, dtype=A_mat.dtype)

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
        if (self.systems or self.kappa_head is not None) and q_ref.shape[1] != 1:
            raise ValueError("fixed-group and learnable-kappa modes require one charge channel")
        hardness = None
        if self.kappa_head is not None:
            if data.get("node_feats") is None:
                raise ValueError("learnable_kappa requires node_feats")
            features = data["node_feats"].reshape(r.shape[0], -1).to(self.kappa_head[0].weight.dtype)
            hardness = torch.nn.functional.softplus(self.kappa_head(features)).reshape(-1)
            data["kappa_inv"] = hardness

        frames = torch.unique(batch_now)
        totals = self._frame_totals(data, len(frames))

        equalized, energies = [], []
        for position, frame in enumerate(frames):
            mask = batch_now == frame
            A_mat = self._coulomb_matrix(r[mask], cells[frame])
            constraint_matrix, constraint_values = self._frame_constraints(
                data, int(frame), mask, q_ref[mask], A_mat, totals[position]
            )
            if self.slab_axis is not None:
                c = constraint_matrix
                target = constraint_values
                neutral = (float(totals[position]) == 0.0 if c is None else bool(
                    ((c == 1).all(dim=1) & (target == 0)).any()))
                if not neutral:
                    raise ValueError("slab correction requires explicit global neutrality")
            try:
                q_eq, _ = equalize_charges(
                    A_mat, q_ref[mask], self.regularization_weight, totals[position],
                    constraint_matrix, constraint_values,
                    None if hardness is None else hardness[mask],
                )
            except ValueError as error:
                raise ValueError(f"charge_eq_latent frame {int(frame)}: {error}") from error
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
