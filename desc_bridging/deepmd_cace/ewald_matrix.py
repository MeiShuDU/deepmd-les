"""CACE Ewald adapter exposing a dense Coulomb operator for constrained solves.

The upstream CACE kernel exposes potential actions but currently has no matrix
API. Keep its compatible dense contraction here, separate from charge solving.
"""

import torch

from cace.modules.ewald import EwaldPotential


class EwaldMatrixPotential(EwaldPotential):
    """Use the upstream matrix API when present, otherwise contract its kernel."""

    def compute_coulomb_matrix(self, r, cell):
        upstream = getattr(super(), "compute_coulomb_matrix", None)
        if upstream is not None:
            return upstream(r, cell)
        return _assemble_coulomb_matrix(self, r, cell)


def coulomb_matrix(kernel, r, cell):
    """Obtain a kernel's matrix, retaining support for plain CACE kernels."""
    method = getattr(kernel, "compute_coulomb_matrix", None)
    if method is not None:
        return method(r, cell)
    return _assemble_coulomb_matrix(kernel, r, cell)


def _assemble_coulomb_matrix(kernel, r, cell):
    """``A_ij``: the potential at ``i`` from a unit charge at ``j``, one frame.

    The KKT solve needs the operator explicitly for the dense first version.
    The matrix is assembled once per frame from the Ewald kernel's reciprocal or
    direct field expression, then reused for the constraints, solve, and energy.
    It is not recomputed by a second energy path. Calling the kernel with an
    identity charge matrix would create unnecessarily large intermediate tensors,
    so the same kernel sums are contracted directly into an ``O(n^2)`` matrix.
    ``check_charge_eq_latent.py`` compares this result entry by entry with the
    kernel's own field output on a small system.

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
    """The reciprocal-space half of :func:`_assemble_coulomb_matrix`, in closed form.

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
