import torch


def project_zero_mean(
    q_r: torch.Tensor,
    A: torch.Tensor,
    w: float,
    total_charge: float = 0.0,
) -> torch.Tensor:
    """
    Solve for:
        min_q 1/2 q^T A q + w (q-q_r)^T (q-q_r)
        s.t. 1^T q = total_charge
    return the optimal q.

    This algorithm is moltivated by the following idea:

    A is the Hessian of the charge-only Ewald energy used by LES, including its
    Gaussian smearing, reciprocal-space cutoff, and self-interaction convention.
    Thus the energy is exactly E(q) = 1/2 q^T A q for the same Ewald settings.

    q_r comes from the LR NN. The constraint 1^T q = total_charge is imposed
    exactly; the default total_charge is zero.
    
    A regularization term w (q-q_r)^T (q-q_r) is added to the optimization problem to avoid a deterministic solution.
    w, a parameter presenting the weight of the NN prediction. Higher w means more trust in the NN prediction, while lower w means more trust in the Coulombic energy.

    """
    if w <= 0.0:
        raise ValueError("regularization weight must be positive")

    original_shape = q_r.shape
    q_r = q_r.reshape(-1)
    n = q_r.numel()
    dtype = q_r.dtype
    device = q_r.device

    A = A.to(dtype=dtype, device=device)
    w = torch.as_tensor(w, dtype=dtype, device=device)

    # A is symmetric
    I = torch.eye(n, dtype=dtype, device=device)
    M = A + 2.0 * w * I

    # KKT matrix: 
    # [ M      1 ] [q]   [2w q_r]
    # [ 1^T    0 ] [λ] = [0     ]
    K = torch.zeros(n + 1, n + 1, dtype=dtype, device=device)
    K[:n, :n] = M
    K[:n, n] = 1.0
    K[n, :n] = 1.0

    rhs = torch.zeros(n + 1, dtype=dtype, device=device)
    rhs[:n] = 2.0 * w * q_r
    rhs[n] = total_charge

    # 若 K 可能奇异，可改用 torch.linalg.lstsq
    sol = torch.linalg.solve(K, rhs)

    return sol[:n].reshape(original_shape)