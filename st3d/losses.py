"""
Loss functions for st3d.

Provides a pure-PyTorch Sinkhorn distance, nearest-neighbor MSE for
expression matching, and a combined loss with curriculum weight scheduling.
"""

import math

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Sinkhorn distance (pure PyTorch)
# ---------------------------------------------------------------------------

def _cost_matrix(x: Tensor, y: Tensor) -> Tensor:
    """Squared-Euclidean cost matrix between point clouds x and y."""
    return torch.cdist(x, y, p=2.0).pow(2)


def sinkhorn_distance(
    x: Tensor,
    y: Tensor,
    blur: float = 0.05,
    n_iters: int = 50,
    p: float = 2.0,
) -> Tensor:
    """Compute the (entropy-regularised) Sinkhorn divergence between two point clouds.

    Args:
        x: Point cloud of shape ``(N, D)``.
        y: Point cloud of shape ``(M, D)``.
        blur: Entropic regularisation strength (epsilon = blur^p).
        n_iters: Number of Sinkhorn iterations.
        p: Exponent for the ground cost (default 2 = squared Euclidean).

    Returns:
        Scalar Sinkhorn divergence (debiased).
    """
    eps = blur ** p
    N, M = x.shape[0], y.shape[0]

    # Uniform marginals (log-domain for numerical stability)
    mu = torch.full((N,), -math.log(N) if N > 0 else 0.0, device=x.device, dtype=x.dtype)
    nu = torch.full((M,), -math.log(M) if M > 0 else 0.0, device=y.device, dtype=y.dtype)

    def _sinkhorn(C: Tensor, log_a: Tensor, log_b: Tensor) -> Tensor:
        """Log-domain Sinkhorn iterations for cost matrix *C*."""
        f = torch.zeros_like(log_a)
        g = torch.zeros_like(log_b)
        for _ in range(n_iters):
            f = -eps * torch.logsumexp((-C + g[None, :]) / eps + log_b[None, :], dim=1)
            g = -eps * torch.logsumexp((-C + f[:, None]) / eps + log_a[:, None], dim=0)
        # Primal cost estimate
        return (f * log_a.exp()).sum() + (g * log_b.exp()).sum()

    C_xy = _cost_matrix(x, y)
    C_xx = _cost_matrix(x, x)
    C_yy = _cost_matrix(y, y)

    # Debiased Sinkhorn divergence: S(x,y) - 0.5*S(x,x) - 0.5*S(y,y)
    loss_xy = _sinkhorn(C_xy, mu, nu)
    loss_xx = _sinkhorn(C_xx, mu, mu)
    loss_yy = _sinkhorn(C_yy, nu, nu)

    return loss_xy - 0.5 * loss_xx - 0.5 * loss_yy


# ---------------------------------------------------------------------------
# Nearest-neighbour expression MSE
# ---------------------------------------------------------------------------

def nearest_neighbor_mse(
    pred_coords: Tensor,
    pred_expr: Tensor,
    true_coords: Tensor,
    true_expr: Tensor,
) -> Tensor:
    """1-NN MSE: for each predicted point, match to the nearest ground-truth
    point by spatial coordinates and compute the MSE on expression features.

    Args:
        pred_coords: ``(N, D)`` predicted spatial coordinates.
        pred_expr: ``(N, F)`` predicted expression features (e.g. PCs).
        true_coords: ``(M, D)`` ground-truth spatial coordinates.
        true_expr: ``(M, F)`` ground-truth expression features.

    Returns:
        Scalar MSE between predicted expression and nearest true expression.
    """
    # (N, M) pairwise distances
    dists = torch.cdist(pred_coords, true_coords)
    nn_idx = dists.argmin(dim=1)  # (N,)
    nn_expr = true_expr[nn_idx]   # (N, F)
    return torch.mean((pred_expr - nn_expr) ** 2)


# ---------------------------------------------------------------------------
# Combined loss with curriculum scheduling
# ---------------------------------------------------------------------------

def curriculum_expr_weight(
    epoch: int,
    start_epoch: int = 0,
    end_epoch: int = 60,
    start_weight: float = 0.1,
    end_weight: float = 1.0,
) -> float:
    """Compute expression loss weight for the current epoch using linear ramp.

    Between ``start_epoch`` and ``end_epoch`` the weight increases linearly
    from ``start_weight`` to ``end_weight``.  Before/after it is clamped.
    """
    if epoch <= start_epoch:
        return start_weight
    if epoch >= end_epoch:
        return end_weight
    frac = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
    return start_weight + frac * (end_weight - start_weight)


def combined_loss(
    pred_coords: Tensor,
    pred_expr: Tensor,
    true_coords: Tensor,
    true_expr: Tensor,
    w_spatial: float = 1.0,
    w_expr: float = 1.0,
    sinkhorn_blur: float = 0.05,
    sinkhorn_iters: int = 50,
) -> dict[str, Tensor]:
    """Weighted combination of Sinkhorn spatial loss and NN-MSE expression loss.

    Returns a dict with keys ``"total"``, ``"spatial"``, ``"expression"``.
    """
    loss_spatial = sinkhorn_distance(
        pred_coords, true_coords, blur=sinkhorn_blur, n_iters=sinkhorn_iters,
    )
    loss_expr = nearest_neighbor_mse(pred_coords, pred_expr, true_coords, true_expr)

    total = w_spatial * loss_spatial + w_expr * loss_expr
    return {
        "total": total,
        "spatial": loss_spatial.detach(),
        "expression": loss_expr.detach(),
    }
