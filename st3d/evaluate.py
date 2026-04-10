"""
Evaluation of a trained Schrödinger bridge model against held-out slices.

For each held-out section, locates the two flanking training sections,
simulates the forward and backward SDE to the held-out z-depth, blends
bidirectionally, and compares to ground truth using three complementary
metrics:

* **Sinkhorn distance** — distributional match on spatial coordinates
  (how well the predicted point cloud covers the same tissue geometry).
* **NN-MSE** — for each predicted cell, finds its nearest spatial
  neighbour in the ground truth and compares expression PCs (how
  accurately the model recovers gene-expression at corresponding
  locations).
* **Pearson correlation** — per-PC correlation between the predicted
  and NN-matched ground-truth expression vectors (how well the model
  preserves relative expression structure, independent of scale).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from st3d.bridge import (
    SchrodingerBridgeModel,
    simulate_sde_to,
)
from st3d.losses import sinkhorn_distance


# ---------------------------------------------------------------------------
# Per-slice metrics
# ---------------------------------------------------------------------------

@dataclass
class SliceMetrics:
    """Evaluation metrics for a single held-out slice."""

    z: float                     # normalised z-coordinate
    bregma: float | None = None  # original Bregma value (if available)

    sinkhorn_spatial: float = 0.0   # on xy coordinates only
    sinkhorn_full: float = 0.0      # on full state (coords + PCs)
    nn_mse_expr: float = 0.0        # NN-MSE on expression PCs
    nn_mse_spatial: float = 0.0     # NN-MSE on xy coordinates
    pearson_per_pc: list[float] = field(default_factory=list)
    pearson_mean: float = 0.0       # mean Pearson r across PCs

    n_pred: int = 0
    n_true: int = 0


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _nn_mse_and_correlation(
    pred: Tensor,
    true: Tensor,
) -> tuple[float, float, float, list[float]]:
    """Compute NN-MSE and Pearson correlation between two point clouds.

    For each predicted point, finds its nearest spatial neighbour in the
    ground truth (using the first 3 coordinate dims), then computes:
    - MSE on expression PCs (dims 3:)
    - MSE on spatial xy (dims 0:2)
    - Per-PC Pearson correlation

    Returns:
        (nn_mse_expr, nn_mse_spatial, pearson_mean, pearson_per_pc)
    """
    pred_coords = pred[:, :3]
    pred_expr = pred[:, 3:]
    true_coords = true[:, :3]
    true_expr = true[:, 3:]

    # Nearest-neighbour matching on spatial coordinates
    dists = torch.cdist(pred_coords, true_coords)  # (N_pred, N_true)
    nn_idx = dists.argmin(dim=1)                     # (N_pred,)

    matched_coords = true_coords[nn_idx]
    matched_expr = true_expr[nn_idx]

    # Expression NN-MSE
    nn_mse_expr = float(torch.mean((pred_expr - matched_expr) ** 2).item())

    # Spatial NN-MSE (xy only)
    nn_mse_spatial = float(torch.mean((pred_coords[:, :2] - matched_coords[:, :2]) ** 2).item())

    # Pearson correlation per PC
    pearson_per_pc: list[float] = []
    n_pcs = pred_expr.shape[1]
    for pc in range(n_pcs):
        p = pred_expr[:, pc]
        t = matched_expr[:, pc]

        p_centered = p - p.mean()
        t_centered = t - t.mean()

        num = (p_centered * t_centered).sum()
        denom = torch.sqrt((p_centered ** 2).sum() * (t_centered ** 2).sum())

        if denom > 1e-12:
            r = float((num / denom).item())
        else:
            r = 0.0
        pearson_per_pc.append(r)

    pearson_mean = float(np.mean(pearson_per_pc))

    return nn_mse_expr, nn_mse_spatial, pearson_mean, pearson_per_pc


# ---------------------------------------------------------------------------
# Predict at a held-out z-depth via bidirectional blending
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_at_z(
    model: SchrodingerBridgeModel,
    u_left: Tensor,
    u_right: Tensor,
    alpha: float,
    sigma: float,
    sde_steps: int,
    device: torch.device,
) -> Tensor:
    """Predict the state at bridge-time *alpha* between two flanking sections.

    1. Forward SDE from ``u_left`` (``t = 0``) to ``t = alpha``.
    2. Backward SDE from ``u_right`` (``t = 1``) to ``t = alpha``.
    3. NN-aligned bidirectional blend weighted by ``alpha``.
    """
    model.eval()
    u_left = u_left.to(device)
    u_right = u_right.to(device)

    x_fwd = simulate_sde_to(
        model.forward_drift, u_left, sigma,
        n_steps=sde_steps, t_target=alpha, forward=True,
    )
    x_bwd = simulate_sde_to(
        model.backward_drift, u_right, sigma,
        n_steps=sde_steps, t_target=alpha, forward=False,
    )

    # NN-aligned blend (anchor = larger cloud)
    if x_fwd.shape[0] >= x_bwd.shape[0]:
        anchor, match = x_fwd, x_bwd
        w_anchor, w_match = 1.0 - alpha, alpha
    else:
        anchor, match = x_bwd, x_fwd
        w_anchor, w_match = alpha, 1.0 - alpha

    dists = torch.cdist(anchor[:, :3], match[:, :3])
    nn_idx = dists.argmin(dim=1)
    match_aligned = match[nn_idx]

    blended = w_anchor * anchor + w_match * match_aligned

    # Fix z-coordinate to the exact target depth
    z_left = u_left[:, 2].mean().item()
    z_right = u_right[:, 2].mean().item()
    blended[:, 2] = z_left + (z_right - z_left) * alpha

    return blended


# ---------------------------------------------------------------------------
# Full evaluation against held-out slices
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_heldout(
    model: SchrodingerBridgeModel,
    train_tensors: list[Tensor],
    train_z: list[float],
    heldout_tensors: list[Tensor],
    heldout_z: list[float],
    sigma: float = 0.1,
    sde_steps: int = 50,
    sinkhorn_blur: float = 0.05,
    sinkhorn_iters: int = 100,
    device: str = "auto",
    heldout_bregma: list[float] | None = None,
) -> list[SliceMetrics]:
    """Evaluate reconstruction quality at each held-out z-plane.

    For every held-out section:

    1. Find the two flanking training sections (the nearest training z
       below and above).
    2. Compute the bridge time ``alpha`` = fractional position between
       the flanking pair.
    3. Predict via bidirectional SDE + blending.
    4. Compare prediction to ground-truth using Sinkhorn distance,
       NN-MSE, and Pearson correlation.

    Args:
        model: Trained ``SchrodingerBridgeModel``.
        train_tensors: Preprocessed training section tensors (z-ordered).
        train_z: Normalised z-coordinates of training sections.
        heldout_tensors: Preprocessed held-out section tensors.
        heldout_z: Normalised z-coordinates of held-out sections.
        sigma: Diffusion coefficient (must match training).
        sde_steps: Euler--Maruyama steps for a full ``[0, 1]`` traversal.
        sinkhorn_blur: Blur for Sinkhorn distance evaluation.
        sinkhorn_iters: Iterations for Sinkhorn distance.
        device: ``"auto"`` / ``"cuda"`` / ``"cpu"``.
        heldout_bregma: Optional original Bregma values for reporting.

    Returns:
        List of ``SliceMetrics``, one per held-out section.
    """
    from st3d.training import get_device

    dev = get_device(device)
    model = model.to(dev)
    model.eval()

    results: list[SliceMetrics] = []

    for h_idx, (h_tensor, h_z) in enumerate(zip(heldout_tensors, heldout_z)):
        bregma = heldout_bregma[h_idx] if heldout_bregma is not None else None

        # --- Find flanking training sections ---
        left_idx = None
        right_idx = None
        for k, tz in enumerate(train_z):
            if tz <= h_z:
                left_idx = k
            if tz >= h_z and right_idx is None:
                right_idx = k

        if left_idx is None or right_idx is None or left_idx == right_idx:
            # Held-out slice is outside the training z-range; skip
            print(f"  [SKIP] Held-out z={h_z:.3f} "
                  f"(Bregma {bregma:+.2f}): outside training range" if bregma
                  else f"  [SKIP] Held-out z={h_z:.3f}: outside training range")
            continue

        z_left = train_z[left_idx]
        z_right = train_z[right_idx]
        gap = z_right - z_left
        alpha = (h_z - z_left) / gap if gap > 0 else 0.5

        # --- Predict via bidirectional SDE ---
        pred = predict_at_z(
            model,
            train_tensors[left_idx],
            train_tensors[right_idx],
            alpha=alpha,
            sigma=sigma,
            sde_steps=sde_steps,
            device=dev,
        )

        true = h_tensor.to(dev)

        # --- Sinkhorn distances ---
        sink_spatial = float(sinkhorn_distance(
            pred[:, :2], true[:, :2],
            blur=sinkhorn_blur, n_iters=sinkhorn_iters,
        ).item())

        sink_full = float(sinkhorn_distance(
            pred, true,
            blur=sinkhorn_blur, n_iters=sinkhorn_iters,
        ).item())

        # --- NN-MSE and Pearson ---
        nn_mse_expr, nn_mse_spatial, pearson_mean, pearson_per_pc = \
            _nn_mse_and_correlation(pred, true)

        metrics = SliceMetrics(
            z=h_z,
            bregma=bregma,
            sinkhorn_spatial=sink_spatial,
            sinkhorn_full=sink_full,
            nn_mse_expr=nn_mse_expr,
            nn_mse_spatial=nn_mse_spatial,
            pearson_per_pc=pearson_per_pc,
            pearson_mean=pearson_mean,
            n_pred=pred.shape[0],
            n_true=true.shape[0],
        )
        results.append(metrics)

        bregma_str = f"Bregma {bregma:+.2f}" if bregma is not None else ""
        print(f"  z={h_z:.3f} {bregma_str:>14s}  |  "
              f"Sink(xy)={sink_spatial:+.4f}  "
              f"Sink(full)={sink_full:+.4f}  |  "
              f"NN-MSE(expr)={nn_mse_expr:.4f}  "
              f"NN-MSE(xy)={nn_mse_spatial:.4f}  |  "
              f"Pearson={pearson_mean:.3f}")

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[SliceMetrics]) -> dict[str, float]:
    """Print a summary table and return aggregate metrics.

    Returns:
        Dict with keys ``"sinkhorn_spatial_mean"``, ``"sinkhorn_full_mean"``,
        ``"nn_mse_expr_mean"``, ``"nn_mse_spatial_mean"``,
        ``"pearson_mean"``.
    """
    if not results:
        print("  No held-out slices evaluated.")
        return {}

    print()
    print("=" * 80)
    print(f"{'z':>7s}  {'Bregma':>8s}  "
          f"{'Sink(xy)':>10s}  {'Sink(full)':>11s}  "
          f"{'NN-MSE(e)':>10s}  {'NN-MSE(xy)':>11s}  "
          f"{'Pearson':>8s}")
    print("-" * 80)

    for m in results:
        bregma_str = f"{m.bregma:+.2f}" if m.bregma is not None else "—"
        print(f"{m.z:7.3f}  {bregma_str:>8s}  "
              f"{m.sinkhorn_spatial:10.4f}  {m.sinkhorn_full:11.4f}  "
              f"{m.nn_mse_expr:10.4f}  {m.nn_mse_spatial:11.4f}  "
              f"{m.pearson_mean:8.3f}")

    print("-" * 80)

    agg = {
        "sinkhorn_spatial_mean": float(np.mean([m.sinkhorn_spatial for m in results])),
        "sinkhorn_full_mean": float(np.mean([m.sinkhorn_full for m in results])),
        "nn_mse_expr_mean": float(np.mean([m.nn_mse_expr for m in results])),
        "nn_mse_spatial_mean": float(np.mean([m.nn_mse_spatial for m in results])),
        "pearson_mean": float(np.mean([m.pearson_mean for m in results])),
    }

    print(f"{'MEAN':>7s}  {'':>8s}  "
          f"{agg['sinkhorn_spatial_mean']:10.4f}  "
          f"{agg['sinkhorn_full_mean']:11.4f}  "
          f"{agg['nn_mse_expr_mean']:10.4f}  "
          f"{agg['nn_mse_spatial_mean']:11.4f}  "
          f"{agg['pearson_mean']:8.3f}")
    print("=" * 80)

    return agg
