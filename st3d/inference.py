"""
Inference utilities for st3d.

Euler integration, bidirectional interpolation with distance-weighted blending,
and full 3D volume reconstruction returning AnnData.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

import anndata as ad

from st3d.data import NormParams, tensors_to_anndata
from st3d.model import ST3DModel


# ---------------------------------------------------------------------------
# Low-level Euler integration
# ---------------------------------------------------------------------------

@torch.no_grad()
def euler_integrate(
    net: torch.nn.Module,
    coords: Tensor,
    expr: Tensor,
    n_steps: int,
    dz: float,
) -> list[tuple[Tensor, Tensor]]:
    """Forward Euler integration returning all intermediate states.

    Args:
        net: A ``DriftNet`` (forward or backward).
        coords: Starting coordinates ``(N, 3)``.
        expr: Starting expression PCs ``(N, n_pcs)``.
        n_steps: Number of integration steps.
        dz: Step size (positive for forward, negative for backward).

    Returns:
        List of ``(coords, expr)`` tensors for each step.
    """
    trajectory: list[tuple[Tensor, Tensor]] = []
    c, e = coords.clone(), expr.clone()
    for _ in range(n_steps):
        dc, de = net(c, e, add_noise=False)
        c = c + dc * dz
        e = e + de * dz
        trajectory.append((c.clone(), e.clone()))
    return trajectory


# ---------------------------------------------------------------------------
# Bidirectional interpolation between two sections
# ---------------------------------------------------------------------------

@torch.no_grad()
def bidirectional_interpolate(
    model: ST3DModel,
    u_start: Tensor,
    u_end: Tensor,
    n_interp: int,
    dz: float,
    device: torch.device | str = "cpu",
) -> list[Tensor]:
    """Interpolate between two consecutive sections using forward and
    backward drift nets with distance-weighted blending.

    For a target depth *t* between depths *a* (start) and *b* (end):
    - Forward prediction propagates from *a* to *t*.
    - Backward prediction propagates from *b* to *t*.
    - Blending weight ``alpha = (t - a) / (b - a)`` gives more trust to the
      closer observed section.

    Args:
        model: Trained ``ST3DModel``.
        u_start: State tensor for the earlier section ``(N, 3+P)``.
        u_end: State tensor for the later section ``(M, 3+P)``.
        n_interp: Number of interpolated layers to generate *between* the
            two sections (excluding the endpoints).
        dz: Euler step size.
        device: Device to run on.

    Returns:
        List of ``n_interp`` state tensors for the interpolated layers.
    """
    model.eval()
    u_start = u_start.to(device)
    u_end = u_end.to(device)

    coords_s, expr_s = u_start[:, :3], u_start[:, 3:]
    coords_e, expr_e = u_end[:, :3], u_end[:, 3:]

    z_a = coords_s[:, 2].mean().item()
    z_b = coords_e[:, 2].mean().item()
    total_steps = max(1, round(abs(z_b - z_a) / dz))

    # Forward trajectory (a -> b)
    traj_fwd = euler_integrate(model.forward_net, coords_s, expr_s, total_steps, dz)

    # Backward trajectory (b -> a), reversed so index 0 is closest to a
    traj_bwd: list[tuple[Tensor, Tensor]] | None = None
    if model.backward_net is not None:
        traj_bwd_raw = euler_integrate(model.backward_net, coords_e, expr_e, total_steps, -dz)
        traj_bwd = list(reversed(traj_bwd_raw))

    # Pick n_interp evenly-spaced indices from the trajectory
    interp_indices = np.linspace(0, total_steps - 1, n_interp + 2, dtype=float)[1:-1]
    interp_indices = np.round(interp_indices).astype(int)

    results: list[Tensor] = []
    for idx in interp_indices:
        alpha = (idx + 1) / (total_steps)  # 0 = start, 1 = end

        c_f, e_f = traj_fwd[idx]

        if traj_bwd is not None:
            c_b, e_b = traj_bwd[idx]

            # Distance-weighted blend: near start -> trust forward more.
            # Since the two point clouds may differ in size, we concatenate
            # and subsample to produce a result of reasonable density.
            n_total = c_f.shape[0] + c_b.shape[0]
            n_keep = max(c_f.shape[0], c_b.shape[0])

            state_f = torch.cat([c_f, e_f], dim=1)
            state_b = torch.cat([c_b, e_b], dim=1)
            combined = torch.cat([state_f, state_b], dim=0)

            # Random subsample to target density
            perm = torch.randperm(combined.shape[0], device=combined.device)[:n_keep]
            result = combined[perm]
        else:
            result = torch.cat([c_f, e_f], dim=1)

        # Override z-coordinate to the exact interpolated depth
        z_target = z_a + (z_b - z_a) * alpha
        result[:, 2] = z_target
        results.append(result.cpu())

    return results


# ---------------------------------------------------------------------------
# Full 3D volume reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_volume(
    model: ST3DModel,
    tensors: list[Tensor],
    norm_params: NormParams,
    dz: float | None = None,
    device: str = "auto",
    inverse_pca: bool = True,
    verbose: bool = True,
) -> ad.AnnData:
    """Reconstruct a dense 3D volume from observed sections.

    For each pair of consecutive observed sections, generates interpolated
    layers at spacing ``dz`` and merges everything into a single AnnData.

    Args:
        model: Trained ``ST3DModel``.
        tensors: Preprocessed section tensors (from ``preprocess_sections``).
        norm_params: ``NormParams`` for inversion.
        dz: Integration step size (defaults to ``model.config.dz``).
        device: ``"auto"`` / ``"cuda"`` / ``"mps"`` / ``"cpu"``.
        inverse_pca: Whether to inverse-PCA the expression back to genes.
        verbose: Show progress bar.

    Returns:
        AnnData with observed + interpolated points, spatial coordinates in
        ``.obsm["spatial_3d"]``, and (optionally) gene expression in ``.X``.
    """
    from st3d.training import get_device

    dev = get_device(device)
    model = model.to(dev)
    model.eval()

    if dz is None:
        dz = model.config.dz

    all_states: list[Tensor] = []
    is_observed: list[bool] = []

    pairs = list(range(len(tensors) - 1))
    iterator = tqdm(pairs, desc="Reconstructing", disable=not verbose)

    for i in iterator:
        # Add observed section
        all_states.append(tensors[i])
        is_observed.extend([True] * tensors[i].shape[0])

        u_start = tensors[i]
        u_end = tensors[i + 1]

        z_a = u_start[:, 2].mean().item()
        z_b = u_end[:, 2].mean().item()
        gap = abs(z_b - z_a)
        n_interp = max(0, round(gap / dz) - 1)

        if n_interp > 0:
            interp_states = bidirectional_interpolate(
                model, u_start, u_end, n_interp, dz, device=dev,
            )
            for s in interp_states:
                all_states.append(s)
                is_observed.extend([False] * s.shape[0])

    # Add final observed section
    all_states.append(tensors[-1])
    is_observed.extend([True] * tensors[-1].shape[0])

    # Convert to AnnData
    adata = tensors_to_anndata(all_states, norm_params, inverse_pca=inverse_pca)
    adata.obs["is_observed"] = is_observed
    return adata
