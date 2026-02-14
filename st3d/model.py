"""
Core neural network modules for st3d.

Provides:
- ``LocalSpatialAttention``: KNN-masked multi-head attention replacing PyG TransformerConv.
- ``DriftNet``: Predicts coordinate and expression drift at each integration step.
- ``ST3DModel``: Top-level model wrapping forward/backward DriftNets.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from st3d.config import ST3DConfig


# ---------------------------------------------------------------------------
# Local Spatial Attention
# ---------------------------------------------------------------------------

class LocalSpatialAttentionLayer(nn.Module):
    """Single layer of multi-head attention restricted to KNN neighbours.

    For each point the attention mask only allows attending to its *k* nearest
    spatial neighbours (plus itself).  This gives local-graph reasoning
    using only standard PyTorch operations.
    """

    def __init__(self, dim: int, n_heads: int = 4, k_neighbors: int = 10, dropout: float = 0.0):
        super().__init__()
        self.k = k_neighbors
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def _build_knn_mask(self, coords: Tensor) -> Tensor:
        """Return a boolean attention mask of shape ``(N, N)``
        where ``True`` means *do not attend* (following PyTorch convention).
        """
        N = coords.shape[0]
        k = min(self.k, N)
        dist = torch.cdist(coords, coords)  # (N, N)
        # For each row keep the k smallest distances (including self)
        _, knn_idx = dist.topk(k, dim=1, largest=False)  # (N, k)
        # Build mask: start as all-True (block everything) then unmask neighbours
        mask = torch.ones(N, N, dtype=torch.bool, device=coords.device)
        row_idx = torch.arange(N, device=coords.device).unsqueeze(1).expand_as(knn_idx)
        mask[row_idx, knn_idx] = False
        return mask  # True = ignore

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        """
        Args:
            x: Node features ``(N, D)``.
            coords: Spatial coordinates ``(N, 2)`` or ``(N, 3)``.

        Returns:
            Updated features ``(N, D)``.
        """
        # Add a fake batch dimension for nn.MultiheadAttention
        mask = self._build_knn_mask(coords)  # (N, N)
        h = self.norm1(x)
        h = h.unsqueeze(0)  # (1, N, D)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        attn_out = attn_out.squeeze(0)  # (N, D)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class LocalSpatialAttention(nn.Module):
    """Stack of ``LocalSpatialAttentionLayer`` modules."""

    def __init__(self, dim: int, n_heads: int = 4, n_layers: int = 3,
                 k_neighbors: int = 10, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            LocalSpatialAttentionLayer(dim, n_heads, k_neighbors, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor, coords: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, coords)
        return x


# ---------------------------------------------------------------------------
# Drift network
# ---------------------------------------------------------------------------

class DriftNet(nn.Module):
    """Predicts per-point coordinate drift and expression drift.

    Architecture::

        [coords || expression_pcs]
              |
        Linear -> hidden_dim
              |
        LocalSpatialAttention (N layers)
              |
        +-----+-----+
        CoordHead   ExprHead
        d(x,y,z)    d(pcs)

    During training, optional Gaussian noise is injected into the output
    (SDE mode) to improve robustness.
    """

    def __init__(self, config: ST3DConfig):
        super().__init__()
        input_dim = 3 + config.n_pcs
        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.encoder = LocalSpatialAttention(
            dim=config.hidden_dim,
            n_heads=config.n_heads,
            n_layers=config.n_attn_layers,
            k_neighbors=config.k_neighbors,
        )
        # Coordinate drift head
        self.coord_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 3),
        )
        # Expression drift head
        self.expr_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.n_pcs),
        )
        # Noise parameters (SDE mode)
        self.noise_std_xy = config.noise_std_xy
        self.noise_std_z = config.noise_std_z
        self.noise_std_expr = config.noise_std_expr

    def forward(
        self, coords: Tensor, expression: Tensor, add_noise: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            coords: ``(N, 3)`` spatial coordinates.
            expression: ``(N, n_pcs)`` expression PCs.
            add_noise: If ``True`` inject Gaussian noise (training only).

        Returns:
            ``(d_coords, d_expr)`` -- drift tensors of the same shapes.
        """
        x = torch.cat([coords, expression], dim=-1)  # (N, 3+n_pcs)
        x = self.proj(x)                              # (N, hidden_dim)
        x = self.encoder(x, coords[:, :2])            # attend over xy-plane

        d_coords = self.coord_head(x)                 # (N, 3)
        d_expr = self.expr_head(x)                     # (N, n_pcs)

        if add_noise and self.training:
            noise_c = torch.randn_like(d_coords)
            noise_c[:, 0] *= self.noise_std_xy
            noise_c[:, 1] *= self.noise_std_xy
            noise_c[:, 2] *= self.noise_std_z
            d_coords = d_coords + noise_c

            noise_e = torch.randn_like(d_expr) * self.noise_std_expr
            d_expr = d_expr + noise_e

        return d_coords, d_expr


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class ST3DModel(nn.Module):
    """Model for 3D spatial transcriptomics reconstruction from serial sections.

    Wraps a forward and (optionally) backward ``DriftNet``.  High-level
    ``fit`` and ``interpolate`` convenience methods are intentionally *not*
    placed here -- they live in ``training.py`` and ``inference.py``
    respectively so that the ``nn.Module`` stays a clean, serialisable
    network definition.
    """

    def __init__(self, config: ST3DConfig | None = None):
        super().__init__()
        if config is None:
            config = ST3DConfig()
        self.config = config

        self.forward_net = DriftNet(config)
        if config.bidirectional:
            self.backward_net = DriftNet(config)
        else:
            self.backward_net = None

    # -- Euler integration helpers ------------------------------------------

    def _euler_step(
        self, net: DriftNet, coords: Tensor, expr: Tensor, dz: float, add_noise: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Single Euler step: state_{k+1} = state_k + drift * dz."""
        dc, de = net(coords, expr, add_noise=add_noise)
        return coords + dc * dz, expr + de * dz

    def forward_integrate(
        self, coords: Tensor, expr: Tensor, n_steps: int, dz: float, add_noise: bool = False,
    ) -> list[tuple[Tensor, Tensor]]:
        """Integrate the forward net for *n_steps*, returning intermediate states."""
        trajectory: list[tuple[Tensor, Tensor]] = []
        c, e = coords, expr
        for _ in range(n_steps):
            c, e = self._euler_step(self.forward_net, c, e, dz, add_noise)
            trajectory.append((c, e))
        return trajectory

    def backward_integrate(
        self, coords: Tensor, expr: Tensor, n_steps: int, dz: float, add_noise: bool = False,
    ) -> list[tuple[Tensor, Tensor]]:
        """Integrate the backward net for *n_steps* (negative direction)."""
        assert self.backward_net is not None, "Model is not bidirectional"
        trajectory: list[tuple[Tensor, Tensor]] = []
        c, e = coords, expr
        for _ in range(n_steps):
            c, e = self._euler_step(self.backward_net, c, e, -dz, add_noise)
            trajectory.append((c, e))
        return trajectory

    # -- Serialisation ------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and config."""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "ST3DModel":
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(config=ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
