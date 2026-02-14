"""
Configuration dataclasses for st3d.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ST3DConfig:
    """Configuration for the ST3DModel.

    Attributes:
        n_pcs: Number of principal components for gene expression.
        hidden_dim: Hidden dimension for drift networks.
        n_heads: Number of attention heads in LocalSpatialAttention.
        n_attn_layers: Number of attention layers in the spatial encoder.
        k_neighbors: Number of nearest neighbors for local attention masking.
        dz: Integration step size along the z-axis (Delta z).
        noise_std_xy: Noise standard deviation for x/y coordinate drift (SDE mode).
        noise_std_z: Noise standard deviation for z coordinate drift (SDE mode).
        noise_std_expr: Noise standard deviation for expression drift (SDE mode).
        bidirectional: Whether to use bidirectional (forward + backward) propagation.
        n_hvgs: Number of highly variable genes for preprocessing.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        optimizer: Optimizer name ("NAdam", "Adam", "AdamW").
        grad_clip: Maximum gradient norm for clipping.
        spatial_loss_weight: Weight for spatial (Sinkhorn) loss.
        expr_loss_weight_start: Starting weight for expression loss (curriculum).
        expr_loss_weight_end: Final weight for expression loss (curriculum).
        curriculum_start_epoch: Epoch to begin ramping expression loss.
        curriculum_end_epoch: Epoch at which expression loss reaches full weight.
        sinkhorn_blur: Blur parameter for Sinkhorn distance.
        sinkhorn_iters: Number of Sinkhorn iterations.
    """

    # Model architecture
    n_pcs: int = 50
    hidden_dim: int = 64
    n_heads: int = 4
    n_attn_layers: int = 3
    k_neighbors: int = 10

    # Integration
    dz: float = 0.01
    noise_std_xy: float = 0.01
    noise_std_z: float = 0.0
    noise_std_expr: float = 0.1
    bidirectional: bool = True

    # Preprocessing
    n_hvgs: int = 2000

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-8
    optimizer: str = "NAdam"
    grad_clip: float = 1.0

    # Loss scheduling (curriculum)
    spatial_loss_weight: float = 1.0
    expr_loss_weight_start: float = 0.1
    expr_loss_weight_end: float = 1.0
    curriculum_start_epoch: int = 0
    curriculum_end_epoch: int = 60

    # Sinkhorn parameters
    sinkhorn_blur: float = 0.05
    sinkhorn_iters: int = 50
