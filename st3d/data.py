"""
Data preprocessing and dataset utilities for st3d.

Handles the full pipeline from raw AnnData sections to normalised tensors
suitable for training, and provides inverse transforms to convert predictions
back into AnnData objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Normalisation metadata (stored alongside the model for inversion)
# ---------------------------------------------------------------------------

@dataclass
class NormParams:
    """Stores all parameters needed to reverse preprocessing."""

    # Coordinate normalisation
    xy_min: np.ndarray = field(default_factory=lambda: np.zeros(2))
    xy_scale: float = 1.0

    # PCA model
    pca: PCA | None = None
    gene_names: list[str] = field(default_factory=list)
    n_pcs: int = 50

    # Per-PC min-max normalisation
    pc_min: np.ndarray | None = None
    pc_scale: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_sections(
    adatas: Sequence[ad.AnnData],
    z_coords: Sequence[float],
    n_hvgs: int = 2000,
    n_pcs: int = 50,
    spatial_key: str = "spatial",
) -> tuple[list[Tensor], NormParams]:
    """Preprocess a list of serial AnnData sections into normalised tensors.

    Pipeline:
    1. Concatenate, normalise total counts, ``log1p``, select HVGs.
    2. Fit PCA on concatenated matrix.
    3. Project each section to PCs; min-max normalise PCs.
    4. Isotropic-scale XY coordinates.
    5. Return list of ``(N_i, 3 + n_pcs)`` tensors and a ``NormParams``.

    Args:
        adatas: One ``AnnData`` per physical section.
        z_coords: Matching z-depth for each section.
        n_hvgs: Number of highly variable genes to select.
        n_pcs: Number of principal components.
        spatial_key: Key in ``adata.obsm`` for XY coordinates.

    Returns:
        ``(tensors, norm_params)`` -- tensors are ready for the model.
    """
    assert len(adatas) == len(z_coords), "Need one z-coordinate per section"

    # --- 1. Concatenate & basic preprocessing ---
    combined = ad.concat(adatas, label="section", keys=list(range(len(adatas))))
    if issparse(combined.X):
        combined.X = combined.X.toarray()
    combined.X = combined.X.astype(np.float32)
    # Standard scanpy order: normalize -> log1p -> HVG
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    sc.pp.highly_variable_genes(combined, n_top_genes=min(n_hvgs, combined.shape[1]), flavor="seurat")
    hvg_mask = combined.var["highly_variable"].values
    gene_names = list(combined.var_names[hvg_mask])

    expr_hvg = combined.X[:, hvg_mask]

    # --- 2. PCA ---
    pca = PCA(n_components=n_pcs)
    pcs_all = pca.fit_transform(expr_hvg).astype(np.float32)

    # Min-max normalise PCs
    pc_min = pcs_all.min(axis=0)
    pc_max = pcs_all.max(axis=0)
    pc_scale = pc_max - pc_min
    pc_scale[pc_scale == 0] = 1.0
    pcs_all = (pcs_all - pc_min) / pc_scale

    # --- 3. XY normalisation (isotropic scaling) ---
    all_coords = []
    section_sizes = []
    for adata in adatas:
        xy = np.array(adata.obsm[spatial_key][:, :2], dtype=np.float32)
        all_coords.append(xy)
        section_sizes.append(xy.shape[0])
    all_xy = np.concatenate(all_coords, axis=0)

    xy_min = all_xy.min(axis=0)
    width = all_xy.max(axis=0) - xy_min
    xy_scale = float(max(width[0], width[1]))
    if xy_scale == 0:
        xy_scale = 1.0

    # --- 4. Build per-section tensors ---
    tensors: list[Tensor] = []
    offset = 0
    for i, n in enumerate(section_sizes):
        xy_normed = (all_coords[i] - xy_min) / xy_scale
        z_col = np.full((n, 1), z_coords[i], dtype=np.float32)
        coords = np.concatenate([xy_normed, z_col], axis=1)  # (N, 3)
        pcs_section = pcs_all[offset: offset + n]             # (N, n_pcs)
        state = np.concatenate([coords, pcs_section], axis=1)  # (N, 3+n_pcs)
        tensors.append(torch.from_numpy(state))
        offset += n

    norm_params = NormParams(
        xy_min=xy_min,
        xy_scale=xy_scale,
        pca=pca,
        gene_names=gene_names,
        n_pcs=n_pcs,
        pc_min=pc_min,
        pc_scale=pc_scale,
    )
    return tensors, norm_params


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def normalize_coords(coords: np.ndarray, xy_min: np.ndarray, xy_scale: float) -> np.ndarray:
    """Apply isotropic XY normalisation."""
    out = coords.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] - xy_min[0]) / xy_scale
    out[:, 1] = (out[:, 1] - xy_min[1]) / xy_scale
    return out


def denormalize_coords(coords: np.ndarray, xy_min: np.ndarray, xy_scale: float) -> np.ndarray:
    """Reverse the coordinate normalisation."""
    out = coords.copy().astype(np.float32)
    out[:, 0] = out[:, 0] * xy_scale + xy_min[0]
    out[:, 1] = out[:, 1] * xy_scale + xy_min[1]
    return out


# ---------------------------------------------------------------------------
# Inverse PCA transform
# ---------------------------------------------------------------------------

def pcs_to_expression(
    pcs: np.ndarray,
    norm_params: NormParams,
) -> pd.DataFrame:
    """Convert normalised PCs back to log-expression space.

    Args:
        pcs: ``(N, n_pcs)`` normalised principal components.
        norm_params: ``NormParams`` from preprocessing.

    Returns:
        ``DataFrame`` of shape ``(N, n_genes)`` in log-expression space.
    """
    assert norm_params.pca is not None, "NormParams has no PCA model"
    # Reverse min-max
    pcs_raw = pcs * norm_params.pc_scale + norm_params.pc_min
    # Inverse PCA
    expr = norm_params.pca.inverse_transform(pcs_raw)
    return pd.DataFrame(expr, columns=norm_params.gene_names)


# ---------------------------------------------------------------------------
# Tensor <-> AnnData conversion
# ---------------------------------------------------------------------------

def tensors_to_anndata(
    states: list[Tensor] | Tensor,
    norm_params: NormParams,
    inverse_pca: bool = True,
) -> ad.AnnData:
    """Convert model-output tensors back into an AnnData.

    Args:
        states: One or more ``(N, 3+n_pcs)`` state tensors (or a single
            concatenated tensor).
        norm_params: Normalisation metadata for inversion.
        inverse_pca: If ``True``, also invert PCA to gene expression.

    Returns:
        AnnData with ``.obsm["spatial"]`` (denormalised), ``.obsm["spatial_3d"]``,
        ``.obs["z"]``, and optionally ``.X`` in log-expression space.
    """
    if isinstance(states, list):
        state = torch.cat(states, dim=0)
    else:
        state = states
    state_np = state.detach().cpu().numpy()

    coords = state_np[:, :3]
    pcs = state_np[:, 3:]

    # Denormalise coordinates
    xy_raw = denormalize_coords(coords[:, :2], norm_params.xy_min, norm_params.xy_scale)

    obs = pd.DataFrame({"z": coords[:, 2]})

    if inverse_pca and norm_params.pca is not None:
        expr_df = pcs_to_expression(pcs, norm_params)
        adata = ad.AnnData(X=expr_df.values.astype(np.float32), obs=obs, var=pd.DataFrame(index=expr_df.columns))
    else:
        adata = ad.AnnData(X=pcs.astype(np.float32), obs=obs)

    adata.obsm["spatial"] = xy_raw
    adata.obsm["spatial_3d"] = np.column_stack([xy_raw, coords[:, 2]])
    adata.obsm["pcs"] = pcs.astype(np.float32)
    return adata


# ---------------------------------------------------------------------------
# SliceDataset
# ---------------------------------------------------------------------------

class SliceDataset(Dataset):
    """Wraps a list of preprocessed section tensors for training.

    Each item is a pair of *consecutive* sections ``(u_k, u_{k+1})``.
    """

    def __init__(self, tensors: list[Tensor]):
        assert len(tensors) >= 2, "Need at least two sections"
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors) - 1

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.tensors[idx], self.tensors[idx + 1]
