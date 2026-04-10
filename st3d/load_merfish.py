"""
Load the MERFISH dataset from squidpy and prepare it for the st3d pipeline.

Uses the pre-processed MERFISH dataset from Moffitt et al. (2018),
which contains ~73,655 cells across 161 genes from serial sections of the
mouse hypothalamus. Each section corresponds to a distinct Bregma coordinate,
providing the z-axis depth needed for 3D reconstruction.

Supports holding out every Nth z-plane as ground truth for evaluation:
train on the remaining sections and later compare the model's interpolated
predictions against the held-out slices.

Reference:
    Moffitt et al., "Molecular, spatial, and functional single-cell profiling
    of the hypothalamic preoptic region", Science (2018).
    https://doi.org/10.1126/science.aau5324
"""

from __future__ import annotations

import math
import sys
import os
from dataclasses import dataclass

import numpy as np
import scanpy as sc
import anndata as ad
import squidpy as sq
import torch
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from torch import Tensor

# Make st3d importable when running this script directly from the st3d/ dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from st3d.data import NormParams
from st3d.losses import sinkhorn_transport_plan


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MerfishSplit:
    """Holds the train/held-out split of MERFISH sections."""

    # Per-section AnnData objects (raw, before preprocessing)
    train_sections: list[ad.AnnData]
    heldout_sections: list[ad.AnnData]

    # Normalised z-coordinates (same scale for both splits)
    train_z: list[float]
    heldout_z: list[float]

    # Preprocessed tensors (ready for st3d model)
    train_tensors: list[Tensor]
    heldout_tensors: list[Tensor]

    # Shared normalisation parameters (fitted on train only)
    norm_params: NormParams

    # Original Bregma values for reference
    train_bregma: list[float]
    heldout_bregma: list[float]


# ---------------------------------------------------------------------------
# Section loading
# ---------------------------------------------------------------------------

def _find_bregma_column(adata: ad.AnnData) -> str:
    """Find the column in adata.obs that contains Bregma coordinates."""
    if "Bregma" in adata.obs.columns:
        return "Bregma"
    candidates = [c for c in adata.obs.columns if "bregma" in c.lower()]
    if candidates:
        return candidates[0]
    raise KeyError(
        f"Could not find a Bregma column in obs. "
        f"Available columns: {list(adata.obs.columns)}"
    )


def _ensure_spatial(section: ad.AnnData) -> None:
    """Ensure obsm['spatial'] exists, building it from obs columns if needed."""
    if "spatial" in section.obsm:
        return
    xy_candidates = [
        ("Cell_X", "Cell_Y"),
        ("x", "y"),
        ("X", "Y"),
        ("cell_x", "cell_y"),
    ]
    for x_col, y_col in xy_candidates:
        if x_col in section.obs.columns and y_col in section.obs.columns:
            section.obsm["spatial"] = np.column_stack([
                section.obs[x_col].values.astype(np.float32),
                section.obs[y_col].values.astype(np.float32),
            ])
            return
    raise KeyError(
        f"No spatial coordinates found. "
        f"obs columns: {list(section.obs.columns)}, "
        f"obsm keys: {list(section.obsm.keys())}"
    )


def _split_into_sections(adata: ad.AnnData) -> tuple[list[ad.AnnData], list[float]]:
    """Split a single AnnData into per-Bregma-section AnnData objects.

    Returns:
        (sections, bregma_values) sorted by ascending Bregma coordinate.
    """
    bregma_col = _find_bregma_column(adata)
    bregma_values = sorted(adata.obs[bregma_col].unique())

    sections = []
    for bregma in bregma_values:
        section = adata[adata.obs[bregma_col] == bregma].copy()
        _ensure_spatial(section)
        if issparse(section.X):
            section.X = section.X.toarray()
        section.X = section.X.astype(np.float32)
        sections.append(section)

    return sections, [float(b) for b in bregma_values]


# ---------------------------------------------------------------------------
# Joint preprocessing (fit on train, transform both splits)
# ---------------------------------------------------------------------------

def _preprocess_joint(
    train_sections: list[ad.AnnData],
    heldout_sections: list[ad.AnnData],
    train_z: list[float],
    heldout_z: list[float],
    n_hvgs: int,
    n_pcs: int,
    spatial_key: str = "spatial",
) -> tuple[list[Tensor], list[Tensor], NormParams]:
    """Preprocess train and held-out sections with shared normalisation.

    HVG selection, PCA, and coordinate scaling are all fitted on the
    *training* sections only, then applied to both splits. This prevents
    information leakage from held-out sections into the model.

    Returns:
        (train_tensors, heldout_tensors, norm_params)
    """
    # --- 1. Fit normalisation on training sections ---
    combined_train = ad.concat(
        train_sections, label="section", keys=list(range(len(train_sections))),
    )
    if issparse(combined_train.X):
        combined_train.X = combined_train.X.toarray()
    combined_train.X = combined_train.X.astype(np.float32)

    sc.pp.normalize_total(combined_train, target_sum=1e4)
    sc.pp.log1p(combined_train)
    sc.pp.highly_variable_genes(
        combined_train,
        n_top_genes=min(n_hvgs, combined_train.shape[1]),
        flavor="seurat",
    )
    hvg_mask = combined_train.var["highly_variable"].values
    gene_names = list(combined_train.var_names[hvg_mask])

    # PCA fitted on training data
    expr_train = combined_train.X[:, hvg_mask]
    pca = PCA(n_components=n_pcs)
    pcs_train = pca.fit_transform(expr_train).astype(np.float32)

    pc_min = pcs_train.min(axis=0)
    pc_max = pcs_train.max(axis=0)
    pc_scale = pc_max - pc_min
    pc_scale[pc_scale == 0] = 1.0

    # XY normalisation fitted on training sections
    all_xy = np.concatenate(
        [np.array(s.obsm[spatial_key][:, :2], dtype=np.float32) for s in train_sections],
        axis=0,
    )
    xy_min = all_xy.min(axis=0)
    width = all_xy.max(axis=0) - xy_min
    xy_scale = float(max(width[0], width[1]))
    if xy_scale == 0:
        xy_scale = 1.0

    norm_params = NormParams(
        xy_min=xy_min,
        xy_scale=xy_scale,
        pca=pca,
        gene_names=gene_names,
        n_pcs=n_pcs,
        pc_min=pc_min,
        pc_scale=pc_scale,
    )

    # --- 2. Helper to transform a list of sections into tensors ---
    def _to_tensors(
        sections: list[ad.AnnData],
        z_coords: list[float],
    ) -> list[Tensor]:
        tensors: list[Tensor] = []
        for section, z in zip(sections, z_coords):
            # Normalise expression through the same pipeline
            X = section.X.copy() if not issparse(section.X) else section.X.toarray().copy()
            X = X.astype(np.float32)
            # Per-cell normalise + log1p (same as scanpy pipeline)
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            X = X / row_sums * 1e4
            X = np.log1p(X)
            # Select HVGs and project through fitted PCA
            X_hvg = X[:, hvg_mask]
            pcs = pca.transform(X_hvg).astype(np.float32)
            # Min-max normalise using training statistics
            pcs = (pcs - pc_min) / pc_scale
            # Normalise coordinates using training statistics
            xy = np.array(section.obsm[spatial_key][:, :2], dtype=np.float32)
            xy_normed = (xy - xy_min) / xy_scale
            z_col = np.full((xy.shape[0], 1), z, dtype=np.float32)
            coords = np.concatenate([xy_normed, z_col], axis=1)
            state = np.concatenate([coords, pcs], axis=1)
            tensors.append(torch.from_numpy(state))
        return tensors

    train_tensors = _to_tensors(train_sections, train_z)
    heldout_tensors = _to_tensors(heldout_sections, heldout_z)

    return train_tensors, heldout_tensors, norm_params


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_merfish(holdout_every: int = 3) -> MerfishSplit:
    """Load the MERFISH dataset and split into train / held-out sections.

    Every ``holdout_every``-th section (by sorted Bregma order) is held out
    as ground truth for evaluation. The first and last sections are never
    held out so the training set always brackets the full z-range.

    Args:
        holdout_every: Hold out every Nth interior section (default 3).
            Set to 0 to disable holdout (all sections used for training).

    Returns:
        A ``MerfishSplit`` dataclass with train/held-out sections, tensors,
        z-coordinates, and shared normalisation parameters.
    """
    # ------------------------------------------------------------------
    # 1. Load the MERFISH dataset from squidpy
    # ------------------------------------------------------------------
    print("Loading MERFISH dataset from squidpy...")
    adata = sq.datasets.merfish()
    print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    print(f"  obs columns: {list(adata.obs.columns)}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    print()

    # ------------------------------------------------------------------
    # 2. Split into per-section AnnData objects
    # ------------------------------------------------------------------
    all_sections, bregma_values = _split_into_sections(adata)
    print(f"  Found {len(bregma_values)} Bregma positions: {bregma_values}")
    for i, (section, bregma) in enumerate(zip(all_sections, bregma_values)):
        print(f"    [{i:2d}] Bregma {bregma:+.2f}: {section.n_obs:>5d} cells")
    print()

    # ------------------------------------------------------------------
    # 3. Train / held-out split
    # ------------------------------------------------------------------
    train_indices = []
    heldout_indices = []

    if holdout_every <= 0:
        train_indices = list(range(len(all_sections)))
    else:
        for i in range(len(all_sections)):
            # Never hold out the first or last section
            is_boundary = (i == 0 or i == len(all_sections) - 1)
            if not is_boundary and i % holdout_every == 0:
                heldout_indices.append(i)
            else:
                train_indices.append(i)

    train_sections = [all_sections[i] for i in train_indices]
    train_bregma = [bregma_values[i] for i in train_indices]
    heldout_sections = [all_sections[i] for i in heldout_indices]
    heldout_bregma = [bregma_values[i] for i in heldout_indices]

    print(f"  Train sections ({len(train_indices)}): "
          f"indices {train_indices}")
    print(f"  Held-out sections ({len(heldout_indices)}): "
          f"indices {heldout_indices}")
    print(f"  Train cells: {sum(s.n_obs for s in train_sections):,}")
    if heldout_sections:
        print(f"  Held-out cells: {sum(s.n_obs for s in heldout_sections):,}")
    print()

    # ------------------------------------------------------------------
    # 4. Normalise z-coordinates (shared scale across both splits)
    # ------------------------------------------------------------------
    all_bregma = bregma_values  # full range for normalisation
    z_min, z_max = min(all_bregma), max(all_bregma)
    z_range = z_max - z_min if z_max != z_min else 1.0

    train_z = [(b - z_min) / z_range for b in train_bregma]
    heldout_z = [(b - z_min) / z_range for b in heldout_bregma]

    print("  Z-coordinate mapping (Bregma -> normalised):")
    for b, z in zip(train_bregma, train_z):
        print(f"    train   {b:+.2f} -> {z:.3f}")
    for b, z in zip(heldout_bregma, heldout_z):
        print(f"    heldout {b:+.2f} -> {z:.3f}")
    print()

    # ------------------------------------------------------------------
    # 5. Preprocess (fit on train, transform both)
    # ------------------------------------------------------------------
    print("Preprocessing (fitting on train sections only)...")
    n_hvgs = min(150, all_sections[0].n_vars)
    n_pcs = min(30, n_hvgs - 1)

    train_tensors, heldout_tensors, norm_params = _preprocess_joint(
        train_sections, heldout_sections,
        train_z, heldout_z,
        n_hvgs=n_hvgs, n_pcs=n_pcs,
    )

    print(f"  HVGs: {n_hvgs}, PCs: {n_pcs}")
    print(f"  Train tensors:")
    for i, t in enumerate(train_tensors):
        print(f"    [{i}] Bregma {train_bregma[i]:+.2f}: {t.shape}")
    if heldout_tensors:
        print(f"  Held-out tensors:")
        for i, t in enumerate(heldout_tensors):
            print(f"    [{i}] Bregma {heldout_bregma[i]:+.2f}: {t.shape}")

    print(f"\nDone. {len(train_tensors)} train + {len(heldout_tensors)} "
          f"held-out sections ready.")

    return MerfishSplit(
        train_sections=train_sections,
        heldout_sections=heldout_sections,
        train_z=train_z,
        heldout_z=heldout_z,
        train_tensors=train_tensors,
        heldout_tensors=heldout_tensors,
        norm_params=norm_params,
        train_bregma=train_bregma,
        heldout_bregma=heldout_bregma,
    )


# ---------------------------------------------------------------------------
# Sinkhorn transport plans between consecutive slices
# ---------------------------------------------------------------------------

def compute_transport_plans(
    tensors: list[Tensor],
    blur: float = 0.05,
    n_iters: int = 100,
    cost_coord_weight: float = 1.0,
    cost_expr_weight: float = 1.0,
    device: str = "cpu",
) -> list[Tensor]:
    """Compute entropic OT couplings between every consecutive pair of slices.

    For each pair ``(tensors[k], tensors[k+1])`` this solves the static
    Schrödinger bridge problem: find the coupling ``pi`` in ``R^{N_k x N_{k+1}}``
    that minimises

        sum_{i,j} C_{ij} pi_{ij} + eps * KL(pi || mu x nu)

    where ``mu``, ``nu`` are uniform marginals and the cost matrix ``C`` is the
    squared Euclidean distance over the joint (spatial + expression) feature
    space, with configurable weighting between the two.

    Args:
        tensors: Preprocessed section tensors, each ``(N_i, 3 + n_pcs)``.
            Must be in z-order (consecutive in physical space).
        blur: Sinkhorn regularisation strength (epsilon = blur^2).
            Smaller values give sharper (more deterministic) couplings but
            need more iterations.  Larger values give smoother plans.
        n_iters: Number of Sinkhorn iterations.
        cost_coord_weight: Weight on the spatial (xyz) block of the cost.
        cost_expr_weight: Weight on the expression (PC) block of the cost.
        device: Device to run the computation on.

    Returns:
        List of ``len(tensors) - 1`` transport plan tensors.
        ``plans[k]`` has shape ``(N_k, N_{k+1})`` and satisfies:
        - rows sum to ``1 / N_k``   (source marginal)
        - columns sum to ``1 / N_{k+1}`` (target marginal)
    """
    plans: list[Tensor] = []

    for k in range(len(tensors) - 1):
        t_src = tensors[k].to(device)
        t_tgt = tensors[k + 1].to(device)

        # Build weighted feature vectors for cost computation
        coords_src = t_src[:, :3] * math.sqrt(cost_coord_weight)
        expr_src = t_src[:, 3:] * math.sqrt(cost_expr_weight)
        feat_src = torch.cat([coords_src, expr_src], dim=1)

        coords_tgt = t_tgt[:, :3] * math.sqrt(cost_coord_weight)
        expr_tgt = t_tgt[:, 3:] * math.sqrt(cost_expr_weight)
        feat_tgt = torch.cat([coords_tgt, expr_tgt], dim=1)

        pi = sinkhorn_transport_plan(feat_src, feat_tgt, blur=blur, n_iters=n_iters)
        plans.append(pi.cpu())

        # Transport cost for this pair: <C, pi>
        C = torch.cdist(feat_src, feat_tgt, p=2.0).pow(2)
        cost = (C * pi).sum().item()
        print(f"  Pair {k} -> {k+1}:  "
              f"{t_src.shape[0]:>5d} x {t_tgt.shape[0]:<5d}  "
              f"transport cost = {cost:.4f}")

    return plans


if __name__ == "__main__":
    split = load_merfish(holdout_every=3)

    print("\n" + "=" * 60)
    print("Computing Sinkhorn transport plans between consecutive "
          "training slices...")
    print("=" * 60)
    plans = compute_transport_plans(split.train_tensors)

    # Quick sanity check: marginals
    print("\nMarginal check (first plan):")
    pi = plans[0]
    row_sums = pi.sum(dim=1)
    col_sums = pi.sum(dim=0)
    N, M = pi.shape
    print(f"  Row sums:  mean={row_sums.mean():.6f}  "
          f"expected={1.0/N:.6f}  max_err={abs(row_sums - 1.0/N).max():.2e}")
    print(f"  Col sums:  mean={col_sums.mean():.6f}  "
          f"expected={1.0/M:.6f}  max_err={abs(col_sums - 1.0/M).max():.2e}")
