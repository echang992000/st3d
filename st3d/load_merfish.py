"""
Load the MERFISH dataset from squidpy and prepare it for the st3d pipeline.

Uses the pre-processed MERFISH dataset from Moffitt et al. (2018),
which contains ~73,655 cells across 161 genes from serial sections of the
mouse hypothalamus. Each section corresponds to a distinct Bregma coordinate,
providing the z-axis depth needed for 3D reconstruction.

Reference:
    Moffitt et al., "Molecular, spatial, and functional single-cell profiling
    of the hypothalamic preoptic region", Science (2018).
    https://doi.org/10.1126/science.aau5324
"""

from __future__ import annotations

import sys
import os

import numpy as np
import squidpy as sq

# Make st3d importable when running this script directly from the st3d/ dir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from st3d.data import preprocess_sections


def load_merfish():
    """Load and explore the MERFISH dataset, then preprocess for st3d.

    Returns:
        tuple: (sections, z_coords, tensors, norm_params)
            - sections: list of AnnData, one per Bregma slice
            - z_coords: list of float z-depths (normalised Bregma values)
            - tensors: list of preprocessed torch Tensors
            - norm_params: NormParams from preprocessing
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
    # 2. Identify serial sections via the Bregma coordinate
    # ------------------------------------------------------------------
    # The MERFISH dataset stores the anterior-posterior position in the
    # "Bregma" column of adata.obs. Each unique Bregma value corresponds
    # to a physical tissue section.
    bregma_col = "Bregma"
    if bregma_col not in adata.obs.columns:
        # Fall back: look for any column that could indicate section identity
        candidates = [c for c in adata.obs.columns if "bregma" in c.lower()]
        if not candidates:
            raise KeyError(
                f"Could not find a Bregma column in obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )
        bregma_col = candidates[0]
        print(f"  Using column '{bregma_col}' for z-coordinates")

    bregma_values = sorted(adata.obs[bregma_col].unique())
    print(f"  Found {len(bregma_values)} unique Bregma positions: {bregma_values}")
    print()

    # ------------------------------------------------------------------
    # 3. Split into per-section AnnData objects
    # ------------------------------------------------------------------
    sections = []
    z_coords = []

    for bregma in bregma_values:
        mask = adata.obs[bregma_col] == bregma
        section = adata[mask].copy()

        # Ensure spatial coordinates are in obsm["spatial"]
        if "spatial" not in section.obsm:
            # Try to build spatial from X/Y obs columns
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
                    break
            else:
                raise KeyError(
                    f"No spatial coordinates found. "
                    f"obs columns: {list(section.obs.columns)}, "
                    f"obsm keys: {list(section.obsm.keys())}"
                )

        # Ensure dense X matrix
        from scipy.sparse import issparse
        if issparse(section.X):
            section.X = section.X.toarray()
        section.X = section.X.astype(np.float32)

        sections.append(section)
        z_coords.append(float(bregma))

        print(f"  Bregma {bregma:+.2f}: {section.n_obs} cells")

    print(f"\n  Total: {sum(s.n_obs for s in sections)} cells across "
          f"{len(sections)} sections")
    print()

    # ------------------------------------------------------------------
    # 4. Normalise z-coordinates to [0, 1] range
    # ------------------------------------------------------------------
    z_min, z_max = min(z_coords), max(z_coords)
    z_range = z_max - z_min if z_max != z_min else 1.0
    z_coords_norm = [(z - z_min) / z_range for z in z_coords]

    print("  Z-coordinate mapping (Bregma -> normalised):")
    for bregma, z_norm in zip(z_coords, z_coords_norm):
        print(f"    {bregma:+.2f} -> {z_norm:.3f}")
    print()

    # ------------------------------------------------------------------
    # 5. Preprocess for st3d
    # ------------------------------------------------------------------
    print("Preprocessing for st3d pipeline...")
    n_hvgs = min(150, sections[0].n_vars)  # MERFISH has only 161 genes
    n_pcs = min(30, n_hvgs - 1)

    tensors, norm_params = preprocess_sections(
        sections,
        z_coords=z_coords_norm,
        n_hvgs=n_hvgs,
        n_pcs=n_pcs,
    )

    print(f"  HVGs selected: {n_hvgs}")
    print(f"  PCs: {n_pcs}")
    for i, t in enumerate(tensors):
        print(f"  Section {i} (Bregma {bregma_values[i]:+.2f}): "
              f"tensor shape {t.shape}")

    print("\nDone. Data is ready for st3d training.")

    return sections, z_coords_norm, tensors, norm_params


if __name__ == "__main__":
    sections, z_coords, tensors, norm_params = load_merfish()
