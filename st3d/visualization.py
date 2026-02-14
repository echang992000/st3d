"""
Visualization utilities for st3d.

All plotting uses Plotly for interactive 3D rendering and Matplotlib for
static 2D plots (training curves).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import anndata as ad

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# 3D scatter of reconstructed volume
# ---------------------------------------------------------------------------

def plot_sections_3d(
    adata: ad.AnnData,
    color_by: str | None = None,
    point_size: float = 1.5,
    opacity: float = 0.6,
    title: str = "3D Spatial Transcriptomics Volume",
    width: int = 900,
    height: int = 700,
) -> "go.Figure":
    """Interactive 3D scatter plot of an AnnData with ``obsm["spatial_3d"]``.

    Args:
        adata: AnnData produced by ``reconstruct_volume`` (needs
            ``.obsm["spatial_3d"]``).
        color_by: Gene name (column in ``.var_names``), ``.obs`` column name,
            or ``None`` for uniform colour.  If a gene name is given, ``.X``
            is used for colour values.
        point_size: Marker size.
        opacity: Marker opacity.
        title: Plot title.

    Returns:
        A Plotly ``Figure``.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for 3D visualisation")

    coords = adata.obsm["spatial_3d"]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    color = None
    colorbar_title = ""
    if color_by is not None:
        if color_by in adata.var_names:
            idx = list(adata.var_names).index(color_by)
            color = np.asarray(adata.X[:, idx]).flatten()
            colorbar_title = color_by
        elif color_by in adata.obs.columns:
            color = adata.obs[color_by].values
            colorbar_title = color_by

    marker_kwargs: dict = dict(size=point_size, opacity=opacity)
    if color is not None:
        marker_kwargs["color"] = color
        marker_kwargs["colorscale"] = "Viridis"
        marker_kwargs["colorbar"] = dict(title=colorbar_title)

    fig = go.Figure(
        data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=marker_kwargs,
        )]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (depth)",
            aspectmode="data",
            dragmode="orbit",
        ),
        width=width,
        height=height,
    )
    return fig


# ---------------------------------------------------------------------------
# Section comparison (observed vs interpolated)
# ---------------------------------------------------------------------------

def plot_interpolation(
    adata: ad.AnnData,
    gene: str,
    n_sections: int = 5,
    point_size: float = 3.0,
    width: int = 1200,
    height: int = 350,
) -> "go.Figure":
    """Side-by-side 2D scatter plots of selected z-layers coloured by a gene.

    Picks ``n_sections`` evenly-spaced z-values and plots the XY scatter for
    each, colouring by the given gene.

    Args:
        adata: Reconstructed volume AnnData.
        gene: Gene name to colour by.
        n_sections: Number of z-layers to display.
        point_size: Marker size.

    Returns:
        Plotly Figure with subplots.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for visualisation")

    z_vals = adata.obs["z"].values
    unique_z = np.sort(np.unique(np.round(z_vals, decimals=4)))
    chosen_idx = np.linspace(0, len(unique_z) - 1, n_sections, dtype=int)
    chosen_z = unique_z[chosen_idx]

    gene_idx = list(adata.var_names).index(gene)
    expr_all = np.asarray(adata.X[:, gene_idx]).flatten()

    fig = make_subplots(
        rows=1, cols=n_sections,
        subplot_titles=[f"z = {z:.3f}" for z in chosen_z],
    )

    vmin, vmax = float(np.nanpercentile(expr_all, 2)), float(np.nanpercentile(expr_all, 98))

    for col, z_target in enumerate(chosen_z, start=1):
        mask = np.abs(z_vals - z_target) < (np.diff(unique_z).min() / 2 if len(unique_z) > 1 else 0.01)
        xy = adata.obsm["spatial_3d"][mask]
        expr = expr_all[mask]

        fig.add_trace(
            go.Scatter(
                x=xy[:, 0], y=xy[:, 1],
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=expr,
                    colorscale="Viridis",
                    cmin=vmin,
                    cmax=vmax,
                    showscale=(col == n_sections),
                    colorbar=dict(title=gene) if col == n_sections else None,
                ),
                showlegend=False,
            ),
            row=1, col=col,
        )
        fig.update_xaxes(scaleanchor=f"y{col}", scaleratio=1, row=1, col=col)

    fig.update_layout(
        title=f"Interpolated sections coloured by {gene}",
        width=width,
        height=height,
    )
    return fig


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | None = None,
) -> None:
    """Plot training loss curves using Matplotlib.

    Args:
        history: Dict returned by ``Trainer.fit``.
        save_path: If set, save the figure to this path.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib is required for training history plots")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    axes[0].plot(history["total"], label="Total")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total Loss")
    axes[0].legend()

    # Components
    axes[1].plot(history["spatial"], label="Spatial (Sinkhorn)", color="tab:blue")
    axes[1].plot(history["expression"], label="Expression (NN-MSE)", color="tab:orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Components")
    axes[1].legend()

    # Expression weight schedule
    axes[2].plot(history["w_expr"], label="w_expr", color="tab:green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Weight")
    axes[2].set_title("Curriculum Schedule")
    axes[2].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
