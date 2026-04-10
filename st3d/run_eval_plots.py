"""
Run the full Schrödinger bridge evaluation pipeline and save visualisations
as PNG files.

Uses synthetic MERFISH-like sections (multiple cell types with spatially-
varying expression across serial z-planes) so the pipeline can run without
network access.

Outputs saved to st3d/notebooks/figures/
"""

from __future__ import annotations

import os
import sys
import math

import numpy as np
import anndata as ad
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from scipy.sparse import issparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from st3d.data import NormParams, denormalize_coords
from st3d.bridge import IPFTrainer, IPFConfig, SchrodingerBridgeModel
from st3d.losses import sinkhorn_transport_plan
from st3d.evaluate import predict_at_z, evaluate_heldout, print_summary
from st3d.load_merfish import compute_transport_plans

np.random.seed(42)
torch.manual_seed(42)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "notebooks", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ===================================================================
# 1.  Build synthetic MERFISH-like sections with cell types
# ===================================================================
print("=" * 60)
print("1.  Building synthetic sections with cell types")
print("=" * 60)

CELL_TYPES = [
    "Excitatory",
    "Inhibitory",
    "Astrocyte",
    "OD Mature",
    "Endothelial",
    "Microglia",
]

# Spatial centres for each cell type (shift smoothly along z)
TYPE_CENTRES = {
    "Excitatory":  np.array([3.0, 5.0]),
    "Inhibitory":  np.array([7.0, 5.0]),
    "Astrocyte":   np.array([5.0, 2.0]),
    "OD Mature":   np.array([5.0, 8.0]),
    "Endothelial": np.array([2.0, 8.0]),
    "Microglia":   np.array([8.0, 2.0]),
}

# Per-type expression signature (each type has a distinct mean expression)
N_GENES = 200
TYPE_SIGNATURES = {}
rng = np.random.RandomState(7)
for ct in CELL_TYPES:
    TYPE_SIGNATURES[ct] = rng.randn(N_GENES).astype(np.float32) * 2.0


def make_section(z: float, n_per_type: int = 80) -> ad.AnnData:
    """Build one section with spatially clustered cell types."""
    all_xy = []
    all_expr = []
    all_labels = []

    for ct in CELL_TYPES:
        centre = TYPE_CENTRES[ct] + np.array([z * 1.5, z * 0.5])
        xy = centre + np.random.randn(n_per_type, 2).astype(np.float32) * 1.2
        # Expression: type signature + z-dependent shift + noise
        sig = TYPE_SIGNATURES[ct]
        expr = (
            sig[None, :]
            + z * np.random.randn(1, N_GENES).astype(np.float32) * 0.3
            + np.random.randn(n_per_type, N_GENES).astype(np.float32) * 0.5
        )
        expr = np.abs(expr) * 5  # non-negative counts-like

        all_xy.append(xy)
        all_expr.append(expr.astype(np.float32))
        all_labels.extend([ct] * n_per_type)

    xy = np.concatenate(all_xy, axis=0)
    X = np.concatenate(all_expr, axis=0)

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"Cell_class": all_labels}),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(N_GENES)]),
    )
    adata.obsm["spatial"] = xy
    return adata


# 7 sections at evenly-spaced z depths
ALL_Z = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0]
all_sections = [make_section(z, n_per_type=80) for z in ALL_Z]

for i, (sec, z) in enumerate(zip(all_sections, ALL_Z)):
    print(f"  [{i}] z={z:.2f}  {sec.n_obs} cells  "
          f"types: {sec.obs['Cell_class'].nunique()}")

# ===================================================================
# 2.  Train / held-out split
# ===================================================================
print()
print("=" * 60)
print("2.  Train / held-out split")
print("=" * 60)

# Hold out indices 2 and 5 (z=0.30, z=0.75)
train_idx = [0, 1, 3, 4, 6]
heldout_idx = [2, 5]

train_sections_raw = [all_sections[i] for i in train_idx]
heldout_sections_raw = [all_sections[i] for i in heldout_idx]

train_z_raw = [ALL_Z[i] for i in train_idx]
heldout_z_raw = [ALL_Z[i] for i in heldout_idx]

print(f"  Train:   indices {train_idx}  z={train_z_raw}")
print(f"  Heldout: indices {heldout_idx}  z={heldout_z_raw}")

CELLTYPE_COL = "Cell_class"

# ===================================================================
# 3.  Preprocess (joint, fit on train only)
# ===================================================================
print()
print("=" * 60)
print("3.  Preprocessing")
print("=" * 60)

import scanpy as sc

# --- fit on train ---
combined = ad.concat(train_sections_raw, label="section",
                     keys=list(range(len(train_sections_raw))))
if issparse(combined.X):
    combined.X = combined.X.toarray()
combined.X = combined.X.astype(np.float32)
sc.pp.normalize_total(combined, target_sum=1e4)
sc.pp.log1p(combined)
sc.pp.highly_variable_genes(combined, n_top_genes=min(150, combined.shape[1]),
                            flavor="seurat")
hvg_mask = combined.var["highly_variable"].values
gene_names = list(combined.var_names[hvg_mask])
n_hvgs = sum(hvg_mask)
n_pcs = min(30, n_hvgs - 1)

pca = PCA(n_components=n_pcs)
expr_train = combined.X[:, hvg_mask]
pcs_train = pca.fit_transform(expr_train).astype(np.float32)

pc_min = pcs_train.min(axis=0)
pc_max = pcs_train.max(axis=0)
pc_scale = pc_max - pc_min
pc_scale[pc_scale == 0] = 1.0

all_xy = np.concatenate(
    [s.obsm["spatial"][:, :2].astype(np.float32) for s in train_sections_raw])
xy_min = all_xy.min(axis=0)
width = all_xy.max(axis=0) - xy_min
xy_scale = float(max(width[0], width[1]))
if xy_scale == 0:
    xy_scale = 1.0

norm_params = NormParams(
    xy_min=xy_min, xy_scale=xy_scale, pca=pca,
    gene_names=gene_names, n_pcs=n_pcs,
    pc_min=pc_min, pc_scale=pc_scale,
)


def sections_to_tensors(sections, z_coords):
    tensors = []
    for sec, z in zip(sections, z_coords):
        X = sec.X.copy().astype(np.float32)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X = X / row_sums * 1e4
        X = np.log1p(X)
        X_hvg = X[:, hvg_mask]
        pcs = pca.transform(X_hvg).astype(np.float32)
        pcs = (pcs - pc_min) / pc_scale
        xy = sec.obsm["spatial"][:, :2].astype(np.float32)
        xy_n = (xy - xy_min) / xy_scale
        z_col = np.full((xy.shape[0], 1), z, dtype=np.float32)
        state = np.concatenate([xy_n, z_col, pcs], axis=1)
        tensors.append(torch.from_numpy(state))
    return tensors


train_tensors = sections_to_tensors(train_sections_raw, train_z_raw)
heldout_tensors = sections_to_tensors(heldout_sections_raw, heldout_z_raw)

print(f"  HVGs: {n_hvgs}, PCs: {n_pcs}")
for i, t in enumerate(train_tensors):
    print(f"  Train [{i}] z={train_z_raw[i]:.2f}: {t.shape}")
for i, t in enumerate(heldout_tensors):
    print(f"  Heldout [{i}] z={heldout_z_raw[i]:.2f}: {t.shape}")

# ===================================================================
# 4.  Compute OT couplings
# ===================================================================
print()
print("=" * 60)
print("4.  Computing OT transport plans")
print("=" * 60)

plans = compute_transport_plans(train_tensors, blur=0.05, n_iters=100)

# ===================================================================
# 5.  Train Schrödinger bridge (IPF)
# ===================================================================
print()
print("=" * 60)
print("5.  Training IPF")
print("=" * 60)

cfg = IPFConfig(
    hidden_dim=128,
    n_blocks=3,
    time_embed_dim=32,
    sigma=0.1,
    batch_size=256,
    n_ipf_iters=5,
    steps_per_half=300,
    sde_steps=30,
    ot_blur=0.05,
    ot_iters=100,
    lr=1e-3,
)

trainer = IPFTrainer(train_tensors, plans, config=cfg, device="cpu")
print(f"  Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
history = trainer.fit(verbose=True)

# ===================================================================
# 6.  FIGURE 1: Training loss curves
# ===================================================================
print()
print("=" * 60)
print("6.  Saving training loss curves")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["fwd_loss"], linewidth=0.5)
axes[0].set_title("Forward bridge loss")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("MSE")
axes[0].set_yscale("log")

axes[1].plot(history["bwd_loss"], linewidth=0.5, color="tab:orange")
axes[1].set_title("Backward bridge loss")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("MSE")
axes[1].set_yscale("log")

fig.tight_layout()
path = os.path.join(FIGURES_DIR, "training_loss.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# ===================================================================
# 7.  Predict held-out slices
# ===================================================================
print()
print("=" * 60)
print("7.  Predicting held-out slices")
print("=" * 60)

model = trainer.model
model.eval()

predictions = []

for h_idx, (h_tensor, h_z) in enumerate(zip(heldout_tensors, heldout_z_raw)):
    left_idx, right_idx = None, None
    for k, tz in enumerate(train_z_raw):
        if tz <= h_z:
            left_idx = k
        if tz >= h_z and right_idx is None:
            right_idx = k

    if left_idx is None or right_idx is None or left_idx == right_idx:
        print(f"  [SKIP] Held-out {h_idx}")
        continue

    z_left = train_z_raw[left_idx]
    z_right = train_z_raw[right_idx]
    alpha = (h_z - z_left) / (z_right - z_left)

    pred = predict_at_z(
        model,
        train_tensors[left_idx],
        train_tensors[right_idx],
        alpha=alpha,
        sigma=cfg.sigma,
        sde_steps=50,
        device=torch.device("cpu"),
    )

    predictions.append((pred.cpu(), h_tensor, h_z, h_idx))
    print(f"  Held-out {h_idx}: z={h_z:.2f}  alpha={alpha:.3f}  "
          f"pred={pred.shape[0]}  true={h_tensor.shape[0]}")

# ===================================================================
# 8.  Build colour palette
# ===================================================================
all_types = sorted(CELL_TYPES)
cmap = plt.cm.get_cmap("tab10", len(all_types))
PALETTE = {ct: cmap(i) for i, ct in enumerate(all_types)}

# ===================================================================
# 9.  FIGURE 2: Cell-type spatial maps (ground truth vs reconstructed)
# ===================================================================
print()
print("=" * 60)
print("8.  Saving cell-type spatial maps")
print("=" * 60)


def plot_celltype_spatial(ax, xy, labels, palette, title, point_size=6):
    for ct in sorted(palette.keys()):
        mask = np.array(labels) == ct
        if mask.sum() == 0:
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=[palette[ct]], s=point_size, alpha=0.7,
                   label=ct, rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")


n_plots = len(predictions)
fig, axes = plt.subplots(n_plots, 2, figsize=(14, 6 * n_plots), squeeze=False)

for row, (pred_tensor, true_tensor, h_z, h_idx) in enumerate(predictions):
    gt_section = heldout_sections_raw[h_idx]
    gt_xy = gt_section.obsm["spatial"][:, :2]
    gt_labels = gt_section.obs[CELLTYPE_COL].values

    plot_celltype_spatial(
        axes[row, 0], gt_xy, gt_labels, PALETTE,
        title=f"Ground Truth  (z={h_z:.2f},  {len(gt_labels)} cells)",
    )

    # Denormalise predicted xy
    pred_xy_norm = pred_tensor[:, :2].numpy()
    pred_xy = denormalize_coords(
        np.column_stack([pred_xy_norm, np.zeros(len(pred_xy_norm))]),
        norm_params.xy_min, norm_params.xy_scale,
    )[:, :2]

    # Transfer cell-type labels from ground truth via spatial NN
    dists = torch.cdist(
        torch.from_numpy(pred_xy).float(),
        torch.from_numpy(gt_xy.astype(np.float32)),
    )
    nn_idx = dists.argmin(dim=1).numpy()
    pred_labels = gt_labels[nn_idx]

    plot_celltype_spatial(
        axes[row, 1], pred_xy, pred_labels, PALETTE,
        title=f"Reconstructed  (z={h_z:.2f},  {len(pred_labels)} cells)",
    )

handles = [mpatches.Patch(color=PALETTE[ct], label=ct) for ct in sorted(PALETTE.keys())]
fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
           fontsize=9, title=CELLTYPE_COL, title_fontsize=10)
fig.suptitle("Cell-type spatial maps: Ground Truth vs. Reconstructed",
             fontsize=14, y=1.01)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, "celltype_spatial_maps.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# ===================================================================
# 10.  FIGURE 3: Cell-type composition bar charts
# ===================================================================
print()
print("=" * 60)
print("9.  Saving cell-type composition bar charts")
print("=" * 60)

for pred_tensor, true_tensor, h_z, h_idx in predictions:
    gt_section = heldout_sections_raw[h_idx]
    gt_labels = gt_section.obs[CELLTYPE_COL].values
    gt_xy = gt_section.obsm["spatial"][:, :2]

    pred_xy_norm = pred_tensor[:, :2].numpy()
    pred_xy = denormalize_coords(
        np.column_stack([pred_xy_norm, np.zeros(len(pred_xy_norm))]),
        norm_params.xy_min, norm_params.xy_scale,
    )[:, :2]
    dists = torch.cdist(
        torch.from_numpy(pred_xy).float(),
        torch.from_numpy(gt_xy.astype(np.float32)),
    )
    nn_idx = dists.argmin(dim=1).numpy()
    pred_labels = gt_labels[nn_idx]

    types = sorted(PALETTE.keys())
    gt_frac = np.array([np.mean(gt_labels == ct) for ct in types])
    pred_frac = np.array([np.mean(pred_labels == ct) for ct in types])

    mask = (gt_frac > 0.01) | (pred_frac > 0.01)
    types_show = [t for t, m in zip(types, mask) if m]
    gt_show = gt_frac[mask]
    pred_show = pred_frac[mask]

    x = np.arange(len(types_show))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(types_show) * 1.0), 4))
    ax.bar(x - width / 2, gt_show, width, label="Ground Truth",
           color=[PALETTE[t] for t in types_show], edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, pred_show, width, label="Reconstructed",
           color=[PALETTE[t] for t in types_show], edgecolor="black", linewidth=0.5,
           alpha=0.5, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(types_show, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of cells")
    ax.set_title(f"Cell-type composition  (z={h_z:.2f})")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f"celltype_composition_z{h_z:.2f}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# ===================================================================
# 11.  Quantitative evaluation
# ===================================================================
print()
print("=" * 60)
print("10. Quantitative evaluation")
print("=" * 60)

results = evaluate_heldout(
    trainer.model,
    train_tensors=train_tensors,
    train_z=train_z_raw,
    heldout_tensors=heldout_tensors,
    heldout_z=heldout_z_raw,
    sigma=cfg.sigma,
    sde_steps=50,
    sinkhorn_blur=0.05,
    sinkhorn_iters=100,
    device="cpu",
)

agg = print_summary(results)

# ===================================================================
# 12.  FIGURE 4: Per-PC Pearson correlation
# ===================================================================
print()
print("=" * 60)
print("11. Saving per-PC Pearson plot")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 4))
for m in results:
    ax.plot(m.pearson_per_pc, marker="o", markersize=3, linewidth=1,
            label=f"z={m.z:.2f}")
ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Pearson r")
ax.set_title("Per-PC Pearson correlation (predicted vs. ground truth)")
ax.legend(fontsize=9)
fig.tight_layout()
path = os.path.join(FIGURES_DIR, "pearson_per_pc.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# ===================================================================
# 13.  FIGURE 5: Metrics bar chart
# ===================================================================
print()
print("=" * 60)
print("12. Saving metrics bar chart")
print("=" * 60)

if len(results) > 1:
    labels = [f"z={m.z:.2f}" for m in results]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    vals = [m.sinkhorn_spatial for m in results]
    axes[0].bar(x, vals, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Sinkhorn distance (xy)")

    vals = [m.nn_mse_expr for m in results]
    axes[1].bar(x, vals, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("NN-MSE (expression PCs)")

    vals = [m.pearson_mean for m in results]
    axes[2].bar(x, vals, color="mediumseagreen")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Mean Pearson r")
    axes[2].set_ylim([-0.1, 1.0])

    fig.suptitle("Evaluation metrics per held-out slice", fontsize=13)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "metrics_bar_chart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

print()
print("=" * 60)
print("Done. All figures saved to:", FIGURES_DIR)
print("=" * 60)
