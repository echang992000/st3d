#!/usr/bin/env python3
"""
Reconstruct 3D spatial transcriptomic data from 2D slices using a
Schrödinger bridge solved via Iterative Proportional Fitting (IPF).

Pipeline:
  1. Load MERFISH data (squidpy), PCA-reduce gene expression
  2. Hold out interior z-planes as ground truth
  3. Compute Sinkhorn (entropic OT) couplings between consecutive observed slices
  4. Train time-conditioned drift networks via IPF: alternating forward/backward
     fitting on Brownian-bridge trajectories between OT-coupled endpoints
  5. Reconstruct held-out slices by Euler–Maruyama integration of the learned SDE
     with bidirectional blending
  6. Evaluate reconstructions (Sinkhorn distance, NN-MSE, Pearson correlation)
  7. Save comparison plots
"""

import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot as pot  # POT library
import scanpy as sc
import squidpy as sq
import torch
import torch.nn as nn
from geomloss import SamplesLoss
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    n_pca: int = 50
    sinkhorn_epsilon: float = 0.05
    sinkhorn_blur: float = 0.05
    sigma: float = 0.5  # Brownian diffusion coefficient
    n_ipf_iters: int = 5
    n_train_steps: int = 500
    batch_size: int = 512
    lr: float = 1e-3
    n_time_steps: int = 20  # Euler-Maruyama discretisation steps
    hidden_dim: int = 256
    seed: int = 42
    output_dir: str = "plots"


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 2. Data loading & preprocessing
# ---------------------------------------------------------------------------


def _create_synthetic_merfish(cfg: Config):
    """Create a synthetic MERFISH-like AnnData when the real dataset cannot be
    downloaded (e.g. in sandboxed / offline environments)."""
    import anndata as ad
    import pandas as pd

    print("  (Using synthetic MERFISH-like data as fallback)")
    rng = np.random.RandomState(cfg.seed)

    # Mimic real MERFISH: ~12 Bregma planes, 100-300 cells each, 161 genes
    z_planes = np.round(np.linspace(-0.29, 0.21, 12), 2)
    n_genes = 161
    gene_names = [f"gene_{i}" for i in range(n_genes)]

    all_X, all_bregma, all_spatial = [], [], []
    # Create a smooth latent gradient across z so slices are related
    base_profile = rng.rand(n_genes).astype(np.float32)
    for z in z_planes:
        n_cells = rng.randint(150, 300)
        # Expression = base + z-dependent shift + noise
        shift = 0.5 * np.sin(2 * np.pi * (z + 0.29) / 0.5) * rng.rand(n_genes).astype(np.float32)
        X = np.abs(
            base_profile[None, :] + shift[None, :] + 0.3 * rng.randn(n_cells, n_genes).astype(np.float32)
        )
        spatial = np.column_stack([
            rng.randn(n_cells) * 200 + 500,
            rng.randn(n_cells) * 200 + 500,
        ]).astype(np.float32)
        all_X.append(X)
        all_bregma.extend([z] * n_cells)
        all_spatial.append(spatial)

    X_full = np.vstack(all_X)
    obs = pd.DataFrame({"Bregma": all_bregma})
    adata = ad.AnnData(
        X=X_full,
        obs=obs,
        var=pd.DataFrame(index=gene_names),
    )
    adata.obsm["spatial"] = np.vstack(all_spatial)
    return adata


def load_and_preprocess(cfg: Config):
    """Return observed / held-out slice dicts, sorted z values, and the adata."""
    print("Loading MERFISH dataset …")
    try:
        adata = sq.datasets.merfish()
    except Exception as e:
        print(f"  Download failed: {e}")
        adata = _create_synthetic_merfish(cfg)

    # Normalise & PCA
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.tl.pca(adata, n_comps=cfg.n_pca)

    z_all = np.sort(adata.obs["Bregma"].unique())
    print(f"  Found {len(z_all)} z-planes: {z_all}")

    # Hold out every other *interior* slice (indices 1, 3, 5, … excluding
    # the first and last planes so flanking observed slices always exist).
    interior_idx = list(range(1, len(z_all) - 1))
    heldout_idx = set(interior_idx[::2])
    observed_z = [z for i, z in enumerate(z_all) if i not in heldout_idx]
    heldout_z = [z for i, z in enumerate(z_all) if i in heldout_idx]

    print(f"  Observed slices ({len(observed_z)}): {observed_z}")
    print(f"  Held-out slices ({len(heldout_z)}): {heldout_z}")

    def _slice_tensor(z):
        mask = adata.obs["Bregma"] == z
        return torch.tensor(adata.obsm["X_pca"][mask.values], dtype=torch.float32)

    observed_slices = {z: _slice_tensor(z) for z in observed_z}
    heldout_slices = {z: _slice_tensor(z) for z in heldout_z}

    # Also keep spatial coords for plotting
    spatial_coords = {}
    for z in z_all:
        mask = adata.obs["Bregma"] == z
        spatial_coords[z] = adata.obsm["spatial"][mask.values]

    for z, X in observed_slices.items():
        print(f"    z={z:+.2f}  →  {X.shape[0]} cells (observed)")
    for z, X in heldout_slices.items():
        print(f"    z={z:+.2f}  →  {X.shape[0]} cells (held-out)")

    return observed_slices, heldout_slices, observed_z, heldout_z, spatial_coords, adata


# ---------------------------------------------------------------------------
# 3. Sinkhorn transport plans
# ---------------------------------------------------------------------------


def compute_sinkhorn_coupling(X0: torch.Tensor, X1: torch.Tensor, epsilon: float):
    """Compute the entropic OT coupling π between two point clouds (CPU, numpy)."""
    X0_np = X0.numpy().astype(np.float64)
    X1_np = X1.numpy().astype(np.float64)
    C = pot.dist(X0_np, X1_np, metric="sqeuclidean")
    # Normalise cost for numerical stability
    C = C / C.max()
    a = np.ones(len(X0_np), dtype=np.float64) / len(X0_np)
    b = np.ones(len(X1_np), dtype=np.float64) / len(X1_np)
    pi = pot.sinkhorn(a, b, C, reg=epsilon, numItermax=1000, warn=False)
    return pi  # (n0, n1) numpy


def sample_coupling(pi: np.ndarray, X0: torch.Tensor, X1: torch.Tensor, batch_size: int):
    """Sample (x0, x1) pairs from the OT coupling π."""
    flat = pi.ravel()
    flat = flat / flat.sum()  # renormalise for safety
    indices = np.random.choice(len(flat), size=batch_size, p=flat)
    n1 = pi.shape[1]
    i_idx = indices // n1
    j_idx = indices % n1
    return X0[i_idx], X1[j_idx]


# ---------------------------------------------------------------------------
# 4. Time-conditioned drift network
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    """Map scalar t → vector of dimension `dim` using sinusoidal encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # t: (B, 1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t * freqs  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


class DriftNet(nn.Module):
    """Time-conditioned MLP: (x, t) → drift vector."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.proj_in = nn.Linear(state_dim, hidden_dim)
        self.block1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.block2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.proj_out = nn.Linear(hidden_dim, state_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: (B, state_dim)
        t: (B, 1)  — normalised time in [0, 1]
        """
        t_emb = self.time_emb(t)  # (B, hidden_dim)
        h = self.proj_in(x) + t_emb  # (B, hidden_dim)
        h = h + self.block1(h)  # residual
        h = h + self.block2(h)  # residual
        return self.proj_out(h)  # (B, state_dim)


# ---------------------------------------------------------------------------
# 5. Brownian bridge utilities
# ---------------------------------------------------------------------------


def sample_brownian_bridge(x0, x1, t, sigma):
    """Sample X_t from a Brownian bridge pinned at x0 (t=0) and x1 (t=1).

    X_t ~ N( (1-t)*x0 + t*x1 ,  σ²·t·(1-t)·I )
    """
    mu = (1.0 - t) * x0 + t * x1
    std = sigma * torch.sqrt(t * (1.0 - t) + 1e-8)
    return mu + std * torch.randn_like(mu)


def bridge_drift_forward(x_t, x1, t):
    """Analytical forward bridge drift: (x1 - x_t) / (1 - t)."""
    return (x1 - x_t) / (1.0 - t + 1e-6)


def bridge_drift_backward(x_t, x0, t):
    """Analytical backward bridge drift: (x0 - x_t) / t."""
    return (x0 - x_t) / (t + 1e-6)


# ---------------------------------------------------------------------------
# 6. IPF training loop
# ---------------------------------------------------------------------------


def _train_drift_one_direction(
    drift_net, optimizer, coupling, X0, X1, target_fn, cfg, desc="fwd"
):
    """Train a drift network for N_TRAIN_STEPS on Brownian-bridge targets.

    target_fn(x_t, x_anchor, t) → target drift vector
    For forward:  anchor = x1, target_fn = bridge_drift_forward
    For backward: anchor = x0, target_fn = bridge_drift_backward
    """
    drift_net.train()
    losses = []
    for step in range(cfg.n_train_steps):
        x0_batch, x1_batch = sample_coupling(coupling, X0, X1, cfg.batch_size)
        x0_batch = x0_batch.to(DEVICE)
        x1_batch = x1_batch.to(DEVICE)

        # Random time t ∈ (ε, 1-ε) to avoid singularities at endpoints
        t = torch.rand(cfg.batch_size, 1, device=DEVICE) * 0.98 + 0.01

        x_t = sample_brownian_bridge(x0_batch, x1_batch, t, cfg.sigma)

        # Determine anchor based on direction
        if desc == "fwd":
            v_target = target_fn(x_t, x1_batch, t)
        else:
            v_target = target_fn(x_t, x0_batch, t)

        v_pred = drift_net(x_t, t)
        loss = nn.functional.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def euler_maruyama_forward(drift_net, x0, t_end, sigma, n_steps):
    """Integrate the SDE forward from t=0 to t=t_end using Euler–Maruyama."""
    drift_net.eval()
    dt = t_end / n_steps
    x = x0.clone()
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full((x.shape[0], 1), i * dt, device=x.device)
            x = x + drift_net(x, t) * dt + sigma * math.sqrt(abs(dt)) * torch.randn_like(x)
    return x


def train_ipf(observed_slices, z_observed, cfg: Config):
    """Train IPF drift networks for every consecutive pair of observed slices."""
    drift_pairs = {}  # (z_left, z_right) → (drift_fwd, drift_bwd)

    for k in range(len(z_observed) - 1):
        z_left, z_right = z_observed[k], z_observed[k + 1]
        X0 = observed_slices[z_left]
        X1 = observed_slices[z_right]
        print(f"\n{'='*60}")
        print(f"Training IPF for slice pair z={z_left:+.2f} → z={z_right:+.2f}")
        print(f"  n_cells: {X0.shape[0]} → {X1.shape[0]}")
        print(f"{'='*60}")

        # Initialise drift networks
        drift_fwd = DriftNet(cfg.n_pca, cfg.hidden_dim).to(DEVICE)
        drift_bwd = DriftNet(cfg.n_pca, cfg.hidden_dim).to(DEVICE)
        opt_fwd = torch.optim.Adam(drift_fwd.parameters(), lr=cfg.lr)
        opt_bwd = torch.optim.Adam(drift_bwd.parameters(), lr=cfg.lr)

        # Initial Sinkhorn coupling
        print("  Computing initial Sinkhorn coupling …")
        coupling = compute_sinkhorn_coupling(X0, X1, cfg.sinkhorn_epsilon)

        for ipf_iter in range(cfg.n_ipf_iters):
            print(f"\n  IPF iteration {ipf_iter + 1}/{cfg.n_ipf_iters}")

            # --- Forward half-step ---
            loss_fwd = _train_drift_one_direction(
                drift_fwd, opt_fwd, coupling, X0, X1,
                bridge_drift_forward, cfg, desc="fwd",
            )
            print(f"    Forward  loss: {loss_fwd:.6f}")

            # --- Backward half-step ---
            loss_bwd = _train_drift_one_direction(
                drift_bwd, opt_bwd, coupling, X0, X1,
                bridge_drift_backward, cfg, desc="bwd",
            )
            print(f"    Backward loss: {loss_bwd:.6f}")

            # --- Re-coupling: push X0 forward with current drift, recompute OT ---
            if ipf_iter < cfg.n_ipf_iters - 1:
                X0_dev = X0.to(DEVICE)
                X1_pushed = euler_maruyama_forward(
                    drift_fwd, X0_dev, t_end=1.0, sigma=cfg.sigma,
                    n_steps=cfg.n_time_steps,
                )
                coupling = compute_sinkhorn_coupling(
                    X1_pushed.cpu(), X1, cfg.sinkhorn_epsilon
                )
                print("    Re-coupled.")

        drift_pairs[(z_left, z_right)] = (drift_fwd, drift_bwd)

    return drift_pairs


# ---------------------------------------------------------------------------
# 7. SDE reconstruction (Euler–Maruyama + bidirectional blending)
# ---------------------------------------------------------------------------


@torch.no_grad()
def reconstruct_slice(
    z_target, z_left, z_right, X_left, X_right,
    drift_fwd, drift_bwd, cfg: Config,
):
    """Reconstruct a held-out slice at z_target by bidirectional SDE integration."""
    drift_fwd.eval()
    drift_bwd.eval()

    t_star = (z_target - z_left) / (z_right - z_left)
    n_steps = cfg.n_time_steps

    # --- Forward integration: z_left → z_target ---
    dt_fwd = t_star / n_steps
    x_fwd = X_left.clone().to(DEVICE)
    for i in range(n_steps):
        t = torch.full((x_fwd.shape[0], 1), i * dt_fwd, device=DEVICE)
        x_fwd = (
            x_fwd
            + drift_fwd(x_fwd, t) * dt_fwd
            + cfg.sigma * math.sqrt(abs(dt_fwd)) * torch.randn_like(x_fwd)
        )

    # --- Backward integration: z_right → z_target ---
    dt_bwd = (1.0 - t_star) / n_steps
    x_bwd = X_right.clone().to(DEVICE)
    for i in range(n_steps):
        t = torch.full((x_bwd.shape[0], 1), 1.0 - i * dt_bwd, device=DEVICE)
        x_bwd = (
            x_bwd
            + drift_bwd(x_bwd, t) * dt_bwd
            + cfg.sigma * math.sqrt(abs(dt_bwd)) * torch.randn_like(x_bwd)
        )

    # --- Bidirectional blending ---
    # We may have different numbers of cells from left vs right.
    # Use Sinkhorn coupling to match them, then blend.
    n_fwd, n_bwd = x_fwd.shape[0], x_bwd.shape[0]
    if n_fwd == n_bwd:
        alpha = 1.0 - t_star
        X_recon = alpha * x_fwd + (1.0 - alpha) * x_bwd
    else:
        # Match via nearest-neighbour in PCA space, blend
        # Use the larger set as base, find NN in the smaller set
        if n_fwd >= n_bwd:
            base, other, w_base = x_fwd, x_bwd, 1.0 - t_star
        else:
            base, other, w_base = x_bwd, x_fwd, t_star

        dists = torch.cdist(base, other)
        nn_idx = dists.argmin(dim=1)
        X_recon = w_base * base + (1.0 - w_base) * other[nn_idx]

    return X_recon.cpu()


# ---------------------------------------------------------------------------
# 8. Visualisation
# ---------------------------------------------------------------------------


def plot_reconstructed_vs_ground_truth(
    reconstructed, heldout_slices, spatial_coords, output_dir="plots",
):
    """Save side-by-side scatter plots and distribution histograms."""
    os.makedirs(output_dir, exist_ok=True)

    for z in sorted(reconstructed.keys()):
        X_recon = reconstructed[z].numpy()
        X_true = heldout_slices[z].numpy()
        coords_true = spatial_coords.get(z)

        # --- Scatter plots (2×2) ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Top-left: ground truth in spatial coords, coloured by PC1
        if coords_true is not None and coords_true.shape[0] == X_true.shape[0]:
            sc_gt = axes[0, 0].scatter(
                coords_true[:, 0], coords_true[:, 1],
                c=X_true[:, 0], s=3, cmap="viridis", alpha=0.7,
            )
            axes[0, 0].set_title(f"Ground Truth (spatial) — PC1\nz = {z:+.2f}")
            plt.colorbar(sc_gt, ax=axes[0, 0], shrink=0.7)
        else:
            axes[0, 0].scatter(
                X_true[:, 0], X_true[:, 1],
                c=X_true[:, 0], s=3, cmap="viridis", alpha=0.7,
            )
            axes[0, 0].set_title(f"Ground Truth (PCA 1v2) — PC1\nz = {z:+.2f}")

        # Top-right: reconstruction in PCA space, coloured by PC1
        sc_rec = axes[0, 1].scatter(
            X_recon[:, 0], X_recon[:, 1],
            c=X_recon[:, 0], s=3, cmap="viridis", alpha=0.7,
        )
        axes[0, 1].set_title(f"Reconstructed (PCA 1v2) — PC1\nz = {z:+.2f}")
        plt.colorbar(sc_rec, ax=axes[0, 1], shrink=0.7)

        # Bottom-left: ground truth PCA space, coloured by PC2
        sc_gt2 = axes[1, 0].scatter(
            X_true[:, 0], X_true[:, 1],
            c=X_true[:, 1] if X_true.shape[1] > 1 else X_true[:, 0],
            s=3, cmap="coolwarm", alpha=0.7,
        )
        axes[1, 0].set_title(f"Ground Truth (PCA 1v2) — PC2\nz = {z:+.2f}")
        plt.colorbar(sc_gt2, ax=axes[1, 0], shrink=0.7)

        # Bottom-right: reconstruction PCA space, coloured by PC2
        sc_rec2 = axes[1, 1].scatter(
            X_recon[:, 0], X_recon[:, 1],
            c=X_recon[:, 1] if X_recon.shape[1] > 1 else X_recon[:, 0],
            s=3, cmap="coolwarm", alpha=0.7,
        )
        axes[1, 1].set_title(f"Reconstructed (PCA 1v2) — PC2\nz = {z:+.2f}")
        plt.colorbar(sc_rec2, ax=axes[1, 1], shrink=0.7)

        for ax in axes.ravel():
            ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        path = os.path.join(output_dir, f"slice_z{z:+.2f}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved scatter plot → {path}")

        # --- Distribution comparison (PC1 histogram) ---
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.hist(X_true[:, 0], bins=60, alpha=0.5, density=True, label="Ground Truth")
        ax2.hist(X_recon[:, 0], bins=60, alpha=0.5, density=True, label="Reconstructed")
        ax2.set_xlabel("PC1 value")
        ax2.set_ylabel("Density")
        ax2.set_title(f"PC1 Distribution — z = {z:+.2f}")
        ax2.legend()
        fig2.tight_layout()
        path2 = os.path.join(output_dir, f"distribution_z{z:+.2f}.png")
        fig2.savefig(path2, dpi=150)
        plt.close(fig2)
        print(f"  Saved distribution  → {path2}")


# ---------------------------------------------------------------------------
# 9. Evaluation metrics
# ---------------------------------------------------------------------------


def compute_sinkhorn_distance(X_recon: torch.Tensor, X_true: torch.Tensor, blur=0.05):
    """Sinkhorn divergence between two point clouds."""
    loss_fn = SamplesLoss("sinkhorn", p=2, blur=blur, scaling=0.5, debias=True)
    return loss_fn(X_recon.to(DEVICE), X_true.to(DEVICE)).item()


def compute_nn_mse(X_recon: torch.Tensor, X_true: torch.Tensor):
    """For each ground-truth cell, find NN in reconstruction and compute MSE."""
    nn_model = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn_model.fit(X_recon.numpy())
    dists, idx = nn_model.kneighbors(X_true.numpy())
    matched = X_recon.numpy()[idx.ravel()]
    mse = np.mean((X_true.numpy() - matched) ** 2)
    return float(mse)


def compute_correlation(X_recon: torch.Tensor, X_true: torch.Tensor):
    """Mean per-gene Pearson correlation (NN-matched pairs)."""
    nn_model = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn_model.fit(X_recon.numpy())
    _, idx = nn_model.kneighbors(X_true.numpy())
    matched = X_recon.numpy()[idx.ravel()]

    corrs = []
    for j in range(X_true.shape[1]):
        col_true = X_true[:, j].numpy()
        col_recon = matched[:, j]
        if np.std(col_true) < 1e-8 or np.std(col_recon) < 1e-8:
            continue
        r, _ = pearsonr(col_true, col_recon)
        corrs.append(r)
    return float(np.mean(corrs)) if corrs else 0.0


def evaluate(reconstructed, heldout_slices):
    """Compute all metrics for each held-out slice and print a table."""
    results = {}
    print(f"\n{'='*70}")
    print(f"{'Slice':>10} | {'Sinkhorn':>12} | {'NN-MSE':>12} | {'Correlation':>12}")
    print(f"{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for z in sorted(reconstructed.keys()):
        X_recon = reconstructed[z]
        X_true = heldout_slices[z]
        sk = compute_sinkhorn_distance(X_recon, X_true)
        mse = compute_nn_mse(X_recon, X_true)
        corr = compute_correlation(X_recon, X_true)
        results[z] = {"sinkhorn": sk, "nn_mse": mse, "correlation": corr}
        print(f"  z={z:+.2f} | {sk:12.6f} | {mse:12.6f} | {corr:12.4f}")
    print(f"{'='*70}")

    # Averages
    avg_sk = np.mean([v["sinkhorn"] for v in results.values()])
    avg_mse = np.mean([v["nn_mse"] for v in results.values()])
    avg_corr = np.mean([v["correlation"] for v in results.values()])
    print(f"  {'Mean':>7} | {avg_sk:12.6f} | {avg_mse:12.6f} | {avg_corr:12.4f}")
    print()
    return results


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("3D Spatial Transcriptomic Reconstruction")
    print("Method: Schrödinger Bridge via IPF")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Seed everything
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)

    # --- Load data ---
    (
        observed_slices, heldout_slices,
        z_observed, z_heldout,
        spatial_coords, adata,
    ) = load_and_preprocess(CFG)

    if len(z_heldout) == 0:
        print("No slices to hold out — need at least 3 z-planes. Exiting.")
        return

    # --- Train IPF ---
    drift_pairs = train_ipf(observed_slices, z_observed, CFG)

    # --- Reconstruct held-out slices ---
    reconstructed = {}
    print(f"\n{'='*60}")
    print("Reconstructing held-out slices …")
    print(f"{'='*60}")
    for z_target in z_heldout:
        # Find flanking observed slices
        left_z = max(z for z in z_observed if z < z_target)
        right_z = min(z for z in z_observed if z > z_target)
        print(f"  z={z_target:+.2f}  flanked by [{left_z:+.2f}, {right_z:+.2f}]")

        key = (left_z, right_z)
        if key not in drift_pairs:
            # The flanking pair may span multiple observed slices;
            # find the pair that contains z_target.
            for (zl, zr), _ in drift_pairs.items():
                if zl <= z_target <= zr:
                    key = (zl, zr)
                    break

        drift_fwd, drift_bwd = drift_pairs[key]
        X_left = observed_slices[key[0]]
        X_right = observed_slices[key[1]]

        X_recon = reconstruct_slice(
            z_target, key[0], key[1], X_left, X_right,
            drift_fwd, drift_bwd, CFG,
        )
        reconstructed[z_target] = X_recon
        print(f"    → reconstructed {X_recon.shape[0]} cells")

    # --- Visualise ---
    print(f"\nSaving comparison plots to '{CFG.output_dir}/' …")
    plot_reconstructed_vs_ground_truth(
        reconstructed, heldout_slices, spatial_coords, output_dir=CFG.output_dir,
    )

    # --- Evaluate ---
    evaluate(reconstructed, heldout_slices)

    print("Done.")


if __name__ == "__main__":
    main()
