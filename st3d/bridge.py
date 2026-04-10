"""
Schrödinger Bridge via Iterative Proportional Fitting (IPF).

Learns a time-conditioned neural drift v(x, t) that transports cells between
consecutive tissue sections.  Training uses the classical IPF algorithm:

1.  Initialise couplings with entropic OT (from ``compute_transport_plans``).
2.  Repeat for K IPF iterations:
    a.  **Forward half-step** — sample ``(x_0, x_1) ~ pi``, draw Brownian
        bridge points ``x_t``, regress forward drift to the bridge velocity
        ``(x_1 - x_t) / (1 - t)``.
    b.  **Re-couple (forward)** — simulate the learned forward SDE from the
        source marginal, compute new OT coupling between the pushed points
        and the true target.
    c.  **Backward half-step** — same bridge regression in the reverse
        direction with target ``(x_0 - x_t) / t``.
    d.  **Re-couple (backward)** — simulate backward SDE from the target,
        OT-couple the pulled points to the true source.

The alternating projections converge to the Schrödinger bridge: the
stochastic process closest (in KL) to Brownian motion whose marginals
match the observed sections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from st3d.losses import sinkhorn_transport_plan


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar time ``t in [0, 1]`` to a sinusoidal positional encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# Residual MLP block
# ---------------------------------------------------------------------------

class ResidualMLPBlock(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Time-conditioned drift network
# ---------------------------------------------------------------------------

class TimeConditionedDriftNet(nn.Module):
    """Predicts velocity ``v(x, t)`` for the state ``x`` at bridge time ``t``.

    Architecture::

        [state || sinusoidal_time_embed]
              |
        Linear -> hidden_dim, SiLU
              |
        ResidualMLPBlock  x  n_blocks
              |
        Linear -> state_dim   (velocity output)
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 32,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.input_proj = nn.Linear(state_dim + time_embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: State ``(B, state_dim)``.
            t: Time ``(B,)`` in ``[0, 1]``.

        Returns:
            Velocity ``(B, state_dim)``.
        """
        t_emb = self.time_embed(t)                          # (B, time_embed_dim)
        h = F.silu(self.input_proj(torch.cat([x, t_emb], dim=-1)))
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Schrödinger Bridge model
# ---------------------------------------------------------------------------

class SchrodingerBridgeModel(nn.Module):
    """Wraps forward and backward time-conditioned drift networks."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 32,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.forward_drift = TimeConditionedDriftNet(
            state_dim, hidden_dim, time_embed_dim, n_blocks,
        )
        self.backward_drift = TimeConditionedDriftNet(
            state_dim, hidden_dim, time_embed_dim, n_blocks,
        )

    def save(self, path: str) -> None:
        torch.save({
            "state_dim": self.state_dim,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "SchrodingerBridgeModel":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(state_dim=ckpt["state_dim"])
        model.load_state_dict(ckpt["state_dict"])
        return model


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def sample_ot_pairs(
    plan: Tensor,
    x_source: Tensor,
    x_target: Tensor,
    n_samples: int,
) -> tuple[Tensor, Tensor]:
    """Draw ``(x_0, x_1)`` pairs from entropic OT coupling ``plan``.

    Sampling is proportional to the coupling weights, with replacement.
    """
    flat = plan.flatten()
    # Ensure exact non-negativity for multinomial
    flat = flat.clamp(min=0.0)
    indices = torch.multinomial(flat, n_samples, replacement=True)
    src_idx = indices.div(plan.shape[1], rounding_mode="trunc")
    tgt_idx = indices % plan.shape[1]
    return x_source[src_idx], x_target[tgt_idx]


def brownian_bridge_sample(
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
    sigma: float,
) -> Tensor:
    """Sample from the Brownian bridge at time ``t`` conditioned on endpoints.

    ``x_t = (1 - t) x_0 + t x_1 + sigma sqrt(t (1-t)) z``

    Args:
        x0: Source states ``(B, D)``.
        x1: Target states ``(B, D)``.
        t:  Bridge times ``(B,)`` in ``(0, 1)``.
        sigma: Diffusion coefficient.

    Returns:
        Bridge samples ``(B, D)``.
    """
    t_ = t[:, None]  # (B, 1)
    mean = (1 - t_) * x0 + t_ * x1
    std = sigma * torch.sqrt(t_ * (1 - t_))
    return mean + std * torch.randn_like(mean)


# ---------------------------------------------------------------------------
# Euler–Maruyama SDE simulation
# ---------------------------------------------------------------------------

@torch.no_grad()
def simulate_sde(
    drift_net: TimeConditionedDriftNet,
    x_init: Tensor,
    sigma: float,
    n_steps: int,
    forward: bool = True,
) -> Tensor:
    """Simulate the learned SDE from ``t=0`` to ``t=1`` (forward) or reverse.

    Forward:  ``x_{t+dt} = x_t + v_f(x_t, t) dt + sigma sqrt(dt) z``
    Backward: ``x_{t-dt} = x_t + v_b(x_t, t) dt + sigma sqrt(dt) z``
              (``v_b`` points toward ``t = 0``; time decrements each step)

    Returns:
        Terminal states ``(N, D)`` — the simulated endpoints.
    """
    dt = 1.0 / n_steps
    sqrt_dt = math.sqrt(dt)
    x = x_init.clone()
    t = torch.full((x.shape[0],), 0.0 if forward else 1.0, device=x.device)

    for _ in range(n_steps):
        v = drift_net(x, t)
        x = x + v * dt + sigma * sqrt_dt * torch.randn_like(x)
        t = t + (dt if forward else -dt)

    return x


# ---------------------------------------------------------------------------
# IPF Trainer
# ---------------------------------------------------------------------------

@dataclass
class IPFConfig:
    """Hyperparameters for IPF training."""

    # Architecture
    hidden_dim: int = 256
    time_embed_dim: int = 32
    n_blocks: int = 3

    # Bridge / SDE
    sigma: float = 0.1
    t_eps: float = 1e-3          # clamp t away from 0 and 1

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    grad_clip: float = 1.0

    # IPF schedule
    n_ipf_iters: int = 5
    steps_per_half: int = 200    # gradient steps per forward/backward half

    # Coupling update
    sde_steps: int = 30          # Euler–Maruyama steps for re-coupling
    ot_blur: float = 0.05        # Sinkhorn blur for re-coupling
    ot_iters: int = 100          # Sinkhorn iterations for re-coupling


class IPFTrainer:
    """Trains a ``SchrodingerBridgeModel`` via Iterative Proportional Fitting.

    Args:
        tensors: Preprocessed section tensors (z-ordered), each ``(N_i, D)``.
        plans: Initial entropic OT couplings from ``compute_transport_plans``.
        config: ``IPFConfig`` hyperparameters.
        device: ``"auto"`` / ``"cuda"`` / ``"cpu"``.
    """

    def __init__(
        self,
        tensors: list[Tensor],
        plans: list[Tensor],
        config: IPFConfig | None = None,
        device: str = "auto",
    ):
        assert len(plans) == len(tensors) - 1
        if config is None:
            config = IPFConfig()
        self.config = config

        # Device
        from st3d.training import get_device
        self.device = get_device(device)

        # Data
        self.tensors = [t.to(self.device) for t in tensors]
        self.plans = [p.to(self.device) for p in plans]

        # Model
        state_dim = tensors[0].shape[1]
        self.model = SchrodingerBridgeModel(
            state_dim=state_dim,
            hidden_dim=config.hidden_dim,
            time_embed_dim=config.time_embed_dim,
            n_blocks=config.n_blocks,
        ).to(self.device)

        # Optimisers (separate for forward and backward, reset each half-step)
        self.opt_f = torch.optim.AdamW(
            self.model.forward_drift.parameters(),
            lr=config.lr, weight_decay=config.weight_decay,
        )
        self.opt_b = torch.optim.AdamW(
            self.model.backward_drift.parameters(),
            lr=config.lr, weight_decay=config.weight_decay,
        )

        # History
        self.history: dict[str, list[float]] = {
            "fwd_loss": [],
            "bwd_loss": [],
        }

    # ------------------------------------------------------------------
    # Bridge regression for one half-step
    # ------------------------------------------------------------------

    def _train_half(
        self,
        drift_net: nn.Module,
        optimizer: torch.optim.Optimizer,
        is_forward: bool,
        pbar_desc: str,
    ) -> list[float]:
        """Train one drift direction on Brownian bridge velocity targets.

        Forward target at ``(x_t, t)``:  ``(x_1 - x_t) / (1 - t)``
        Backward target at ``(x_t, t)``: ``(x_0 - x_t) / t``
        """
        cfg = self.config
        drift_net.train()
        losses: list[float] = []

        pbar = tqdm(range(cfg.steps_per_half), desc=pbar_desc, leave=False)
        for _ in pbar:
            total_loss = torch.tensor(0.0, device=self.device)

            for k, plan in enumerate(self.plans):
                x_src = self.tensors[k]
                x_tgt = self.tensors[k + 1]

                # Sample OT-coupled pairs
                x0, x1 = sample_ot_pairs(plan, x_src, x_tgt, cfg.batch_size)

                # Random bridge time, clamped away from singularities
                t = torch.rand(cfg.batch_size, device=self.device)
                t = t * (1.0 - 2 * cfg.t_eps) + cfg.t_eps  # [eps, 1-eps]

                # Bridge sample
                xt = brownian_bridge_sample(x0, x1, t, cfg.sigma)

                # Target velocity
                if is_forward:
                    target = (x1 - xt) / (1.0 - t)[:, None]
                else:
                    target = (x0 - xt) / t[:, None]

                pred = drift_net(xt, t)
                total_loss = total_loss + F.mse_loss(pred, target)

            # Average over pairs
            total_loss = total_loss / len(self.plans)

            optimizer.zero_grad()
            total_loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(drift_net.parameters(), cfg.grad_clip)
            optimizer.step()

            losses.append(total_loss.item())
            pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        return losses

    # ------------------------------------------------------------------
    # Coupling update via forward/backward simulation + OT
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_couplings(self, is_forward: bool) -> None:
        """Re-compute OT couplings after learning one drift direction.

        Forward:  simulate forward SDE from each source, couple pushed points
                  to true targets.
        Backward: simulate backward SDE from each target, couple true sources
                  to pulled points.
        """
        cfg = self.config
        self.model.eval()

        for k in range(len(self.plans)):
            x_src = self.tensors[k]
            x_tgt = self.tensors[k + 1]

            if is_forward:
                x_pushed = simulate_sde(
                    self.model.forward_drift, x_src,
                    sigma=cfg.sigma, n_steps=cfg.sde_steps, forward=True,
                )
                self.plans[k] = sinkhorn_transport_plan(
                    x_pushed, x_tgt, blur=cfg.ot_blur, n_iters=cfg.ot_iters,
                )
            else:
                x_pulled = simulate_sde(
                    self.model.backward_drift, x_tgt,
                    sigma=cfg.sigma, n_steps=cfg.sde_steps, forward=False,
                )
                self.plans[k] = sinkhorn_transport_plan(
                    x_src, x_pulled, blur=cfg.ot_blur, n_iters=cfg.ot_iters,
                )

    # ------------------------------------------------------------------
    # Full IPF loop
    # ------------------------------------------------------------------

    def fit(self, verbose: bool = True) -> dict[str, list[float]]:
        """Run Iterative Proportional Fitting.

        Returns:
            Loss history with keys ``"fwd_loss"`` and ``"bwd_loss"``.
        """
        cfg = self.config

        for ipf_iter in range(cfg.n_ipf_iters):
            header = f"IPF {ipf_iter + 1}/{cfg.n_ipf_iters}"

            # --- Forward half-step ---
            fwd_losses = self._train_half(
                self.model.forward_drift, self.opt_f,
                is_forward=True,
                pbar_desc=f"{header} fwd",
            )
            self.history["fwd_loss"].extend(fwd_losses)

            if verbose:
                mean_f = sum(fwd_losses[-20:]) / len(fwd_losses[-20:])
                print(f"  {header}  fwd loss = {mean_f:.4f}", end="")

            # Re-couple after forward
            self._update_couplings(is_forward=True)

            # --- Backward half-step ---
            bwd_losses = self._train_half(
                self.model.backward_drift, self.opt_b,
                is_forward=False,
                pbar_desc=f"{header} bwd",
            )
            self.history["bwd_loss"].extend(bwd_losses)

            if verbose:
                mean_b = sum(bwd_losses[-20:]) / len(bwd_losses[-20:])
                print(f"  bwd loss = {mean_b:.4f}")

            # Re-couple after backward
            self._update_couplings(is_forward=False)

        return self.history
