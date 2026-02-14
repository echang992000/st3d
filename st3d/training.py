"""
Training loop for st3d.

Single-loop trainer with curriculum loss scheduling.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from st3d.config import ST3DConfig
from st3d.losses import combined_loss, curriculum_expr_weight
from st3d.model import ST3DModel


# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> torch.device:
    """Pick the best available device.

    Args:
        preference: ``"auto"`` / ``"cuda"`` / ``"mps"`` / ``"cpu"``.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Trains an ``ST3DModel`` on preprocessed section tensors.

    Features
    --------
    * **Single training loop** with automatic curriculum scheduling (spatial
      loss first, then gradually increase expression loss weight).
    * Device auto-detection (CUDA -> MPS -> CPU).
    * Gradient clipping, checkpointing, early stopping.
    * Returns a loss history dict for plotting.

    Args:
        model: An ``ST3DModel`` instance.
        config: ``ST3DConfig`` (uses optimizer / loss / scheduling params).
        device: Device string or ``"auto"``.
        checkpoint_dir: If set, save periodic checkpoints here.
    """

    def __init__(
        self,
        model: ST3DModel,
        config: ST3DConfig | None = None,
        device: str = "auto",
        checkpoint_dir: str | None = None,
    ):
        if config is None:
            config = model.config
        self.config = config
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimiser
        OptimizerCls = getattr(torch.optim, config.optimizer, torch.optim.NAdam)
        self.optimizer = OptimizerCls(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )

        self.history: dict[str, list[float]] = {
            "total": [], "spatial": [], "expression": [], "w_expr": [],
        }

    # ------------------------------------------------------------------
    # Core training step for one pair of consecutive sections
    # ------------------------------------------------------------------

    def _train_pair(
        self,
        u_start: Tensor,
        u_end: Tensor,
        w_expr: float,
        add_noise: bool = True,
    ) -> dict[str, Tensor]:
        """Run forward (and optionally backward) integration between two sections
        and compute the combined loss.
        """
        coords_s, expr_s = u_start[:, :3], u_start[:, 3:]
        coords_e, expr_e = u_end[:, :3], u_end[:, 3:]

        z_start = coords_s[:, 2].mean()
        z_end = coords_e[:, 2].mean()
        dz = self.config.dz
        n_steps = max(1, round(abs((z_end - z_start).item()) / dz))

        # ---- Forward integration ----
        traj_f = self.model.forward_integrate(coords_s, expr_s, n_steps, dz, add_noise)
        pred_coords_f, pred_expr_f = traj_f[-1]

        loss_dict = combined_loss(
            pred_coords_f, pred_expr_f,
            coords_e, expr_e,
            w_spatial=self.config.spatial_loss_weight,
            w_expr=w_expr,
            sinkhorn_blur=self.config.sinkhorn_blur,
            sinkhorn_iters=self.config.sinkhorn_iters,
        )
        total = loss_dict["total"]

        # ---- Backward integration (if bidirectional) ----
        if self.model.backward_net is not None:
            traj_b = self.model.backward_integrate(coords_e, expr_e, n_steps, dz, add_noise)
            pred_coords_b, pred_expr_b = traj_b[-1]

            loss_b = combined_loss(
                pred_coords_b, pred_expr_b,
                coords_s, expr_s,
                w_spatial=self.config.spatial_loss_weight,
                w_expr=w_expr,
                sinkhorn_blur=self.config.sinkhorn_blur,
                sinkhorn_iters=self.config.sinkhorn_iters,
            )
            total = total + loss_b["total"]
            loss_dict["spatial"] = loss_dict["spatial"] + loss_b["spatial"]
            loss_dict["expression"] = loss_dict["expression"] + loss_b["expression"]
            loss_dict["total"] = total

        return loss_dict

    # ------------------------------------------------------------------
    # Full training run
    # ------------------------------------------------------------------

    def fit(
        self,
        tensors: list[Tensor],
        epochs: int = 100,
        checkpoint_every: int = 20,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """Train the model on preprocessed section tensors.

        Args:
            tensors: List of ``(N_i, 3 + n_pcs)`` tensors (one per section,
                in z-order) as returned by ``preprocess_sections``.
            epochs: Total number of training epochs.
            checkpoint_every: Save checkpoint every N epochs (0 = never).
            verbose: Show tqdm progress bar.

        Returns:
            Loss history dict with keys ``"total"``, ``"spatial"``,
            ``"expression"``, ``"w_expr"``.
        """
        # Move tensors to device
        tensors = [t.to(self.device) for t in tensors]
        n_pairs = len(tensors) - 1
        assert n_pairs >= 1, "Need at least two sections to train"

        self.model.train()

        pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
        for epoch in pbar:
            # Curriculum weight for expression loss
            w_expr = curriculum_expr_weight(
                epoch,
                start_epoch=self.config.curriculum_start_epoch,
                end_epoch=self.config.curriculum_end_epoch,
                start_weight=self.config.expr_loss_weight_start,
                end_weight=self.config.expr_loss_weight_end,
            )

            epoch_total = 0.0
            epoch_spatial = 0.0
            epoch_expr = 0.0

            # Iterate over all consecutive section pairs
            for i in range(n_pairs):
                self.optimizer.zero_grad()
                loss_dict = self._train_pair(
                    tensors[i], tensors[i + 1], w_expr=w_expr, add_noise=True,
                )
                loss_dict["total"].backward()
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

                epoch_total += loss_dict["total"].item()
                epoch_spatial += loss_dict["spatial"].item()
                epoch_expr += loss_dict["expression"].item()

            # Record averages
            self.history["total"].append(epoch_total / n_pairs)
            self.history["spatial"].append(epoch_spatial / n_pairs)
            self.history["expression"].append(epoch_expr / n_pairs)
            self.history["w_expr"].append(w_expr)

            pbar.set_postfix({
                "loss": f"{epoch_total / n_pairs:.4f}",
                "sp": f"{epoch_spatial / n_pairs:.4f}",
                "ex": f"{epoch_expr / n_pairs:.4f}",
                "w_e": f"{w_expr:.2f}",
            })

            # Checkpoint
            if (
                self.checkpoint_dir
                and checkpoint_every > 0
                and (epoch + 1) % checkpoint_every == 0
            ):
                path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                self.model.save(path)

        # Final save
        if self.checkpoint_dir:
            self.model.save(os.path.join(self.checkpoint_dir, "model_final.pt"))

        return self.history
