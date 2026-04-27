"""
CFM Trainer for Stage 3 of ADynamics.

Handles training the Velocity Field Network using Conditional Flow Matching loss.
Samples NC and AD latent pairs from pre-built global pools to ensure pure
NC->AD correspondence in every training step.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from engine.losses import cfm_loss
from models.vector_field import VelocityFieldNet


class CFMTrainer:
    """
    Trainer class for Conditional Flow Matching (Stage 3).

    Handles:
        - Pre-pooled NC/AD latent sampling from global pools
        - Time sampling t ~ U(0, 1)
        - Linear interpolation z_t = (1-t)*z0 + t*z1
        - CFM loss and velocity field network training
        - AMP for memory-efficient HD training
        - Checkpointing and validation

    Attributes:
        model: The VelocityFieldNet being trained
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        scaler: GradScaler for AMP training
        device: Device to train on (cuda/cpu)
        config: Training configuration dictionary
        velocity_loss_weight: Weight for velocity loss
    """

    def __init__(
        self,
        model: VelocityFieldNet,
        optimizer: AdamW,
        device: Union[str, torch.device],
        config: Dict[str, Any],
        scheduler: Optional[CosineAnnealingLR] = None,
    ) -> None:
        """
        Initialize the CFM trainer.

        Args:
            model: The VelocityFieldNet to train
            optimizer: AdamW optimizer
            device: Device to train on ("cuda" or "cpu")
            config: Configuration dictionary containing:
                - ode_steps: Number of ODE integration steps
                - velocity_loss_weight: Weight for velocity loss (default: 1.0)
                - use_amp: Enable AMP training (default: True)
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.config = config

        # Loss weight with default value
        self.velocity_loss_weight = config.get("velocity_loss_weight", 1.0)

        # AMP configuration: only create scaler when use_amp is True
        self.use_amp = config.get("use_amp", True)
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Move model to device
        self.model.to(self.device)

        # Global pools for NC and AD latents (populated before training)
        self.nc_latent_pool: List[Tensor] = []
        self.ad_latent_pool: List[Tensor] = []
        self.nc_condition_pool: List[Tensor] = []
        self.ad_condition_pool: List[Tensor] = []

        # Demographics pools (age, sex) for ablation when use_demographics=True
        self.nc_age_pool: List[Tensor] = []
        self.ad_age_pool: List[Tensor] = []
        self.nc_sex_pool: List[Tensor] = []
        self.ad_sex_pool: List[Tensor] = []

        # Check if using demographics embedding
        self.use_demographics = config.get("use_demographics", False)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def build_latent_pools(
        self,
        latent_loader: DataLoader,
    ) -> None:
        """
        Pre-build global NC and AD latent pools before training.

        This ensures every sampled pair is a true NC-AD pair, eliminating
        the risk of fallback sampling that would break CFM physics.

        Args:
            latent_loader: DataLoader yielding (z, labels, conditions) tuples
        """
        self.nc_latent_pool = []
        self.ad_latent_pool = []
        self.nc_condition_pool = []
        self.ad_condition_pool = []
        self.nc_age_pool = []
        self.ad_age_pool = []
        self.nc_sex_pool = []
        self.ad_sex_pool = []

        self.model.eval()
        with torch.no_grad():
            for batch in latent_loader:
                z = batch["latent"]
                labels = batch["label"]
                c = batch.get("condition", None)
                age = batch.get("age", None)
                sex = batch.get("sex", None)

                # NC: label == 0
                nc_mask = labels == 0
                for i, is_nc in enumerate(nc_mask):
                    if is_nc:
                        self.nc_latent_pool.append(z[i])
                        if c is not None:
                            self.nc_condition_pool.append(c[i])
                        if age is not None:
                            self.nc_age_pool.append(age[i])
                        if sex is not None:
                            self.nc_sex_pool.append(sex[i])

                # AD: label == 3
                ad_mask = labels == 3
                for i, is_ad in enumerate(ad_mask):
                    if is_ad:
                        self.ad_latent_pool.append(z[i])
                        if c is not None:
                            self.ad_condition_pool.append(c[i])
                        if age is not None:
                            self.ad_age_pool.append(age[i])
                        if sex is not None:
                            self.ad_sex_pool.append(sex[i])

        # Move to device for faster sampling during training
        self.nc_latent_pool = [z.to(self.device) for z in self.nc_latent_pool]
        self.ad_latent_pool = [z.to(self.device) for z in self.ad_latent_pool]
        self.nc_condition_pool = [c.to(self.device) for c in self.nc_condition_pool]
        self.ad_condition_pool = [c.to(self.device) for c in self.ad_condition_pool]
        self.nc_age_pool = [a.to(self.device) for a in self.nc_age_pool]
        self.ad_age_pool = [a.to(self.device) for a in self.ad_age_pool]
        self.nc_sex_pool = [s.to(self.device) for s in self.nc_sex_pool]
        self.ad_sex_pool = [s.to(self.device) for s in self.ad_sex_pool]

        print(f"Latent pools built: {len(self.nc_latent_pool)} NC, {len(self.ad_latent_pool)} AD")
        if self.use_demographics:
            print(f"Demographics pools: {len(self.nc_age_pool)} age, {len(self.nc_sex_pool)} sex")

    def sample_latent_pairs(
        self,
        batch_size: int,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Sample NC and AD latent pairs from pre-built global pools.

        Args:
            batch_size: Number of pairs to sample

        Returns:
            Tuple of (z0, z1, c0, c1, age0, sex0) where:
                - z0: NC latents [B, C, D, H, W]
                - z1: AD latents [B, C, D, H, W]
                - c0: NC conditions [B, num_conditions] or None (legacy)
                - c1: AD conditions [B, num_conditions] or None (legacy)
                - age0: NC ages [B, 1] or None
                - sex0: NC sexes [B, 1] or None
        """
        if len(self.nc_latent_pool) == 0 or len(self.ad_latent_pool) == 0:
            raise RuntimeError(
                "Latent pools are empty. Call build_latent_pools() before training."
            )

        # Random sampling from pools
        nc_indices = torch.randint(0, len(self.nc_latent_pool), (batch_size,))
        ad_indices = torch.randint(0, len(self.ad_latent_pool), (batch_size,))

        z0 = torch.stack([self.nc_latent_pool[i] for i in nc_indices])
        z1 = torch.stack([self.ad_latent_pool[i] for i in ad_indices])

        c0 = None
        c1 = None
        if self.nc_condition_pool and self.ad_condition_pool:
            c0 = torch.stack([self.nc_condition_pool[i] for i in nc_indices])
            c1 = torch.stack([self.ad_condition_pool[i] for i in ad_indices])

        # Demographics (age, sex)
        age0 = None
        sex0 = None
        if self.use_demographics and self.nc_age_pool and self.nc_sex_pool:
            age0 = torch.stack([self.nc_age_pool[i] for i in nc_indices])
            sex0 = torch.stack([self.nc_sex_pool[i] for i in nc_indices])

        return z0, z1, c0, c1, age0, sex0

    def sample_time(self, batch_size: int) -> Tensor:
        """
        Sample time steps t ~ U(0, 1).

        Args:
            batch_size: Number of time steps to sample

        Returns:
            Tensor of time steps [B] in [0, 1]
        """
        t = torch.rand(batch_size, device=self.device)
        return t

    def interpolate_latents(
        self,
        z0: Tensor,
        z1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """
        Compute linear interpolation z_t = (1-t)*z0 + t*z1.

        Args:
            z0: Source latents [B, C, D, H, W]
            z1: Target latents [B, C, D, H, W]
            t: Time steps [B] in [0, 1]

        Returns:
            Interpolated latents [B, C, D, H, W]
        """
        t = t.view(-1, 1, 1, 1, 1)
        z_t = (1 - t) * z0 + t * z1
        return z_t

    def train_step(
        self,
        z0: Tensor,
        z1: Tensor,
        c0: Optional[Tensor] = None,
        c1: Optional[Tensor] = None,
        age0: Optional[Tensor] = None,
        sex0: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Perform one CFM training step with AMP.

        1. Sample t ~ U(0, 1)
        2. Compute z_t = (1-t)*z0 + t*z1
        3. Predict v_pred = Model(z_t, t, c) or Model(z_t, t, age=age, sex=sex)
        4. Compute loss = ||v_pred - (z1 - z0)||^2

        Args:
            z0: Source latents (NC) [B, C, D, H, W]
            z1: Target latents (AD) [B, C, D, H, W]
            c0: Optional NC conditions [B, num_conditions] (legacy)
            c1: Optional AD conditions [B, num_conditions] (legacy)
            age0: Optional NC ages [B, 1] normalized to [0, 1]
            sex0: Optional NC sexes [B, 1] (0=female, 1=male)

        Returns:
            Dictionary of losses
        """
        batch_size = z0.shape[0]

        t = self.sample_time(batch_size)

        z_t = self.interpolate_latents(z0, z1, t)

        if self.use_demographics:
            # Use separate age/sex embeddings for ablation
            v_pred = self.model(z_t, t, c=None, age=age0, sex=sex0)
        else:
            # Use legacy condition embedding
            v_pred = self.model(z_t, t, c0)

        loss = cfm_loss(v_pred, z0, z1)

        total_loss = self.velocity_loss_weight * loss

        return {
            "loss": total_loss,
            "velocity_loss": loss,
        }

    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch with optional AMP.

        Returns:
            Dictionary of average training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_velocity_loss = 0.0
        num_batches = 0

        # Get batch size from config or use first pool size
        batch_size = self.config.get("batch_size", 8)

        # Determine number of batches based on pool sizes
        num_pairs = min(len(self.nc_latent_pool), len(self.ad_latent_pool))
        num_batches_total = max(1, num_pairs // batch_size)

        for _ in range(num_batches_total):
            z0, z1, c0, c1, age0, sex0 = self.sample_latent_pairs(batch_size)

            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                losses = self.train_step(z0, z1, c0, c1, age0, sex0)

            if self.use_amp:
                self.scaler.scale(losses["loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += losses["loss"].item()
            total_velocity_loss += losses["velocity_loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_velocity_loss = total_velocity_loss / max(num_batches, 1)

        return {
            "loss": avg_loss,
            "velocity_loss": avg_velocity_loss,
        }

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """
        Run one validation epoch with optional AMP.

        Returns:
            Dictionary of average validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_velocity_loss = 0.0
        num_batches = 0

        batch_size = self.config.get("batch_size", 8)
        num_pairs = min(len(self.nc_latent_pool), len(self.ad_latent_pool))
        num_batches_total = max(1, num_pairs // batch_size)

        for _ in range(num_batches_total):
            z0, z1, c0, c1, age0, sex0 = self.sample_latent_pairs(batch_size)

            with autocast('cuda', enabled=self.use_amp):
                losses = self.train_step(z0, z1, c0, c1, age0, sex0)

            total_loss += losses["loss"].item()
            total_velocity_loss += losses["velocity_loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_velocity_loss = total_velocity_loss / max(num_batches, 1)

        return {
            "loss": avg_loss,
            "velocity_loss": avg_velocity_loss,
        }

    def integrate_ode(
        self,
        z0: Tensor,
        c: Optional[Tensor] = None,
        steps: int = 20,
        method: str = "euler",
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Integrate the learned velocity field to evolve latent from t=0 to t=1.

        Currently supports Euler integration. For higher accuracy, consider
        using torchdiffeq library with methods like 'dopri5' (Dormand-Prince)
        or 'rk4' (Runge-Kutta 4th order).

        Args:
            z0: Initial latent [B, C, D, H, W]
            c: Optional clinical conditions [B, num_conditions]
            steps: Number of integration steps (for Euler)
            method: Integration method ('euler' or 'dopri5' if torchdiffeq available)

        Returns:
            Tuple of (z_final, trajectory) where trajectory is list of z_t
        """
        self.model.eval()

        z_t = z0.clone()
        dt = 1.0 / steps
        trajectory = [z_t.clone()]

        with torch.no_grad():
            if method == "euler":
                for i in range(steps):
                    t = torch.full((z0.shape[0],), i * dt, device=z0.device, dtype=z0.dtype)
                    v_t = self.model(z_t, t, c)
                    z_t = z_t + v_t * dt
                    trajectory.append(z_t.clone())
            else:
                # Placeholder for torchdiffeq integration
                # TODO: if method == "dopri5":
                #   from torchdiffeq import dopri5_deprecated
                #   ...
                raise NotImplementedError(f"Integration method '{method}' not implemented. Use 'euler'.")

        return z_t, trajectory

    def save_checkpoint(
        self,
        filepath: str,
        include_optimizer: bool = True,
        include_scheduler: bool = True,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
            include_optimizer: Whether to save optimizer state
            include_scheduler: Whether to save scheduler state
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if include_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        if include_scheduler and self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def train(
        self,
        latent_loader_train: DataLoader,
        latent_loader_val: DataLoader,
        num_epochs: int,
        save_interval: int = 50,
        output_dir: str = "./checkpoints",
    ) -> Dict[str, List[float]]:
        """
        Run full training loop with KL annealing.

        Args:
            latent_loader_train: Training data loader (used to build pools)
            latent_loader_val: Validation data loader
            num_epochs: Number of epochs to train
            save_interval: Checkpoint save interval
            output_dir: Directory to save checkpoints

        Returns:
            Training history dictionary
        """
        # Build global latent pools before training
        self.build_latent_pools(latent_loader_train)

        history = {
            "train_loss": [],
            "train_velocity_loss": [],
            "val_loss": [],
            "val_velocity_loss": [],
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_metrics['loss']:.4f} (vel: {train_metrics['velocity_loss']:.4f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} (vel: {val_metrics['velocity_loss']:.4f})"
            )

            history["train_loss"].append(train_metrics["loss"])
            history["train_velocity_loss"].append(train_metrics["velocity_loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_velocity_loss"].append(val_metrics["velocity_loss"])

            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(output_dir, f"cfm_epoch_{epoch+1}.pt")
                self.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                best_path = os.path.join(output_dir, "cfm_best.pt")
                self.save_checkpoint(best_path)
                print(f"Best model saved to {best_path} (val_loss: {self.best_val_loss:.4f})")

        return history