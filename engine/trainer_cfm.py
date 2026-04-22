"""
CFM Trainer for Stage 3 of ADynamics.

Handles training the Velocity Field Network using Conditional Flow Matching loss.
The trainer samples NC and AD latent pairs, interpolates them, and trains
the network to predict the optimal transport velocity field.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.vector_field import VelocityFieldNet, cfm_velocity_loss


class CFMTrainer:
    """
    Trainer class for Conditional Flow Matching (Stage 3).

    Handles:
        - Sampling z0 (NC) and z1 (AD) latent pairs
        - Time sampling t ~ U(0, 1)
        - Computing linear interpolation z_t = (1-t)*z0 + t*z1
        - Computing CFM loss and training the velocity field network
        - Checkpointing and validation

    Attributes:
        model: The VelocityFieldNet being trained
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        config: Training configuration dictionary
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
                - velocity_loss_weight: Weight for velocity loss
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.config = config

        # Move model to device
        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def sample_latent_pairs(
        self,
        z_all: Tensor,
        labels: Tensor,
        batch_size: int,
        num_nc: Optional[int] = None,
        num_ad: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample NC and AD latent pairs for CFM training.

        Randomly samples z0 (NC) and z1 (AD) from the latent batch.

        Args:
            z_all: All latents in batch [N, C, D, H, W]
            labels: Disease labels [N] (0=NC, 1=SCD, 2=MCI, 3=AD)
            batch_size: Number of pairs to sample
            num_nc: Optional number of NC samples (default: batch_size // 2)
            num_ad: Optional number of AD samples (default: batch_size // 2)

        Returns:
            Tuple of (z0 [B, C, D, H, W], z1 [B, C, D, H, W])
        """
        # Get NC indices (label == 0)
        nc_mask = labels == 0
        # Get AD indices (label == 3)
        ad_mask = labels == 3

        nc_indices = torch.where(nc_mask)[0]
        ad_indices = torch.where(ad_mask)[0]

        # Fallback if not enough NC/AD samples - use any samples
        if len(nc_indices) < 2:
            nc_indices = torch.arange(len(labels))
        if len(ad_indices) < 2:
            ad_indices = torch.arange(len(labels))

        # Sample pairs
        num_nc = num_nc or (batch_size // 2)
        num_ad = num_ad or (batch_size // 2)

        nc_samples = nc_indices[torch.randperm(len(nc_indices))[:num_nc]]
        ad_samples = ad_indices[torch.randperm(len(ad_indices))[:num_ad]]

        z0 = z_all[nc_samples].to(self.device)
        z1 = z_all[ad_samples].to(self.device)

        return z0, z1

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
        # Reshape t for broadcasting: [B] -> [B, 1, 1, 1, 1]
        t = t.view(-1, 1, 1, 1, 1)

        z_t = (1 - t) * z0 + t * z1
        return z_t

    def train_step(
        self,
        z0: Tensor,
        z1: Tensor,
        c: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Perform one CFM training step.

        1. Sample t ~ U(0, 1)
        2. Compute z_t = (1-t)*z0 + t*z1
        3. Predict v_pred = Model(z_t, t, c)
        4. Compute loss = ||v_pred - (z1 - z0)||^2

        Args:
            z0: Source latents (NC) [B, C, D, H, W]
            z1: Target latents (AD) [B, C, D, H, W]
            c: Optional clinical conditions [B, num_conditions]

        Returns:
            Dictionary of losses
        """
        batch_size = z0.shape[0]

        # Sample time
        t = self.sample_time(batch_size)
        # shape: [B]

        # Interpolate latents
        z_t = self.interpolate_latents(z0, z1, t)
        # shape: [B, C, D, H, W]

        # Predict velocity
        v_pred = self.model(z_t, t, c)
        # shape: [B, C, D, H, W]

        # Compute CFM loss
        loss = cfm_velocity_loss(v_pred, z0, z1)

        # Total loss
        velocity_weight = self.config.get("velocity_loss_weight", 1.0)
        total_loss = velocity_weight * loss

        return {
            "loss": total_loss,
            "velocity_loss": loss,
        }

    def train_epoch(
        self,
        latent_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Run one training epoch.

        Args:
            latent_loader: DataLoader yielding (z, labels, conditions) tuples

        Returns:
            Dictionary of average training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_velocity_loss = 0.0
        num_batches = 0

        for batch in latent_loader:
            # Get latent and labels
            z = batch["latent"].to(self.device)
            # shape: [B, C, D, H, W]
            labels = batch["label"].to(self.device)
            # shape: [B]

            # Optional conditions (e.g., age)
            c = None
            if "condition" in batch:
                c = batch["condition"].to(self.device)
                # shape: [B, num_conditions]

            # Sample NC and AD pairs from the batch
            z0, z1 = self.sample_latent_pairs(
                z, labels, batch_size=z.shape[0]
            )

            # Training step
            losses = self.train_step(z0, z1, c)

            # Backward pass
            self.optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate
            total_loss += losses["loss"].item()
            total_velocity_loss += losses["velocity_loss"].item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_velocity_loss = total_velocity_loss / num_batches

        return {
            "loss": avg_loss,
            "velocity_loss": avg_velocity_loss,
        }

    @torch.no_grad()
    def validate_epoch(
        self,
        latent_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Run one validation epoch.

        Args:
            latent_loader: DataLoader yielding (z, labels, conditions) tuples

        Returns:
            Dictionary of average validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_velocity_loss = 0.0
        num_batches = 0

        for batch in latent_loader:
            # Get latent and labels
            z = batch["latent"].to(self.device)
            labels = batch["label"].to(self.device)

            # Optional conditions
            c = None
            if "condition" in batch:
                c = batch["condition"].to(self.device)

            # Sample NC and AD pairs
            z0, z1 = self.sample_latent_pairs(
                z, labels, batch_size=z.shape[0]
            )

            # Training step (without backward)
            losses = self.train_step(z0, z1, c)

            total_loss += losses["loss"].item()
            total_velocity_loss += losses["velocity_loss"].item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_velocity_loss = total_velocity_loss / num_batches

        return {
            "loss": avg_loss,
            "velocity_loss": avg_velocity_loss,
        }

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

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
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

    def integrate_ode(
        self,
        z0: Tensor,
        c: Optional[Tensor] = None,
        steps: int = 20,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Euler integration of the learned velocity field.

        Solves: dz/dt = v(z, t) from t=0 to t=1

        Args:
            z0: Initial latent [B, C, D, H, W]
            c: Optional clinical conditions [B, num_conditions]
            steps: Number of integration steps

        Returns:
            Tuple of (z_final, trajectory) where trajectory is list of z_t
        """
        self.model.eval()

        z_t = z0.clone()
        dt = 1.0 / steps
        trajectory = [z_t.clone()]

        with torch.no_grad():
            for i in range(steps):
                t = torch.full((z0.shape[0],), i * dt, device=z0.device)
                v_t = self.model(z_t, t, c)
                z_t = z_t + v_t * dt
                trajectory.append(z_t.clone())

        return z_t, trajectory

    def train(
        self,
        latent_loader_train: DataLoader,
        latent_loader_val: DataLoader,
        num_epochs: int,
        save_interval: int = 50,
        output_dir: str = "./checkpoints",
    ) -> Dict[str, List[float]]:
        """
        Run full training loop.

        Args:
            latent_loader_train: Training data loader
            latent_loader_val: Validation data loader
            num_epochs: Number of epochs to train
            save_interval: Checkpoint save interval
            output_dir: Directory to save checkpoints

        Returns:
            Training history dictionary
        """
        history = {
            "train_loss": [],
            "train_velocity_loss": [],
            "val_loss": [],
            "val_velocity_loss": [],
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Training epoch
            train_metrics = self.train_epoch(latent_loader_train)

            # Validation epoch
            val_metrics = self.validate_epoch(latent_loader_val)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_metrics['loss']:.4f} (vel: {train_metrics['velocity_loss']:.4f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} (vel: {val_metrics['velocity_loss']:.4f})"
            )

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["train_velocity_loss"].append(train_metrics["velocity_loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_velocity_loss"].append(val_metrics["velocity_loss"])

            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(
                    output_dir,
                    f"cfm_epoch_{epoch+1}.pt",
                )
                self.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                best_path = os.path.join(output_dir, "cfm_best.pt")
                self.save_checkpoint(best_path)
                print(f"Best model saved to {best_path} (val_loss: {self.best_val_loss:.4f})")

        return history
