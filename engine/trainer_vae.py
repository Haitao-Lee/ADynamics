"""
VAE Trainer for Stage 1 of ADynamics.

Handles training loop, validation, checkpointing, and logging for the 3D VAE.
Supports AMP (Automatic Mixed Precision) for memory-efficient HD training.
"""

import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from engine.losses import total_vae_loss


class VAETrainer:
    """
    Trainer class for 3D VAE in Stage 1 of ADynamics.

    Handles:
        - AMP training for memory-efficient HD (256x256x192) training
        - Training and validation epochs
        - Loss computation via engine.losses.total_vae_loss
        - KL annealing to prevent posterior collapse
        - Checkpoint saving and loading
        - Learning rate scheduling
        - Logging of training metrics

    Attributes:
        model: The VAE model being trained
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        scaler: GradScaler for AMP training
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (cuda/cpu)
        config: Training configuration dictionary
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: AdamW,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Union[str, torch.device],
        config: Dict[str, Any],
        scheduler: Optional[CosineAnnealingLR] = None,
    ) -> None:
        """
        Initialize the VAE trainer.

        Args:
            model: The ADynamicsVAE3D model to train
            optimizer: AdamW optimizer instance
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to train on ("cuda" or "cpu")
            config: Configuration dictionary containing:
                - kl_weight: Weight for KL divergence term
                - recon_loss_type: Type of reconstruction loss ("l1" or "l2")
                - kl_warmup_epochs: Epochs for KL annealing (default: 5)
                - use_amp: Enable AMP training (default: True)
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.config = config

        # AMP configuration: only create scaler when use_amp is True
        self.use_amp = config.get("use_amp", False)
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Move model to device
        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        # Fixed visualization batch for deterministic loss component logging
        self.viz_batch = None
        if self.val_loader is not None and len(self.val_loader) > 0:
            for batch in self.val_loader:
                self.viz_batch = batch["image"].to(self.device)
                break

    def train_epoch(self, current_kl_weight: float) -> Dict[str, float]:
        """
        Run one training epoch with optional AMP.

        Performs forward pass with optional autocast, loss computation,
        backward pass with gradient scaling, and optimizer step.

        Args:
            current_kl_weight: KL weight for this epoch (supports annealing)

        Returns:
            Dictionary containing average training metrics:
                - loss: Total VAE loss
                - recon_loss: Reconstruction loss component
                - kl_loss: KL divergence component
        """
        self.model.train()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        recon_loss_type = self.config.get("recon_loss_type", "l1")

        for batch in self.train_loader:
            images = batch["image"]
            images = images.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with AMP autocast if enabled
            with autocast('cuda', enabled=self.use_amp):
                recon, mu, logvar = self.model(images)
                loss = total_vae_loss(
                    recon,
                    images,
                    mu,
                    logvar,
                    kl_weight=current_kl_weight,
                    recon_loss_type=recon_loss_type,
                )

            # Backward pass with gradient scaling (if AMP enabled)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Accumulate metrics (only during logging to avoid sync overhead)
            total_loss += loss.item()
            num_batches += 1

        # Compute averages
        avg_loss = total_loss / num_batches

        # Compute individual components for logging
        avg_recon_loss, avg_kl_loss = self._compute_loss_components(recon_loss_type)

        return {
            "loss": avg_loss,
            "recon_loss": avg_recon_loss,
            "kl_loss": avg_kl_loss,
        }

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """
        Run one validation epoch with optional AMP.

        Performs forward pass with optional autocast and loss computation
        without gradient tracking. No optimizer updates.

        Returns:
            Dictionary containing average validation metrics:
                - loss: Total VAE loss
                - recon_loss: Reconstruction loss component
                - kl_loss: KL divergence component
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        recon_loss_type = self.config.get("recon_loss_type", "l1")
        kl_weight = self.config.get("kl_weight", 0.0001)

        for batch in self.val_loader:
            images = batch["image"]
            images = images.to(self.device)

            # Forward pass with AMP autocast if enabled
            with autocast('cuda', enabled=self.use_amp):
                recon, mu, logvar = self.model(images)
                loss = total_vae_loss(
                    recon,
                    images,
                    mu,
                    logvar,
                    kl_weight=kl_weight,
                    recon_loss_type=recon_loss_type,
                )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_recon_loss, avg_kl_loss = self._compute_loss_components(recon_loss_type)

        return {
            "loss": avg_loss,
            "recon_loss": avg_recon_loss,
            "kl_loss": avg_kl_loss,
        }

    def _compute_loss_components(self, recon_loss_type: str) -> tuple:
        """
        Compute reconstruction and KL loss components for logging.

        This is called after training to get component-wise losses
        without extra forward passes.

        Args:
            recon_loss_type: Type of reconstruction loss

        Returns:
            Tuple of (avg_recon_loss, avg_kl_loss)
        """
        # Use fixed viz_batch for deterministic loss component logging
        if self.viz_batch is None:
            return 0.0, 0.0
        images = self.viz_batch

        with autocast('cuda', enabled=self.use_amp):
            recon, mu, logvar = self.model(images)
            recon_loss, kl_loss = self._get_loss_components(
                recon, images, mu, logvar, recon_loss_type
            )

        return recon_loss.item(), kl_loss.item()

    def _get_loss_components(
        self,
        recon: torch.Tensor,
        images: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        recon_loss_type: str,
    ) -> tuple:
        """
        Extract individual loss components from total_vae_loss.

        Args:
            recon: Reconstructed image
            images: Original image
            mu: Latent mean
            logvar: Latent log variance
            recon_loss_type: Type of reconstruction loss

        Returns:
            Tuple of (recon_loss, kl_loss) without reduction
        """
        from engine.losses import vae_kl_loss, vae_reconstruction_loss

        recon_loss = vae_reconstruction_loss(recon, images, loss_type=recon_loss_type)
        kl_loss = vae_kl_loss(mu, logvar, reduction="mean")

        return recon_loss, kl_loss

    def save_checkpoint(
        self,
        filepath: str,
        include_optimizer: bool = True,
        include_scheduler: bool = True,
    ) -> None:
        """
        Save model checkpoint to disk.

        Args:
            filepath: Path to save checkpoint file
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
        Load model checkpoint from disk.

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
        num_epochs: int,
        save_interval: int = 50,
        output_dir: str = "./checkpoints",
    ) -> Dict[str, list]:
        """
        Run full training loop with KL annealing and AMP.

        KL annealing starts from 0 and linearly increases to target_kl_weight
        over kl_warmup_epochs to prevent posterior collapse.

        Args:
            num_epochs: Number of epochs to train
            save_interval: Interval for saving checkpoints
            output_dir: Directory to save checkpoints

        Returns:
            Dictionary containing training history with lists of:
                - train_loss, train_recon_loss, train_kl_loss
                - val_loss, val_recon_loss, val_kl_loss
        """
        target_kl_weight = self.config.get("kl_weight", 0.0001)
        kl_warmup_epochs = self.config.get("kl_warmup_epochs", 5)

        history = {
            "train_loss": [],
            "train_recon_loss": [],
            "train_kl_loss": [],
            "val_loss": [],
            "val_recon_loss": [],
            "val_kl_loss": [],
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Compute KL weight with annealing (warmup from 0 to target)
            if epoch < kl_warmup_epochs:
                current_kl_weight = target_kl_weight * (epoch + 1) / kl_warmup_epochs
            else:
                current_kl_weight = target_kl_weight

            # Training epoch with current KL weight
            train_metrics = self.train_epoch(current_kl_weight)

            # Validation epoch
            val_metrics = self.validate_epoch()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log metrics
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"LR: {current_lr:.6f} | "
                f"KL_w: {current_kl_weight:.6f} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["train_recon_loss"].append(train_metrics["recon_loss"])
            history["train_kl_loss"].append(train_metrics["kl_loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_recon_loss"].append(val_metrics["recon_loss"])
            history["val_kl_loss"].append(val_metrics["kl_loss"])

            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(output_dir, f"vae_epoch_{epoch+1}.pt")
                self.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                best_path = os.path.join(output_dir, "vae_best.pt")
                self.save_checkpoint(best_path)
                print(f"Best model saved to {best_path} (val_loss: {self.best_val_loss:.4f})")

        return history