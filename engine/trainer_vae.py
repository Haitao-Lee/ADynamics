"""
VAE Trainer for Stage 1 of ADynamics.

Handles training loop, validation, checkpointing, and logging for the 3D VAE.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from engine.losses import total_vae_loss


class VAETrainer:
    """
    Trainer class for 3D VAE in Stage 1 of ADynamics.

    Handles:
        - Training and validation epochs
        - Loss computation (reconstruction + KL)
        - Checkpoint saving and loading
        - Learning rate scheduling
        - Logging of training metrics

    Attributes:
        model: The VAE model being trained
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
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
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.config = config

        # Move model to device
        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.

        Performs forward pass, loss computation, backward pass,
        and optimizer step for all batches in the training set.

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

        for batch in self.train_loader:
            # Get input data
            images = batch["image"]
            # shape: [B, 1, D, H, W]

            # Move to device
            images = images.to(self.device)

            # Forward pass
            recon, mu, logvar = self.model(images)

            # Compute losses
            recon_loss = torch.nn.functional.l1_loss(recon, images)

            # KL loss normalized by latent elements
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            num_latent_elements = mu.numel()
            kl_loss_normalized = kl_loss / num_latent_elements

            # Total loss
            kl_weight = self.config.get("kl_weight", 0.0001)
            loss = recon_loss + kl_weight * kl_loss_normalized

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss_normalized.item()
            num_batches += 1

        # Compute averages
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches

        return {
            "loss": avg_loss,
            "recon_loss": avg_recon_loss,
            "kl_loss": avg_kl_loss,
        }

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """
        Run one validation epoch.

        Performs forward pass and loss computation without gradient tracking.
        No optimizer updates are performed.

        Returns:
            Dictionary containing average validation metrics:
                - loss: Total VAE loss
                - recon_loss: Reconstruction loss component
                - kl_loss: KL divergence component
        """
        self.model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # Get input data
            images = batch["image"]
            # shape: [B, 1, D, H, W]

            # Move to device
            images = images.to(self.device)

            # Forward pass
            recon, mu, logvar = self.model(images)

            # Compute losses
            recon_loss = torch.nn.functional.l1_loss(recon, images)

            # KL loss normalized by latent elements
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            num_latent_elements = mu.numel()
            kl_loss_normalized = kl_loss / num_latent_elements

            # Total loss
            kl_weight = self.config.get("kl_weight", 0.0001)
            loss = recon_loss + kl_weight * kl_loss_normalized

            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss_normalized.item()
            num_batches += 1

        # Compute averages
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches

        return {
            "loss": avg_loss,
            "recon_loss": avg_recon_loss,
            "kl_loss": avg_kl_loss,
        }

    def save_checkpoint(
        self,
        filepath: str,
        include_optimizer: bool = True,
        include_scheduler: bool = True,
    ) -> None:
        """
        Save model checkpoint to disk.

        Saves model state, optimizer state, scheduler state, and epoch info.

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

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

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
        Run full training loop for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train
            save_interval: Interval for saving checkpoints
            output_dir: Directory to save checkpoints

        Returns:
            Dictionary containing training history with lists of:
                - train_loss, train_recon_loss, train_kl_loss
                - val_loss, val_recon_loss, val_kl_loss
        """
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

            # Training epoch
            train_metrics = self.train_epoch()

            # Validation epoch
            val_metrics = self.validate_epoch()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]

            # Log metrics
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_metrics['loss']:.4f} (recon: {train_metrics['recon_loss']:.4f}, "
                f"kl: {train_metrics['kl_loss']:.6f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} (recon: {val_metrics['recon_loss']:.4f}, "
                f"kl: {val_metrics['kl_loss']:.6f})"
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
                checkpoint_path = os.path.join(
                    output_dir,
                    f"vae_epoch_{epoch+1}.pt",
                )
                self.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                best_path = os.path.join(output_dir, "vae_best.pt")
                self.save_checkpoint(best_path)
                print(f"Best model saved to {best_path} (val_loss: {self.best_val_loss:.4f})")

        return history
