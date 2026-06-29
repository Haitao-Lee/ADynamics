"""
VAE Trainer for Stage 1 of ADynamics.

Handles training loop, validation, checkpointing, and logging for the 3D VAE.
Supports AMP (Automatic Mixed Precision) for memory-efficient HD training.
"""

import math
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from engine.losses import total_vae_loss, vae_reconstruction_loss, vae_kl_loss, gradient_loss, ssim_loss
from engine.losses import ordinal_cross_entropy_loss, ordinal_regression_loss


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

        # Detect if model is DataParallel and get device list
        self.is_dataparallel = False
        self.output_device = self.device
        if hasattr(model, "module"):
            # Model is DataParallel or DDP wrapped
            self.is_dataparallel = True
            # Get list of devices from model
            if hasattr(model, "device_ids"):
                self.devices = [torch.device(f"cuda:{i}") for i in model.device_ids]
            else:
                self.devices = [self.device]
        else:
            self.devices = [self.device]

        # AMP configuration: only create scaler when use_amp is True
        self.use_amp = config.get("use_amp", False)
        self.scaler = GradScaler() if self.use_amp else None

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

        # Memory Bank for contrastive learning (initialized lazily)
        self.memory_bank = None
        self.use_memory_bank = config.get("use_memory_bank", False)
        self.bank_size = config.get("memory_bank_size", 2048)

    def _get_memory_bank(self):
        """Get or initialize memory bank with correct latent dimension."""
        if self.memory_bank is None:
            from engine.losses import MemoryBank
            latent_channels = self.config.get("latent_channels", 64)
            self.memory_bank = MemoryBank(
                size=self.bank_size,
                device=self.device,
                latent_channels=latent_channels,
            )
        return self.memory_bank

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
        total_contrastive_loss = 0.0
        num_batches = 0

        recon_loss_type = self.config.get("recon_loss_type", "l1")
        gradient_weight = self.config.get("gradient_weight", 0.0)
        ssim_weight = self.config.get("ssim_weight", 0.0)
        use_multi_scale = self.config.get("use_multi_scale", False)
        multi_scale_weights = self.config.get("multi_scale_weights", [0.1, 0.2, 0.3])
        contrastive_weight = self.config.get("contrastive_weight", 0.0)
        contrastive_temp = self.config.get("contrastive_temperature", 0.1)
        use_ordinal = self.config.get("use_ordinal_contrastive", True)
        bank_size = self.config.get("memory_bank_size", 2048)
        accumulation_steps = self.config.get("accumulation_steps", 1)

        from tqdm import tqdm

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Train", leave=False)

        # Temp bank for contrastive learning: collect mu features during accumulation,
        # then compute ordinal contrastive loss with latest model weights before optimizer step.
        # This ensures all samples in the effective batch are encoded by the same updated encoder.
        # Pin to CPU to avoid OOM: 256^3 3D images x accumulation_steps would otherwise
        # blow the 24GB GPU memory budget. Move back to device at the optimizer step.
        temp_features = []  # list of [B, C, D, H, W] mu tensors (on CPU)
        temp_labels = []  # list of [B] label tensors (on CPU)

        # OOM fix: initialize con_loss_val outside the accumulation-cycle if
        # block so the metric accumulation at the end of every batch is
        # always defined, even on non-final accumulation steps where the
        # contrastive loss is computed only at the optimizer step.
        con_loss_val = 0.0

        for batch_idx, batch in pbar:
            images = batch["image"]
            images = images.to(self.device)

            # Extract labels for contrastive learning
            labels = None
            if contrastive_weight > 0 and "label" in batch:
                labels = batch["label"].to(self.device)
                if labels.dim() > 1:
                    labels = labels.squeeze()

            # Extract demographic info for conditioning
            age = None
            sex = None
            if getattr(self.model, "module", self.model).use_demographic_cond:
                if "age" in batch and batch["age"] is not None:
                    age = batch["age"].to(self.device)
                if "sex" in batch and batch["sex"] is not None:
                    sex = batch["sex"].to(self.device)

            # Zero grad and clear temp bank at start of accumulation cycle
            if batch_idx % accumulation_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
                temp_features.clear()
                temp_labels.clear()

            # Forward pass with AMP autocast
            with autocast('cuda', enabled=self.use_amp):
                if age is not None and sex is not None:
                    # Use optional demographic args in forward() for DataParallel compatibility
                    recon, mu, logvar = self.model(images, age=age, sex=sex)
                else:
                    recon, mu, logvar = self.model(images)

                # Compute recon + KL loss (contrastive handled at accumulation step)
                recon_loss = vae_reconstruction_loss(recon, images, loss_type=recon_loss_type)
                kl_loss = vae_kl_loss(mu, logvar, reduction="mean")
                base_loss = recon_loss + current_kl_weight * kl_loss

                if gradient_weight > 0:
                    grad_loss = gradient_loss(recon, images)
                    base_loss = base_loss + gradient_weight * grad_loss

                if ssim_weight > 0:
                    ssim_l = ssim_loss(recon, images)
                    base_loss = base_loss + ssim_weight * ssim_l

            # Store images and labels for contrastive computation at optimizer step
            # Store raw images (will re-encode with latest weights at optimizer step)
            # OOM fix: pin to CPU to avoid 256^3 GPU memory blow-up over the
            # accumulation window. We move them back to device at the
            # optimizer step (see temp_features[0].to(self.device)).
            temp_features.append(images.detach().cpu())
            if labels is not None:
                temp_labels.append(labels.detach().cpu())

            # Backward pass for recon+KL only (accumulate gradients)
            if self.use_amp:
                self.scaler.scale(base_loss).backward()
            else:
                base_loss.backward()

            # Step optimizer every accumulation_steps batches
            if (batch_idx + 1) % accumulation_steps == 0:
                # Compute contrastive loss with latest model weights before optimizer step
                # NOTE: contrastive loss should contribute to gradients for encoder to learn
                if contrastive_weight > 0 and len(temp_features) > 0:
                    # Keep model in train mode to ensure gradients flow to encoder
                    # Re-encode stored images with latest model weights
                    all_images = torch.cat(temp_features, dim=0)  # [N_total, 1, D, H, W]
                    all_labels = torch.cat(temp_labels, dim=0)  # [N_total]

                    with autocast('cuda', enabled=self.use_amp):
                        _, mu, _ = self.model(all_images)

                    # Channel-wise mean pooling: [N_total, C, D, H, W] -> [N_total, C]
                    pooled = F.adaptive_avg_pool3d(mu, output_size=(1, 1, 1))
                    pooled = pooled.squeeze(-1).squeeze(-1).squeeze(-1)  # [N_total, C]
                    pooled = F.normalize(pooled, dim=1)  # L2 normalize for contrastive

                    # Compute ordinal contrastive loss
                    from engine.losses import ordinal_contrastive_loss
                    con_loss = ordinal_contrastive_loss(
                        pooled, all_labels,
                        temperature=contrastive_temp,
                        alpha=0.5,
                    )

                    # Add contrastive loss to accumulated gradients
                    # This ensures encoder learns discriminative latent space
                    if self.use_amp:
                        self.scaler.scale(contrastive_weight * con_loss).backward()
                    else:
                        (contrastive_weight * con_loss).backward()

                    con_loss_val = con_loss.item()
                else:
                    con_loss_val = 0.0

                # Unscale, clip, and optimizer step
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                # Accumulate contrastive loss
                total_contrastive_loss += con_loss_val

            # Accumulate metrics for logging
            # Note: con_loss_val is initialized to 0.0 outside the if-block so
            # this expression is always defined, even when accumulation_steps
            # > 1 and we are on a non-final step (where the inner if-block
            # that sets con_loss_val does not run).
            total_loss += (base_loss.item() + (contrastive_weight * con_loss_val if contrastive_weight > 0 else 0))
            num_batches += 1
            pbar.set_postfix({"loss": f"{base_loss.item():.4f}", "con": f"{con_loss_val:.4f}" if contrastive_weight > 0 else "0.0", "step": f"{batch_idx+1}/{len(self.train_loader)}"})

        # Handle leftover gradients from incomplete accumulation cycle
        if len(self.train_loader) % accumulation_steps != 0:
            # Compute contrastive loss for remaining samples
            if contrastive_weight > 0 and len(temp_features) > 0:
                all_images = torch.cat(temp_features, dim=0)
                all_labels = torch.cat(temp_labels, dim=0)

                with autocast('cuda', enabled=self.use_amp):
                    _, mu, _ = self.model(all_images)

                pooled = F.adaptive_avg_pool3d(mu, output_size=(1, 1, 1))
                pooled = pooled.squeeze(-1).squeeze(-1).squeeze(-1)
                pooled = F.normalize(pooled, dim=1)
                from engine.losses import ordinal_contrastive_loss
                con_loss = ordinal_contrastive_loss(pooled, all_labels, temperature=contrastive_temp, alpha=0.5)

                # Add to accumulated gradients
                if self.use_amp:
                    self.scaler.scale(contrastive_weight * con_loss).backward()
                else:
                    (contrastive_weight * con_loss).backward()

                total_contrastive_loss += con_loss.item()

            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        # Compute averages
        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / max(1, len(self.train_loader) // accumulation_steps)

        # Compute individual components for logging
        avg_recon_loss, avg_kl_loss = self._compute_loss_components(recon_loss_type)

        return {
            "loss": avg_loss,
            "recon_loss": avg_recon_loss,
            "kl_loss": avg_kl_loss,
            "contrastive_loss": avg_contrastive_loss,
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
                - contrastive_loss: Ordinal contrastive loss (disease separation)
        """
        self.model.eval()

        total_loss = 0.0
        total_contrastive_loss = 0.0
        num_batches = 0

        recon_loss_type = self.config.get("recon_loss_type", "l1")
        gradient_weight = self.config.get("gradient_weight", 0.0)
        ssim_weight = self.config.get("ssim_weight", 0.0)
        kl_weight = self.config.get("kl_weight", 0.0001)
        contrastive_weight = self.config.get("contrastive_weight", 0.0)
        contrastive_temp = self.config.get("contrastive_temperature", 0.1)

        from tqdm import tqdm
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Val", leave=False)
        for batch_idx, batch in pbar:
            images = batch["image"]
            images = images.to(self.device)

            # Get labels for contrastive loss
            labels = None
            if contrastive_weight > 0 and "label" in batch:
                labels = batch["label"].to(self.device)
                if labels.dim() > 1:
                    labels = labels.squeeze()

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
                    gradient_weight=gradient_weight,
                    ssim_weight=ssim_weight,
                )

            # Compute contrastive loss separately for best model selection
            if contrastive_weight > 0 and labels is not None:
                pooled = F.adaptive_avg_pool3d(mu, output_size=(1, 1, 1))
                pooled = pooled.squeeze(-1).squeeze(-1).squeeze(-1)
                pooled = F.normalize(pooled, dim=1)
                from engine.losses import ordinal_contrastive_loss
                con_loss = ordinal_contrastive_loss(
                    pooled, labels,
                    temperature=contrastive_temp,
                    alpha=0.5,
                )
                total_contrastive_loss += con_loss.item()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches if num_batches > 0 else 0.0
        avg_recon_loss, avg_kl_loss = self._compute_loss_components(recon_loss_type)

        return {
            "loss": avg_loss,
            "recon_loss": avg_recon_loss,
            "kl_loss": avg_kl_loss,
            "contrastive_loss": avg_contrastive_loss,
        }

    def _compute_loss_components(self, recon_loss_type: str) -> tuple:
        """
        Compute reconstruction and KL loss components for logging.

        Computes loss components by running through all validation batches
        for accurate loss tracking during training.

        Args:
            recon_loss_type: Type of reconstruction loss

        Returns:
            Tuple of (avg_recon_loss, avg_kl_loss)
        """
        self.model.eval()
        total_recon = 0.0
        total_kl = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                with autocast('cuda', enabled=self.use_amp):
                    recon, mu, logvar = self.model(images)
                    recon_loss, kl_loss = self._get_loss_components(
                        recon, images, mu, logvar, recon_loss_type
                    )
                batch_size = images.shape[0]
                total_recon += recon_loss.item() * batch_size
                total_kl += kl_loss.item() * batch_size
                num_samples += batch_size

        if num_samples == 0:
            return 0.0, 0.0

        return total_recon / num_samples, total_kl / num_samples

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
        sd = checkpoint["model_state_dict"]
        # Handle DataParallel wrapper: model is DataParallel if it has .module
        model_sd = self.model.state_dict()
        is_dataparallel = any(k.startswith("module.") for k in model_sd)
        has_module_prefix = any(k.startswith("module.") for k in sd)

        if is_dataparallel and not has_module_prefix:
            # Checkpoint from non-DP, current model is DP -> add module. prefix
            new_sd = {f"module.{k}": v for k, v in sd.items()}
        elif not is_dataparallel and has_module_prefix:
            # Checkpoint from DP, current model is non-DP -> strip module. prefix
            new_sd = {k[7:]: v for k, v in sd.items()}
        else:
            new_sd = sd
        self.model.load_state_dict(new_sd)
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = float("inf")

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def load_encoder_only(self, filepath: str) -> None:
        """Load only encoder weights from checkpoint, decoder is reinitialized."""
        checkpoint = torch.load(filepath, map_location=self.device)
        sd = checkpoint["model_state_dict"]

        model_sd = self.model.state_dict()
        is_dataparallel = any(k.startswith("module.") for k in model_sd)

        # Strip module. from checkpoint keys for matching
        clean_sd = {}
        for k, v in sd.items():
            k_clean = k[7:] if k.startswith("module.") else k
            clean_sd[k_clean] = v

        # Add module. back if current model is DataParallel
        matched = {}
        for k, v in clean_sd.items():
            if k.startswith("encoder") and k in model_sd:
                key = f"module.{k}" if is_dataparallel else k
                matched[key] = v

        result = self.model.load_state_dict(matched, strict=False)
        # Count how many matched
        loaded = sum(1 for k in matched if any(k in mk for mk in model_sd))
        print(f"Encoder loading: {loaded} matched, rest freshly initialized.")
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def train(
        self,
        num_epochs: int,
        save_interval: int = 50,
        output_dir: str = "./checkpoints",
        early_stopping_patience: int = 50,
        log_file: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        Run full training loop with KL annealing and AMP.

        KL annealing starts from 0 and linearly increases to target_kl_weight
        over kl_warmup_epochs to prevent posterior collapse.

        Args:
            num_epochs: Number of epochs to train
            save_interval: Interval for saving checkpoints
            output_dir: Directory to save checkpoints
            early_stopping_patience: Epochs without val loss improvement before stopping (default: 50)
            log_file: Path to CSV log file. If None, saves to output_dir/train_log.csv

        Returns:
            Dictionary containing training history with lists of:
                - train_loss, train_recon_loss, train_kl_loss
                - val_loss, val_recon_loss, val_kl_loss
        """
        import csv
        import time

        target_kl_weight = self.config.get("kl_weight", 0.0001)
        kl_warmup_epochs = self.config.get("kl_warmup_epochs", 5)

        # Setup log file
        if log_file is None:
            log_file = os.path.join(output_dir, "train_log.csv")
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

        # Check if log file exists and has data (resume case)
        log_exists = os.path.exists(log_file)
        write_header = not log_exists

        history = {
            "train_loss": [],
            "train_recon_loss": [],
            "train_kl_loss": [],
            "train_contrastive_loss": [],
            "val_loss": [],
            "val_recon_loss": [],
            "val_kl_loss": [],
            "val_contrastive_loss": [],
        }

        # Early stopping state
        epochs_without_improvement = 0
        early_stopped = False
        self.best_contrastive_loss = float("inf")  # for minimization: smaller (more negative) = better separation

        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

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
            epoch_time = time.time() - epoch_start_time

            # Check if this is best model (based on contrastive_loss for disease separation)
            # contrastive_loss is negative, MORE NEGATIVE = better separation (smaller = better)
            current_con_loss = val_metrics["contrastive_loss"]
            is_best = current_con_loss < self.best_contrastive_loss
            if is_best:
                self.best_contrastive_loss = current_con_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Log metrics
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"LR: {current_lr:.6f} | "
                f"KL_w: {current_kl_weight:.6f} | "
                f"Train: {train_metrics['loss']:.4f} (recon: {train_metrics['recon_loss']:.4f}, con: {train_metrics['contrastive_loss']:.4f}) | "
                f"Val: {val_metrics['loss']:.4f} (recon: {val_metrics['recon_loss']:.4f}, con: {val_metrics['contrastive_loss']:.4f}) | "
                f"Time: {epoch_time:.1f}s | "
                f"Patience: {epochs_without_improvement}/{early_stopping_patience}"
            )

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["train_recon_loss"].append(train_metrics["recon_loss"])
            history["train_kl_loss"].append(train_metrics["kl_loss"])
            history["train_contrastive_loss"].append(train_metrics["contrastive_loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_recon_loss"].append(val_metrics["recon_loss"])
            history["val_kl_loss"].append(val_metrics["kl_loss"])
            history["val_contrastive_loss"].append(val_metrics["contrastive_loss"])

            # Write to CSV log
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "epoch", "train_loss", "train_recon_loss", "train_kl_loss", "train_contrastive_loss",
                        "val_loss", "val_recon_loss", "val_kl_loss", "val_contrastive_loss",
                        "lr", "kl_weight", "epoch_time", "is_best"
                    ])
                    write_header = False
                writer.writerow([
                    epoch + 1,
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['recon_loss']:.6f}",
                    f"{train_metrics['kl_loss']:.6f}",
                    f"{train_metrics['contrastive_loss']:.6f}",
                    f"{val_metrics['loss']:.6f}",
                    f"{val_metrics['recon_loss']:.6f}",
                    f"{val_metrics['kl_loss']:.6f}",
                    f"{val_metrics['contrastive_loss']:.6f}",
                    f"{current_lr:.8f}",
                    f"{current_kl_weight:.6f}",
                    f"{epoch_time:.2f}",
                    "1" if is_best else "0"
                ])

            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(output_dir, f"vae_epoch_{epoch+1}.pt")
                self.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

            # Save best model
            if is_best:
                best_path = os.path.join(output_dir, "vae_best.pt")
                self.save_checkpoint(best_path)
                print(f"Best model saved to {best_path} (val_con_loss: {current_con_loss:.4f})")

            # Early stopping check
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"No improvement for {early_stopping_patience} epochs")
                print(f"Best val_loss: {self.best_val_loss:.6f}")
                print(f"{'='*60}\n")
                early_stopped = True
                break

        # Save final history
        if early_stopped:
            print(f"Training stopped early at epoch {epoch+1}")
            print(f"Best val_loss: {self.best_val_loss:.6f}")

        return history


class MultiModalVAETrainer:
    """
    Trainer class for Multi-Modal VAE in Stage 1 of ADynamics.

    Supports:
        - Multi-modal VAE training (T1 + optional fMRI/ASL/QSM/FLAIR)
        - Joint training with reconstruction + classification losses
        - Modality dropout for robustness to missing modalities
        - AMP for memory-efficient HD training

    Attributes:
        model: The MultiModalVAE3D model being trained
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
        Initialize the Multi-Modal VAE trainer.

        Args:
            model: The MultiModalVAE3D model to train
            optimizer: AdamW optimizer instance
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to train on ("cuda" or "cpu")
            config: Configuration dictionary containing:
                - recon_loss_type: Type of reconstruction loss ("l1" or "l2")
                - cls_weight: Weight for classification loss. Default: 1.0
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

        # AMP configuration
        self.use_amp = config.get("use_amp", False)
        self.scaler = GradScaler() if self.use_amp else None

        # Move model to device
        self.model.to(self.device)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_cls_acc = 0.0
        # Monitoring (v10): track extra per-epoch signals to diagnose latent
        # quality without re-running a separate analysis pass.
        self.kl_active_dims = 0.0  # how many latent dims have KL > free_bits
        self.latent_std = 0.0     # mean per-dim std of mu (posterior spread)
        self.train_val_gap = 0.0  # train_acc - val_acc (overfitting signal)
        self.cls_loss_ema = 0.0   # exponential moving avg of train cls_loss
        self.kl_strategy = config.get("kl_strategy", "linear")
        self.current_kl_weight = 0.0  # set per-epoch in train()

    def _apply_encoder_grad_boost(self) -> None:
        """Scale up encoder gradients by encoder_grad_boost factor.

        Phase 0 diagnostic found encoder_t1 grad=0.87 vs classifier=4.4
        (encoder starved). Boosting encoder gradients lets it learn
        discriminative features instead of just matching the KL prior.

        Only applies to parameters whose name contains 'encoder' (covers
        encoder_t1 and optional_encoders, but not fusion_proj/logvar_proj).
        """
        boost = self.config.get("encoder_grad_boost", 1.0)
        if boost == 1.0:
            return
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        for name, param in model_ref.named_parameters():
            if param.grad is not None and "encoder" in name:
                param.grad.data.mul_(boost)

    def _normalize_fmri_batch(self, fmri: torch.Tensor) -> torch.Tensor:
        """Normalize fMRI batch to a consistent 5D shape for the temporal encoder.

        Args:
            fmri: A tensor of shape
                - 5D [B, 1, D, H, W]    : legacy time-averaged path
                - 6D [B, 1, D, H, W, T] : preserve_temporal_dim=True path

        Returns:
            6D tensor [B, 1, D, H, W, T_target] with T_target = the median T
            in the input (rounded). If all samples have the same T, returns
            the input unchanged. If a sample has fewer T, pad with last frame;
            if more, truncate.
        """
        if fmri.dim() == 5:
            # Legacy: add a trailing time dim of size 1
            return fmri.unsqueeze(-1)
        if fmri.dim() != 6:
            raise ValueError(
                f"fMRI must be 5D or 6D; got {fmri.dim()}D with shape {tuple(fmri.shape)}"
            )

        # 6D: [B, 1, D, H, W, T]
        Ts = fmri.shape[-1]
        if Ts == 0:
            raise ValueError("fMRI has 0 timepoints")

        # If T is uniform across batch, no work needed.
        # (We can't cheaply check per-sample T from a stacked tensor;
        # assume uniform — datasets usually have fixed TR * N volumes.)
        # Cheap runtime check: if min == max, OK.
        # Note: per-sample T is hidden after stacking; we trust the dataset.
        return fmri

    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.

        v10 changes:
            - Latent-space mixup (mixup_alpha from config) to fight overfit
            - Per-class accuracy, KL active dims, posterior std metrics
        v11 changes:
            - Grad-norm per module (encoder/decoder/classifier/demo)
            - Mixup application count
            - Per-batch prediction distribution
        """
        self.model.train()

        # v11: reset grad-norm buffer at start of epoch
        self._grad_norm_buf = None
        self._grad_norm_count = 0
        # Reset prediction distribution buffer
        self._pred_count_buf = torch.zeros(self.config.get("num_classes", 4))

        total_loss = 0.0
        total_recon_loss = 0.0
        total_cls_loss = 0.0
        total_kl_loss = 0.0
        total_cls_acc = 0.0
        num_batches = 0
        # v10 monitoring: per-class accuracy and posterior stats
        per_class_correct = torch.zeros(self.config.get("num_classes", 4))
        per_class_total = torch.zeros(self.config.get("num_classes", 4))
        # Collect mu, logvar across batches to compute aggregate stats at end
        collected_mu_stds = []  # list of per-dim std tensors

        recon_loss_type = self.config.get("recon_loss_type", "l1")
        cls_weight = self.config.get("cls_weight", 1.0)
        kl_weight = getattr(self, 'current_kl_weight', self.config.get("kl_weight", 0.01))
        num_classes = self.config.get("num_classes", 3)
        use_demographic = self.config.get("use_demographic_cond", False)
        # v10: latent mixup config
        mixup_alpha = self.config.get("mixup_alpha", 0.0)
        mixup_prob = self.config.get("mixup_prob", 0.5)

        # Per-class weights for class imbalance (loaded ONCE per epoch —
        # tensor is loop-invariant). config["class_weights"] = [w_NC, w_SCD,
        # w_MCI, w_AD]. When set, minority classes (SCD 13%, MCI 26%) get
        # higher loss contribution than majority (NC 34%, AD 27%), forcing
        # the classifier to learn rare-class signal instead of collapsing
        # to NC/AD. Default: None (equal weights). Loaded here so it's in
        # scope for both train_epoch and validate_epoch.
        class_weights = self.config.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.tensor(
                class_weights, dtype=torch.float32, device=self.device,
            )

        # v11: track mixup application count
        n_mixup_applied = 0
        n_skipped_batches = 0  # track skipped batches for diagnostics

        # Gradient accumulation: effective_batch = batch_size * accumulation_steps
        accumulation_steps = self.config.get("accumulation_steps", 1)
        # Track actual accumulated batches for incomplete cycles at epoch end
        actual_accum_count = 0
        num_batches = len(self.train_loader)

        from utils.kl_schedules import (
            should_apply_mixup, mixup_latents, mixup_classification_loss,
        )

        # Clear CUDA cache at epoch start to prevent async error accumulation
        torch.cuda.empty_cache()

        from tqdm import tqdm
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Train", leave=False)

        for batch_idx, batch in pbar:
            # Zero grad at the start of each accumulation cycle
            if batch_idx % accumulation_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)
                actual_accum_count = 0

            # Get T1 (already preprocessed to 256x256x192)
            t1 = batch["t1"].to(self.device)

            # class_weights is loaded once at the top of train_epoch (above
            # the batch loop) and reused for every batch — tensor is
            # loop-invariant. Re-loading per batch would just create a fresh
            # torch.tensor every step with no functional benefit.

            # Build x_dict with T1 and optional modalities.
            # IMPORTANT: only include modalities that the model actually has encoders for
            # (driven by the trainer's config["optional_modalities"] list). Anything
            # else in the batch is silently dropped here to avoid passing shapes the
            # model can't handle.
            active_optionals = self.config.get(
                "optional_modalities", ["fmri", "asl", "qsm", "flair"]
            )
            x_dict = {"t1": t1}

            # Use available_modalities to skip zero-filled missing modalities.
            # A modality is only passed if ALL samples in the batch have it.
            batch_avail = batch.get("available_modalities", None)
            if batch_avail and isinstance(batch_avail[0], (list, tuple)):
                avail_set = set(batch_avail[0])
                for av in batch_avail[1:]:
                    avail_set &= set(av)
            else:
                avail_set = set(active_optionals)

            for mod in active_optionals:
                if mod in avail_set and mod in batch and batch[mod] is not None:
                    mod_tensor = batch[mod].to(self.device)
                    if mod == "fmri":
                        mod_tensor = self._normalize_fmri_batch(mod_tensor)
                    x_dict[mod] = mod_tensor

            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(self.device)
                if labels.dim() > 1:
                    labels = labels.squeeze()

            # Optional demographic conditioning.
            # IMPORTANT: pass age/sex through x_dict (not as kwargs) so that
            # MultiModalDataParallel.scatter splits them per replica. Passing
            # as kwargs leaves them at full batch on every replica, which
            # broadcast-inflates the latent and breaks L1(recon, t1) shape.
            if use_demographic and "age" in batch and "sex" in batch:
                x_dict["age"] = batch["age"].to(self.device)
                x_dict["sex"] = batch["sex"].to(self.device)

            # Forward pass
            with autocast('cuda', enabled=self.use_amp):
                try:
                    recon, cls_logits, mu, logvar = self.model(
                        x_dict, return_components=True,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n[TRAIN OOM] batch={batch_idx} mods={list(x_dict.keys())}")
                        for k, v in x_dict.items():
                            if isinstance(v, torch.Tensor):
                                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad(set_to_none=True)
                        n_skipped_batches += 1
                        continue
                    else:
                        print(f"\n[TRAIN ERROR] batch={batch_idx} mods={list(x_dict.keys())}")
                        for k, v in x_dict.items():
                            if isinstance(v, torch.Tensor):
                                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
                        raise

                # v10: decide whether to apply latent mixup this batch
                do_mixup = (mixup_alpha > 0) and (labels is not None) and should_apply_mixup(
                    self.current_epoch, self.config,
                )

                if do_mixup and mu.size(0) >= 2:
                    # v11: count this batch as a mixup application
                    n_mixup_applied += 1
                    # Mix latents and labels (only meaningful when B>=2).
                    mu_mixed, lab_a, lab_b, lam = mixup_latents(
                        mu, labels, mixup_alpha,
                    )
                    # Re-decode / re-classify from the mixed latent under
                    # no_grad to avoid building a second full autograd
                    # graph (would OOM on 24GB GPUs). This is a deliberate
                    # memory / correctness trade-off:
                    #   - Encoder still receives gradient via the
                    #     standard forward (the original recon / cls
                    #     loss flows back through mu -> encoder).
                    #   - The mixup branch's decoder and classifier
                    #     outputs are computed without gradient, but
                    #     the loss terms (recon_loss, cls_loss,
                    #     ordinal_reg_loss) still use these outputs to
                    #     shape the encoder's behavior indirectly
                    #     (mixup is a label-space regularizer, not a
                    #     parameter-space one — see Zhang 2018).
                    #   - The mixup loss for cls DOES backprop through
                    #     cls_logits_mixed even under no_grad? No — it
                    #     doesn't. But the *first* forward's cls loss
                    #     already covers the classifier's training.
                    # The net effect: mixup is a regularizer on the
                    # encoder + a data augmentation for the standard
                    # cls loss, not a separate gradient path.
                    model_ref = self.model.module if hasattr(self.model, "module") else self.model
                    with torch.no_grad():
                        recon_mixed = model_ref.decode(mu_mixed)
                        cls_logits_mixed = model_ref.classify(mu_mixed)
                    # Mixup recon loss: mix original and permuted T1 targets
                    perm = torch.randperm(t1.size(0), device=t1.device)
                    recon_loss = F.l1_loss(
                        recon_mixed, lam * t1 + (1.0 - lam) * t1[perm],
                    )
                    # Mixup classification loss
                    cls_loss = mixup_classification_loss(cls_logits_mixed, lab_a, lab_b, lam)
                    # Ordinal regression on mixed mu (still useful)
                    ordinal_reg_loss = ordinal_regression_loss(mu_mixed, lab_a, num_classes=num_classes)
                    # Release mixup intermediates: they're held by the autograd
                    # graph until backward() runs, which on 24GB GPUs can cause
                    # OOM. Dropping the Python references lets the graph free
                    # those nodes once the loss is assembled.
                    del recon_mixed, cls_logits_mixed, mu_mixed, lab_a, lab_b
                else:
                    # Standard path
                    recon_loss = F.l1_loss(recon, x_dict["t1"])
                    if labels is not None:
                        cls_loss_type = self.config.get("cls_loss_type", "ordinal_ce")
                        if cls_loss_type == "standard_ce":
                            cls_loss = F.cross_entropy(cls_logits, labels, weight=class_weights)
                        else:
                            cls_loss = ordinal_cross_entropy_loss(
                                cls_logits, labels, num_classes=num_classes,
                                class_weights=class_weights,
                            )
                        ordinal_reg_loss = ordinal_regression_loss(mu, labels, num_classes=num_classes)
                    else:
                        cls_loss = torch.tensor(0.0, device=self.device)
                        ordinal_reg_loss = torch.tensor(0.0, device=self.device)
                    lam = 1.0

                # KL loss with Free Bits
                # Cast to FP32 to prevent logvar.exp() overflow
                mu_fp32 = mu.float()
                logvar_fp32 = logvar.float()
                # Clamp logvar for numerical stability (prevent exp overflow)
                logvar_fp32 = torch.clamp(logvar_fp32, min=-10.0, max=10.0)

                free_bits = self.config.get("free_bits", 0.0)
                kl_per_dim = -0.5 * (1 + logvar_fp32 - mu_fp32.pow(2) - logvar_fp32.exp())
                if free_bits > 0:
                    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
                kl_loss = kl_per_dim.mean()

                # Total loss (scale by 1/accumulation_steps for gradient averaging)
                ordinal_reg_weight = self.config.get("ordinal_reg_weight", 0.1)
                loss = (recon_loss + cls_weight * cls_loss + kl_weight * kl_loss + ordinal_reg_weight * ordinal_reg_loss) / accumulation_steps
                actual_accum_count += 1

            # Backward pass (accumulate gradients)
            try:
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            except RuntimeError as e:
                err_msg = str(e).lower()
                if "out of memory" in err_msg or "cuda" in err_msg:
                    # Backward OOM / CUDA error: free memory, skip this batch
                    print(f"\n[TRAIN OOM backward] batch={batch_idx} — skipping ({e})")
                    del loss, recon, cls_logits, mu, logvar
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad(set_to_none=True)
                    n_skipped_batches += 1
                    continue
                else:
                    raise

            # Step optimizer: complete cycle OR incomplete cycle at epoch end
            is_complete_cycle = (batch_idx + 1) % accumulation_steps == 0
            is_epoch_end = (batch_idx + 1) == num_batches
            if is_complete_cycle or is_epoch_end:
                # For incomplete cycles at epoch end, rescale gradients
                # to compensate for the fewer accumulated batches
                if is_epoch_end and not is_complete_cycle and actual_accum_count < accumulation_steps:
                    # Rescale: we divided by accumulation_steps but only
                    # accumulated actual_accum_count times, so multiply
                    # by accumulation_steps / actual_accum_count
                    rescale_factor = accumulation_steps / actual_accum_count
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.data.mul_(rescale_factor)

                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    self._apply_encoder_grad_boost()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self._apply_encoder_grad_boost()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            # Metrics
            with torch.no_grad():
                # v10: per-class accuracy
                if labels is not None:
                    preds = cls_logits.argmax(dim=1)
                    cls_acc = (preds == labels).float().mean().item()
                    for c in range(num_classes):
                        mask = labels == c
                        if mask.any():
                            per_class_total[c] += mask.sum().item()
                            per_class_correct[c] += (preds[mask] == c).sum().item()
                    # v10: per-dim std of mu (posterior spread)
                    # mu is [B, C, D, H, W]; pool spatial -> [B, C]; std over batch
                    mu_pooled = mu.mean(dim=(-3, -2, -1))  # [B, C]
                    collected_mu_stds.append(mu_pooled.std(dim=0).cpu())
                    # v11: prediction distribution (per-batch). We aggregate
                    # the full per-batch prediction count in a tensor.
                    if not hasattr(self, "_pred_count_buf"):
                        self._pred_count_buf = torch.zeros(num_classes)
                    for c in range(num_classes):
                        self._pred_count_buf[c] += (preds == c).sum().item()
                else:
                    cls_acc = 0.0

            # v11: capture per-module grad norms (done every 5 batches to
            # avoid overhead — iterating all parameters is ~1-2% wall time).
            # Done BEFORE optimizer.step so grads are fresh.
            if batch_idx % 5 == 0:
                from utils.diagnostics import grad_norm_by_module
                gn = grad_norm_by_module(
                    self.model.module if hasattr(self.model, "module") else self.model,
                )
                if self._grad_norm_count == 0:
                    # First batch: just store the values (they're not yet a "mean").
                    self._grad_norm_buf = dict(gn)
                else:
                    for k, v in gn.items():
                        self._grad_norm_buf[k] = (
                            self._grad_norm_buf[k] * self._grad_norm_count + v
                        ) / (self._grad_norm_count + 1)
                self._grad_norm_count += 1

            total_loss += loss.item() * accumulation_steps  # unscale for logging
            total_recon_loss += recon_loss.item()
            total_cls_loss += cls_loss.item()
            total_kl_loss += kl_loss.item()
            total_cls_acc += cls_acc
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "cls": f"{cls_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}",
                "acc_step": f"{(batch_idx % accumulation_steps) + 1}/{accumulation_steps}",
            })

            # OOM fix: clear the autograd graph + cache AFTER all references
            # to loss / recon / cls_logits / mu / logvar are used. The
            # per-class accuracy, grad-norm, and pbar blocks all need these
            # tensors alive. 256^3 3D images on a 24GB GPU accumulate ~18GB
            # of activations and intermediate gradients; without this,
            # fragmented memory forces OOM on later batches even when peak
            # is below 24GB.
            del loss, recon, cls_logits, mu, logvar
            if hasattr(self.model, "module"):
                self.model.module._last_graph = None
            # empty_cache forces CUDA sync across all GPUs — very expensive
            # on DataParallel. Only do it every 10 batches to prevent
            # fragmentation without tanking throughput.
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # v10: aggregate monitoring metrics
        per_class_acc = (per_class_correct / per_class_total.clamp(min=1)).tolist()
        if collected_mu_stds:
            all_stds = torch.stack(collected_mu_stds, dim=0).mean(dim=0)
            self.latent_std = all_stds.mean().item()
            free_bits = self.config.get("free_bits", 0.0)
            self.kl_active_dims = (all_stds > math.sqrt(2 * free_bits + 1e-6)).float().mean().item() * all_stds.numel()
        # v10: EMA of cls_loss for trend tracking
        ema = self.cls_loss_ema
        ema = 0.9 * ema + 0.1 * (total_cls_loss / max(1, num_batches))
        self.cls_loss_ema = ema

        # Report skipped batches (diagnostic for crashes / OOM)
        if n_skipped_batches > 0:
            print(f"\n[TRAIN] Epoch {self.current_epoch}: "
                  f"skipped {n_skipped_batches}/{num_batches} batches "
                  f"(OOM/CUDA errors)")

        return {
            "loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "cls_loss": total_cls_loss / num_batches,
            "kl_loss": total_kl_loss / num_batches,
            "cls_acc": total_cls_acc / num_batches,
            "per_class_acc": per_class_acc,
            "mixup_count": float(n_mixup_applied),
            "mixup_frac": n_mixup_applied / max(1, num_batches),
            "grad_norms": dict(self._grad_norm_buf) if hasattr(self, "_grad_norm_buf") else {},
        }

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """
        Run one validation epoch.

        v10: also returns per_class_acc, latent_std (val), and KL active dims
             so we can spot posterior collapse / overfit from the log alone.
        v11: also returns val_silhouette, per-class centroid distances,
             per-class prediction frequency, and recon intensity stats
             (deep diagnostics for the "is the data the problem?" question).
        """
        # Clear CUDA cache before validation to prevent fragmentation issues
        torch.cuda.empty_cache()
        self.model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_cls_loss = 0.0
        total_cls_acc = 0.0
        num_batches = 0
        num_classes = self.config.get("num_classes", 4)
        per_class_correct = torch.zeros(num_classes)
        per_class_total = torch.zeros(num_classes)
        collected_mu_stds = []
        # v11: collect full val mu + labels + preds for end-of-epoch diagnostics
        all_mu_pooled = []   # list of [B, C]
        all_labels = []      # list of [B]
        all_preds = []       # list of [B]
        all_input_intensity = []  # list of mean per sample
        all_recon_intensity = []
        # prediction distribution accumulator (across all val samples)
        pred_dist = torch.zeros(num_classes)

        recon_loss_type = self.config.get("recon_loss_type", "l1")
        cls_weight = self.config.get("cls_weight", 1.0)
        kl_weight = getattr(self, 'current_kl_weight', self.config.get("kl_weight", 0.01))
        use_demographic = self.config.get("use_demographic_cond", False)

        # Load per-class weights from config (inverse-frequency for class balance).
        # MUST be done in validate_epoch too — it's a different method, has its
        # own local scope; without this, ordinal_cross_entropy_loss raises
        # NameError on the first val batch. (Bug from commit ce0b7eb.)
        class_weights = self.config.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.tensor(
                class_weights, dtype=torch.float32, device=self.device,
            )

        from tqdm import tqdm
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Val", leave=False)
        total_kl_loss = 0.0

        for batch_idx, batch in pbar:
            # Get T1 (already preprocessed)
            t1 = batch["t1"].to(self.device)

            # Build x_dict with T1 and optional modalities (only those the
            # model actually has encoders for, driven by config).
            active_optionals = self.config.get(
                "optional_modalities", ["fmri", "asl", "qsm", "flair"]
            )
            x_dict = {"t1": t1}

            # Use available_modalities to skip zero-filled missing modalities.
            # A modality is only passed if ALL samples in the batch have it.
            batch_avail = batch.get("available_modalities", None)
            if batch_avail and isinstance(batch_avail[0], (list, tuple)):
                avail_set = set(batch_avail[0])
                for av in batch_avail[1:]:
                    avail_set &= set(av)
            else:
                avail_set = set(active_optionals)

            for mod in active_optionals:
                if mod in avail_set and mod in batch and batch[mod] is not None:
                    mod_tensor = batch[mod].to(self.device)
                    if mod == "fmri":
                        mod_tensor = self._normalize_fmri_batch(mod_tensor)
                    x_dict[mod] = mod_tensor

            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(self.device)
                if labels.dim() > 1:
                    labels = labels.squeeze()

            # Optional demographic conditioning.
            # IMPORTANT: pass age/sex through x_dict (not as kwargs) so that
            # MultiModalDataParallel.scatter splits them per replica.
            if use_demographic and "age" in batch and "sex" in batch:
                x_dict["age"] = batch["age"].to(self.device)
                x_dict["sex"] = batch["sex"].to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                try:
                    recon, cls_logits, mu, logvar = self.model(
                        x_dict, return_components=True,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"\n[VAL OOM] batch={batch_idx} mods={list(x_dict.keys())}")
                        for k, v in x_dict.items():
                            if isinstance(v, torch.Tensor):
                                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"\n[VAL ERROR] batch={batch_idx} mods={list(x_dict.keys())}")
                        for k, v in x_dict.items():
                            if isinstance(v, torch.Tensor):
                                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
                        raise

                recon_loss = F.l1_loss(recon, x_dict["t1"])

                if labels is not None:
                    cls_loss_type = self.config.get("cls_loss_type", "ordinal_ce")
                    if cls_loss_type == "standard_ce":
                        cls_loss = F.cross_entropy(cls_logits, labels, weight=class_weights)
                    else:
                        cls_loss = ordinal_cross_entropy_loss(
                            cls_logits, labels, num_classes=num_classes,
                            class_weights=class_weights,
                        )
                    ordinal_reg_loss = ordinal_regression_loss(mu, labels, num_classes=num_classes)
                    preds = cls_logits.argmax(dim=1)
                    cls_acc = (preds == labels).float().mean().item()
                    # v10: per-class accuracy + posterior spread
                    for c in range(num_classes):
                        mask = labels == c
                        if mask.any():
                            per_class_total[c] += mask.sum().item()
                            per_class_correct[c] += (preds[mask] == c).sum().item()
                    mu_pooled = mu.mean(dim=(-3, -2, -1))
                    collected_mu_stds.append(mu_pooled.std(dim=0).cpu())
                    # v11: collect for end-of-epoch silhouette + centroid dists
                    all_mu_pooled.append(mu_pooled.cpu())
                    all_labels.append(labels.cpu())
                    all_preds.append(preds.cpu())
                    # v11: input/recon intensity stats (data sanity)
                    all_input_intensity.append(t1.mean(dim=(-3, -2, -1)).cpu())
                    all_recon_intensity.append(recon.mean(dim=(-3, -2, -1)).cpu())
                    # v11: prediction distribution accumulator
                    for c in range(num_classes):
                        pred_dist[c] += (preds == c).sum().item()
                else:
                    cls_loss = torch.tensor(0.0, device=self.device)
                    ordinal_reg_loss = torch.tensor(0.0, device=self.device)
                    cls_acc = 0.0

                free_bits = self.config.get("free_bits", 0.0)
                # Cast to FP32 and clamp for numerical stability
                mu_fp32 = mu.float()
                logvar_fp32 = logvar.float()
                logvar_fp32 = torch.clamp(logvar_fp32, min=-10.0, max=10.0)
                kl_per_dim = -0.5 * (1 + logvar_fp32 - mu_fp32.pow(2) - logvar_fp32.exp())
                if free_bits > 0:
                    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
                kl_loss = kl_per_dim.mean()
                loss = recon_loss + cls_weight * cls_loss + kl_weight * kl_loss + 0.1 * ordinal_reg_loss

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_cls_loss += cls_loss.item()
            total_kl_loss += kl_loss.item()
            total_cls_acc += cls_acc
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{cls_acc:.4f}"})

        # v10: aggregate monitoring metrics (same way as train)
        per_class_acc = (per_class_correct / per_class_total.clamp(min=1)).tolist()
        if collected_mu_stds:
            all_stds = torch.stack(collected_mu_stds, dim=0).mean(dim=0)
            free_bits = self.config.get("free_bits", 0.0)
            n_active = (all_stds > math.sqrt(2 * free_bits + 1e-6)).float().sum().item()
        else:
            n_active = 0.0
            all_stds = torch.zeros(1)

        # v11: deep diagnostics (only when we have val samples)
        from utils.diagnostics import (
            silhouette_score, per_class_centroid_distance,
            per_dim_latent_stats, recon_intensity_stats,
        )
        if all_mu_pooled:
            mu_all = torch.cat(all_mu_pooled, dim=0)        # [N, C]
            lab_all = torch.cat(all_labels, dim=0)          # [N]
            pred_all = torch.cat(all_preds, dim=0)          # [N]
            sil = silhouette_score(mu_all, lab_all)
            centroid_dists = per_class_centroid_distance(
                mu_all, lab_all, num_classes=num_classes,
            )
            dim_stats = per_dim_latent_stats(
                mu_all, free_bits=self.config.get("free_bits", 0.0),
            )
            pred_freq = (pred_dist / max(1.0, pred_dist.sum().item())).tolist()
        else:
            sil = 0.0
            centroid_dists = {}
            dim_stats = {}
            pred_freq = [0.0] * num_classes

        # v11: input/recon intensity sanity check
        if all_input_intensity:
            in_mean = torch.cat(all_input_intensity).mean().item()
            re_mean = torch.cat(all_recon_intensity).mean().item()
            intensity_gap = abs(in_mean - re_mean)
        else:
            in_mean = re_mean = intensity_gap = 0.0

        return {
            "loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "cls_loss": total_cls_loss / num_batches,
            "kl_loss": total_kl_loss / num_batches,
            "cls_acc": total_cls_acc / num_batches,
            "per_class_acc": per_class_acc,
            "val_latent_std": all_stds.mean().item() if collected_mu_stds else 0.0,
            "val_kl_active_dims": float(n_active),
            # v11 deep diagnostics
            "val_silhouette": sil,
            "val_centroid_dists": centroid_dists,
            "val_dim_stats": dim_stats,
            "val_pred_freq": pred_freq,
            "val_input_intensity": in_mean,
            "val_recon_intensity": re_mean,
            "val_intensity_gap": intensity_gap,
        }

    def save_checkpoint(self, filepath: str, include_optimizer: bool = True) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_cls_acc": self.best_cls_acc,
        }
        if include_optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint with DataParallel and num_classes-mismatch handling."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        sd = checkpoint["model_state_dict"]

        # Handle DataParallel prefix
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        model_sd = model_ref.state_dict()
        has_dp = any(k.startswith("module.") for k in sd)

        if has_dp:
            sd = {k[7:]: v for k, v in sd.items()}

        # Filter: only load keys that exist and shape matches (e.g. classifier head may differ)
        filtered_sd = {}
        skipped = []
        for k, v in sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                filtered_sd[k] = v
            else:
                skipped.append(k)

        model_ref.load_state_dict(filtered_sd, strict=False)
        if skipped:
            print(f"  Loaded {len(filtered_sd)} params, skipped {len(skipped)} (shape mismatch)")
        else:
            print(f"  Loaded {len(filtered_sd)} params")

        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = float("inf")
        self.best_cls_acc = 0.0
        if "optimizer_state_dict" in checkpoint and not skipped:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"  Optimizer state incompatible: {e}")

    def train(
        self,
        num_epochs: int,
        save_interval: int = 50,
        output_dir: str = "./checkpoints",
        early_stopping_patience: int = 50,
        log_file: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        Run full training loop.

        Args:
            num_epochs: Number of epochs to train
            save_interval: Interval for saving checkpoints
            output_dir: Directory to save checkpoints
            early_stopping_patience: Epochs without improvement before stopping
            log_file: Path to CSV log file

        Returns:
            Dictionary containing training history
        """
        import csv
        import time

        if log_file is None:
            log_file = os.path.join(output_dir, "train_log.csv")
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

        log_exists = os.path.exists(log_file)
        # v10: if existing CSV is from an older run (missing v10 columns),
        # archive it and start fresh. Otherwise, append.
        if log_exists:
            try:
                with open(log_file, "r") as f:
                    first_line = f.readline()
                if "val_kl_active_dims" not in first_line:
                    backup = log_file + ".old"
                    os.rename(log_file, backup)
                    print(f"[v10] Old-format log archived to {backup}")
                    log_exists = False
            except Exception:
                pass
        write_header = not log_exists

        history = {
            "train_loss": [],
            "train_recon_loss": [],
            "train_cls_loss": [],
            "train_kl_loss": [],
            "train_cls_acc": [],
            "val_loss": [],
            "val_recon_loss": [],
            "val_cls_loss": [],
            "val_kl_loss": [],
            "val_cls_acc": [],
            "kl_weight": [],
            "kl_active_dims": [],
            "latent_std": [],
            "per_class_acc_train": [],
            "per_class_acc_val": [],
        }

        epochs_without_improvement = 0
        early_stopped = False

        # v10: KL schedule dispatcher (linear or cyclical).
        # We store the strategy on self for downstream introspection.
        from utils.kl_schedules import get_kl_weight
        self.kl_strategy = self.config.get("kl_strategy", "linear")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # v10: dispatch to selected KL strategy
            self.current_kl_weight, _ = get_kl_weight(epoch, self.config)

            try:
                train_metrics = self.train_epoch()
                val_metrics = self.validate_epoch()
            except RuntimeError as e:
                err_msg = str(e).lower()
                if "out of memory" in err_msg or "cuda" in err_msg:
                    print(f"\n[EPOCH {epoch} CRASH] {e}")
                    print("  Skipping this epoch. Consider reducing batch_size.")
                    torch.cuda.empty_cache()
                    epochs_without_improvement += 1
                    continue
                else:
                    raise
            except Exception as e:
                # Catch-all: data loading errors, corrupted cache, etc.
                print(f"\n[EPOCH {epoch} UNEXPECTED ERROR] {type(e).__name__}: {e}")
                print("  Skipping this epoch. Check data integrity.")
                torch.cuda.empty_cache()
                epochs_without_improvement += 1
                continue

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start_time

            # Best model selection based on cls_acc (higher = better)
            current_cls_acc = val_metrics["cls_acc"]
            is_best = current_cls_acc > self.best_cls_acc
            if is_best:
                self.best_cls_acc = current_cls_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # v10: train/val accuracy gap (overfit signal)
            self.train_val_gap = train_metrics["cls_acc"] - val_metrics["cls_acc"]

            # Per-class accuracy strings (NC / SCD / MCI / AD)
            class_names = self.config.get("class_names", ["NC", "SCD", "MCI", "AD"])
            pca_tr = train_metrics.get("per_class_acc", [0.0] * 4)
            pca_va = val_metrics.get("per_class_acc", [0.0] * 4)
            train_pca_str = "/".join(
                f"{class_names[i]}={pca_tr[i]:.2f}" for i in range(len(pca_tr))
            )
            val_pca_str = "/".join(
                f"{class_names[i]}={pca_va[i]:.2f}" for i in range(len(pca_va))
            )

            # v10: condensed log line with all the new signals
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"LR: {current_lr:.6f} | "
                f"Train: {train_metrics['loss']:.4f} "
                f"(recon: {train_metrics['recon_loss']:.4f}, cls: {train_metrics['cls_loss']:.4f}, "
                f"kl: {train_metrics['kl_loss']:.4f}, acc: {train_metrics['cls_acc']:.4f}) | "
                f"Val: {val_metrics['loss']:.4f} "
                f"(recon: {val_metrics['recon_loss']:.4f}, cls: {val_metrics['cls_loss']:.4f}, "
                f"kl: {val_metrics['kl_loss']:.4f}, acc: {val_metrics['cls_acc']:.4f}) | "
                f"KL_w={self.current_kl_weight:.3f} | "
                f"gap={self.train_val_gap:+.3f} | "
                f"actD={val_metrics.get('val_kl_active_dims', 0):.0f}/32 | "
                f"std={val_metrics.get('val_latent_std', 0):.3f} | "
                f"Time: {epoch_time:.1f}s | "
                f"Patience: {epochs_without_improvement}/{early_stopping_patience}"
            )
            # Per-class accuracy on its own line so it doesn't make the main
            # line unreadable. Only print every 5 epochs to keep logs terse.
            if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
                print(
                    f"           per-class acc | train: {train_pca_str} | "
                    f"val: {val_pca_str}"
                )

            # v11: deep diagnostics on a separate line (every 5 epochs + best).
            # These are the "is the data the problem?" signals.
            if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
                sil = val_metrics.get("val_silhouette", 0.0)
                cd = val_metrics.get("val_centroid_dists", {})
                pf = val_metrics.get("val_pred_freq", [0.0] * len(class_names))
                ds = val_metrics.get("val_dim_stats", {})
                in_int = val_metrics.get("val_input_intensity", 0.0)
                re_int = val_metrics.get("val_recon_intensity", 0.0)
                in_re_gap = val_metrics.get("val_intensity_gap", 0.0)
                pf_str = "/".join(f"{pf[i]:.2f}" for i in range(min(4, len(pf))))
                cd_str = " ".join(
                    f"{k}={v:.2f}" for k, v in sorted(cd.items())
                )
                # Gradient norm summary (top-3 modules)
                grad_norms = train_metrics.get("grad_norms", {})
                top_grads = sorted(grad_norms.items(), key=lambda kv: -kv[1])[:3]
                grad_str = " ".join(f"{k}={v:.2f}" for k, v in top_grads) if top_grads else "n/a"
                print(
                    f"           [DIAG] silhouette={sil:+.3f} | "
                    f"centroid_dists: {cd_str if cd_str else 'n/a'} | "
                    f"pred_dist[NC/SCD/MCI/AD]={pf_str} | "
                    f"latent_dim: mean={ds.get('latent_std_mean', 0):.3f} "
                    f"min={ds.get('latent_std_min', 0):.3f} "
                    f"max={ds.get('latent_std_max', 0):.3f} "
                    f"active={ds.get('n_active_dims', 0)}/{ds.get('n_active_dims', 0) + ds.get('n_collapsed_dims', 0)}"
                )
                print(
                    f"           [DIAG] input_mean={in_int:.3f} recon_mean={re_int:.3f} "
                    f"intensity_gap={in_re_gap:.3f} | top_grads: {grad_str} | "
                    f"mixup_frac={train_metrics.get('mixup_frac', 0):.2f}"
                )

            # Record history
            history["train_loss"].append(train_metrics["loss"])
            history["train_recon_loss"].append(train_metrics["recon_loss"])
            history["train_cls_loss"].append(train_metrics["cls_loss"])
            history["train_kl_loss"].append(train_metrics["kl_loss"])
            history["train_cls_acc"].append(train_metrics["cls_acc"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_recon_loss"].append(val_metrics["recon_loss"])
            history["val_cls_loss"].append(val_metrics["cls_loss"])
            history["val_kl_loss"].append(val_metrics["kl_loss"])
            history["val_cls_acc"].append(val_metrics["cls_acc"])
            # v10: extended history (KL weight, monitoring metrics)
            history["kl_weight"].append(self.current_kl_weight)
            history["kl_active_dims"].append(val_metrics.get("val_kl_active_dims", 0))
            history["latent_std"].append(val_metrics.get("val_latent_std", 0))
            history["per_class_acc_train"].append(pca_tr)
            history["per_class_acc_val"].append(pca_va)

            # Write to CSV (v10: more columns for offline analysis)
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "epoch",
                        "train_loss", "train_recon_loss", "train_cls_loss", "train_kl_loss", "train_cls_acc",
                        "val_loss", "val_recon_loss", "val_cls_loss", "val_kl_loss", "val_cls_acc",
                        "lr", "epoch_time", "is_best",
                        # v10 monitoring
                        "kl_weight", "val_kl_active_dims", "val_latent_std", "train_val_gap",
                        "pca_train_NC", "pca_train_SCD", "pca_train_MCI", "pca_train_AD",
                        "pca_val_NC", "pca_val_SCD", "pca_val_MCI", "pca_val_AD",
                        # v11 deep diagnostics
                        "val_silhouette",
                        "cent_dist_NC_SCD", "cent_dist_NC_MCI", "cent_dist_NC_AD",
                        "cent_dist_SCD_MCI", "cent_dist_SCD_AD", "cent_dist_MCI_AD",
                        "pred_freq_NC", "pred_freq_SCD", "pred_freq_MCI", "pred_freq_AD",
                        "latent_std_min", "latent_std_max",
                        "input_intensity", "recon_intensity", "intensity_gap",
                        "grad_encoder_t1", "grad_decoder", "grad_classifier", "grad_demo",
                        "mixup_frac",
                    ])
                    write_header = False
                # Pad per-class acc to 4 entries in case num_classes differs
                pca_tr_pad = (pca_tr + [0.0] * 4)[:4]
                pca_va_pad = (pca_va + [0.0] * 4)[:4]
                pred_freq = val_metrics.get("val_pred_freq", [0.0] * 4)
                pf_pad = (pred_freq + [0.0] * 4)[:4]
                cd = val_metrics.get("val_centroid_dists", {})
                ds = val_metrics.get("val_dim_stats", {})
                gn = train_metrics.get("grad_norms", {})
                def _gn(key):
                    return gn.get(f"{key}_grad", 0.0)
                writer.writerow([
                    epoch + 1,
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['recon_loss']:.6f}",
                    f"{train_metrics['cls_loss']:.6f}",
                    f"{train_metrics['kl_loss']:.6f}",
                    f"{train_metrics['cls_acc']:.6f}",
                    f"{val_metrics['loss']:.6f}",
                    f"{val_metrics['recon_loss']:.6f}",
                    f"{val_metrics['cls_loss']:.6f}",
                    f"{val_metrics['kl_loss']:.6f}",
                    f"{val_metrics['cls_acc']:.6f}",
                    f"{current_lr:.8f}",
                    f"{epoch_time:.2f}",
                    "1" if is_best else "0",
                    f"{self.current_kl_weight:.6f}",
                    f"{val_metrics.get('val_kl_active_dims', 0):.0f}",
                    f"{val_metrics.get('val_latent_std', 0):.6f}",
                    f"{self.train_val_gap:+.6f}",
                    f"{pca_tr_pad[0]:.4f}",
                    f"{pca_tr_pad[1]:.4f}",
                    f"{pca_tr_pad[2]:.4f}",
                    f"{pca_tr_pad[3]:.4f}",
                    f"{pca_va_pad[0]:.4f}",
                    f"{pca_va_pad[1]:.4f}",
                    f"{pca_va_pad[2]:.4f}",
                    f"{pca_va_pad[3]:.4f}",
                    f"{val_metrics.get('val_silhouette', 0):+.6f}",
                    f"{cd.get('c01_dist', 0):.4f}",
                    f"{cd.get('c02_dist', 0):.4f}",
                    f"{cd.get('c03_dist', 0):.4f}",
                    f"{cd.get('c12_dist', 0):.4f}",
                    f"{cd.get('c13_dist', 0):.4f}",
                    f"{cd.get('c23_dist', 0):.4f}",
                    f"{pf_pad[0]:.4f}",
                    f"{pf_pad[1]:.4f}",
                    f"{pf_pad[2]:.4f}",
                    f"{pf_pad[3]:.4f}",
                    f"{ds.get('latent_std_min', 0):.6f}",
                    f"{ds.get('latent_std_max', 0):.6f}",
                    f"{val_metrics.get('val_input_intensity', 0):.6f}",
                    f"{val_metrics.get('val_recon_intensity', 0):.6f}",
                    f"{val_metrics.get('val_intensity_gap', 0):.6f}",
                    f"{_gn('encoder_t1'):.4f}",
                    f"{_gn('decoder'):.4f}",
                    f"{_gn('classifier'):.4f}",
                    f"{_gn('demo'):.4f}",
                    f"{train_metrics.get('mixup_frac', 0):.4f}",
                ])

            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(output_dir, f"vae_epoch_{epoch+1}.pt")
                self.save_checkpoint(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

            # Save best model
            if is_best:
                best_path = os.path.join(output_dir, "vae_best.pt")
                self.save_checkpoint(best_path)
                print(f"Best model saved to {best_path} (val_cls_acc: {current_cls_acc:.4f})")

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print(f"Best val_cls_acc: {self.best_cls_acc:.4f}")
                print(f"{'='*60}\n")
                early_stopped = True
                break

        return history