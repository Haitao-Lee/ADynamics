"""
Stage 5: Joint Fine-tuning for Multi-Modal VAE.

Fine-tunes all pretrained modules together end-to-end:
    - Multi-Modal VAE Encoder + Decoder (from Stage 1/2)
    - CFM Velocity Field (from Stage 3)
    - Deformation Generator (from Stage 4)

Combined loss:
    L_total = lambda_recon * L_recon + lambda_cfm * L_cfm + lambda_def * L_def

Where:
    L_recon: L1 reconstruction loss
    L_cfm: CFM velocity field loss
    L_def: Deformation smoothness + similarity loss

All modules are unfrozen but use lower learning rates for pretrained weights.

Usage:
    python scripts/train_stage5_joint.py \
        --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \
        --decoder_checkpoint ./checkpoints/stage2_decoder/decoder_best.pt \
        --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt \
        --deform_checkpoint ./checkpoints/stage4_def/def_best.pt \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --output_dir ./checkpoints/stage5_joint
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_train_transforms, get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D
from models.vector_field import VelocityFieldNet
from models.spatial_transform import (
    DeformationGenerator,
    SpatialTransformer,
    compute_jacobian_penalty,
)
from engine.losses import GradientSmoothingLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 5 Joint Fine-tuning")
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage5_joint")
    parser.add_argument("--encoder_checkpoint", type=str,
                        default="./checkpoints/stage1_multimodal/vae_best.pt")
    parser.add_argument("--decoder_checkpoint", type=str, default=None)
    parser.add_argument("--cfm_checkpoint", type=str,
                        default="./checkpoints/stage3_cfm/cfm_best.pt")
    parser.add_argument("--deform_checkpoint", type=str,
                        default="./checkpoints/stage4_def/def_best.pt")

    # Model params (must match Stage 1)
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--cfm_weight", type=float, default=0.1)
    parser.add_argument("--def_weight", type=float, default=0.1)
    parser.add_argument("--smooth_weight", type=float, default=0.05)
    parser.add_argument("--jacobian_weight", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--early_stopping", type=int, default=30)
    parser.add_argument("--no_amp", action="store_true", default=False)
    return parser.parse_args()


def load_data(json_path: str) -> list:
    """Load and validate multi-modal dataset."""
    import nibabel as nib
    from monai.transforms import LoadImaged, EnsureChannelFirstd, Orientationd, Compose

    with open(json_path, "r") as f:
        data = json.load(f)

    quick_transforms = Compose([
        LoadImaged(keys=['t1'], reader='NibabelReader'),
        EnsureChannelFirstd(keys=['t1']),
        Orientationd(keys=['t1'], axcodes='RAS'),
    ])

    valid_data = []
    corrupted = 0
    for item in data:
        t1_path = item.get("t1")
        if not t1_path or not os.path.exists(t1_path):
            continue
        try:
            img = nib.load(t1_path)
            if any(s == 0 for s in img.shape):
                corrupted += 1
                continue
            img_data = img.get_fdata()
            if img_data.min() == img_data.max():
                corrupted += 1
                continue
            quick_transforms({'t1': str(t1_path)})
        except Exception:
            corrupted += 1
            continue
        valid_data.append(item)

    if corrupted > 0:
        print(f"Warning: Skipped {corrupted} corrupted T1 files")
    print(f"Loaded {len(valid_data)} valid samples")
    return valid_data


class JointTrainer:
    """Trainer for Stage 5 joint fine-tuning of all modules."""

    def __init__(
        self,
        vae: nn.Module,
        cfm: nn.Module,
        def_model: nn.Module,
        optimizer: AdamW,
        device: torch.device,
        config: Dict[str, Any],
        scheduler: Optional[CosineAnnealingLR] = None,
        output_dir: str = "./checkpoints/stage5_joint",
    ) -> None:
        self.vae = vae
        self.cfm = cfm
        self.def_model = def_model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scheduler = scheduler
        self.output_dir = output_dir

        self.use_amp = config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None

        self.recon_weight = config.get("recon_weight", 1.0)
        self.cfm_weight = config.get("cfm_weight", 0.1)
        self.def_weight = config.get("def_weight", 0.1)
        self.smooth_weight = config.get("smooth_weight", 0.05)
        self.jacobian_weight = config.get("jacobian_weight", 0.01)

        self.stn = SpatialTransformer(mode="bilinear", padding_mode="border")
        self.smooth_loss_fn = GradientSmoothingLoss(penalty_type="l2")

        vae.to(device)
        cfm.to(device)
        def_model.to(device)

        self.best_val_loss = float("inf")
        self.current_epoch = 0

    def encode(self, x_dict: Dict[str, torch.Tensor]):
        """Encode multi-modal inputs."""
        return self.vae.encode(x_dict)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to image."""
        return self.vae.decode(z)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.vae.train()
        self.cfm.train()
        self.def_model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_cfm = 0.0
        total_def = 0.0
        total_smooth = 0.0
        total_jac = 0.0
        num_batches = 0

        from tqdm import tqdm
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train", leave=False)

        for batch_idx, batch in pbar:
            t1 = batch["t1"].to(self.device)
            x_dict = {"t1": t1}
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    x_dict[mod] = batch[mod].to(self.device)

            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(self.device)
                if labels.dim() > 1:
                    labels = labels.squeeze()

            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                # Full forward pass (encode + decode + classify)
                _, _, mu, logvar = self.vae(x_dict, return_components=True)

                # 1. Reconstruction loss
                z_sample = self.vae.reparameterize(mu, logvar)
                recon = self.decode(z_sample)
                recon_loss = F.l1_loss(recon, t1)

                # 2. CFM loss on NC->AD pairs
                cfm_loss = torch.tensor(0.0, device=self.device)
                if labels is not None:
                    nc_mask = labels == 0
                    ad_mask = labels == 3
                    if nc_mask.sum() > 0 and ad_mask.sum() > 0:
                        z0 = mu[nc_mask]
                        z1 = mu[ad_mask]
                        n_pairs = min(len(z0), len(z1))
                        if n_pairs > 0:
                            t = torch.rand(n_pairs, device=self.device)
                            z_t = (1 - t.view(-1, 1, 1, 1, 1)) * z0[:n_pairs] + t.view(-1, 1, 1, 1, 1) * z1[:n_pairs]
                            v_pred = self.cfm(z_t, t)
                            v_target = z1[:n_pairs] - z0[:n_pairs]
                            cfm_loss = F.mse_loss(v_pred, v_target)

                # 3. Deformation loss (from evolved latent)
                def_loss = torch.tensor(0.0, device=self.device)
                smooth_loss = torch.tensor(0.0, device=self.device)
                jacobian_loss = torch.tensor(0.0, device=self.device)

                if labels is not None and (labels == 0).sum() > 0:
                    z_nc = mu[labels == 0][:1]
                    flow = self.def_model(z_nc)
                    warped = self.stn(t1[labels == 0][:1], flow)
                    # Target: AD image if available
                    if (labels == 3).sum() > 0:
                        ad_image = t1[labels == 3][:1]
                        def_loss = F.l1_loss(warped, ad_image)
                    smooth_loss = self.smooth_loss_fn(flow)
                    jacobian_loss = compute_jacobian_penalty(flow, spacing=(1.0, 1.0, 1.0))

                total_loss_batch = (
                    self.recon_weight * recon_loss
                    + self.cfm_weight * cfm_loss
                    + self.def_weight * def_loss
                    + self.smooth_weight * smooth_loss
                    + self.jacobian_weight * jacobian_loss
                )

            if self.use_amp:
                self.scaler.scale(total_loss_batch).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.cfm.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.def_model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.cfm.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.def_model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += total_loss_batch.item()
            total_recon += recon_loss.item()
            total_cfm += cfm_loss.item()
            total_def += def_loss.item()
            total_smooth += smooth_loss.item()
            total_jac += jacobian_loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{total_loss_batch.item():.4f}"})

        return {
            "loss": total_loss / num_batches,
            "recon": total_recon / num_batches,
            "cfm": total_cfm / num_batches,
            "def": total_def / num_batches,
            "smooth": total_smooth / num_batches,
            "jacobian": total_jac / num_batches,
        }

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.vae.eval()
        self.cfm.eval()
        self.def_model.eval()

        total_loss = 0.0
        total_recon = 0.0
        num_batches = 0

        from tqdm import tqdm
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Val", leave=False)

        for batch_idx, batch in pbar:
            t1 = batch["t1"].to(self.device)
            x_dict = {"t1": t1}
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    x_dict[mod] = batch[mod].to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                _, _, mu, logvar = self.vae(x_dict, return_components=True)
                z_sample = self.vae.reparameterize(mu, logvar)
                recon = self.decode(z_sample)
                recon_loss = F.l1_loss(recon, t1)

            total_loss += recon_loss.item()
            total_recon += recon_loss.item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "recon": total_recon / num_batches,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_interval: int,
        early_stopping_patience: int,
    ) -> None:
        log_file = os.path.join(self.output_dir, "train_log.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        write_header = not os.path.exists(log_file)

        epochs_no_improve = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - t0

            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"LR: {current_lr:.6f} | "
                f"Train: {train_metrics['loss']:.4f} (recon={train_metrics['recon']:.4f}, "
                f"cfm={train_metrics['cfm']:.4f}, def={train_metrics['def']:.4f}) | "
                f"Val: {val_metrics['loss']:.4f} | "
                f"Time: {epoch_time:.1f}s | "
                f"Best: {self.best_val_loss:.4f} (epoch {best_epoch+1}) | "
                f"Patience: {epochs_no_improve}/{early_stopping_patience}"
            )

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "epoch", "train_loss", "train_recon", "train_cfm", "train_def",
                        "val_loss", "val_recon", "lr", "epoch_time", "is_best"
                    ])
                    write_header = False
                writer.writerow([
                    epoch + 1,
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['recon']:.6f}",
                    f"{train_metrics['cfm']:.6f}",
                    f"{train_metrics['def']:.6f}",
                    f"{val_metrics['loss']:.6f}",
                    f"{val_metrics['recon']:.6f}",
                    f"{current_lr:.8f}",
                    f"{epoch_time:.2f}",
                    "1" if is_best else "0",
                ])

            if (epoch + 1) % save_interval == 0:
                torch.save({
                    "epoch": epoch,
                    "vae_state_dict": self.vae.state_dict(),
                    "cfm_state_dict": self.cfm.state_dict(),
                    "def_model_state_dict": self.def_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_loss": self.best_val_loss,
                }, os.path.join(self.output_dir, f"joint_epoch_{epoch+1}.pt"))

            if is_best:
                torch.save({
                    "epoch": epoch,
                    "vae_state_dict": self.vae.state_dict(),
                    "cfm_state_dict": self.cfm.state_dict(),
                    "def_model_state_dict": self.def_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_loss": self.best_val_loss,
                }, os.path.join(self.output_dir, "joint_best.pt"))
                print(f"  -> Best model saved (val_loss: {self.best_val_loss:.4f})")

            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        print(f"\nTraining complete. Best val_loss: {self.best_val_loss:.4f}")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_list = load_data(args.json)
    print(f"Total samples: {len(data_list)}")

    train_transforms = get_multimodal_train_transforms()
    val_transforms = get_multimodal_val_transforms()

    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        data_list, test_size=0.15, stratify=[d.get("label", 0) for d in data_list], random_state=42
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = MultiModalDataset(train_data, transform=train_transforms)
    val_dataset = MultiModalDataset(val_data, transform=val_transforms)

    from core_data.dataset import multimodal_collate_fn
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=multimodal_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=multimodal_collate_fn
    )

    # Load VAE
    vae = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        decoder_depth=args.decoder_depth,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
    )

    def load_ckpt(path, model):
        if not path or not os.path.exists(path):
            print(f"  Warning: checkpoint not found: {path}, using random init")
            return False
        print(f"Loading from {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        model_sd = model.state_dict()
        has_module_prefix = any(k.startswith("module.") for k in sd)
        model_has_module = any(k.startswith("module.") for k in model_sd)
        if has_module_prefix and not model_has_module:
            sd = {k[7:]: v for k, v in sd.items()}
        elif not has_module_prefix and model_has_module:
            sd = {f"module.{k}": v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        return True

    print("Loading VAE...")
    load_ckpt(args.encoder_checkpoint, vae)
    vae = vae.to(device)

    # Load CFM
    print("Loading CFM...")
    cfm = VelocityFieldNet(
        latent_channels=args.latent_channels,
        latent_spatial=(16, 16, 12),
        time_embed_dim=128,
        time_hidden_dim=256,
    ).to(device)
    load_ckpt(args.cfm_checkpoint, cfm)

    # Load Deformation Generator
    print("Loading DeformationGenerator...")
    def_model = DeformationGenerator(
        latent_channels=args.latent_channels,
        latent_spatial=(16, 16, 12),
        output_spatial=MULTI_MODAL_SPATIAL_SIZES["t1"],
        base_channels=16,
    ).to(device)
    load_ckpt(args.deform_checkpoint, def_model)

    print(f"VAE: {sum(p.numel() for p in vae.parameters()) / 1e6:.1f}M params")
    print(f"CFM: {sum(p.numel() for p in cfm.parameters()) / 1e6:.1f}M params")
    print(f"Def: {sum(p.numel() for p in def_model.parameters()) / 1e6:.1f}M params")

    # Optimizer with differential learning rates
    # Fallback: if no trainable params found (e.g. checkpoint missing), use all params
    vae_params = list(vae.parameters())
    cfm_params = list(cfm.parameters())
    def_params = list(def_model.parameters())

    if not vae_params:
        vae_params = list(vae.parameters())
    if not cfm_params:
        cfm_params = list(cfm.parameters())
    if not def_params:
        def_params = list(def_model.parameters())

    optimizer = AdamW([
        {"params": vae_params, "lr": args.learning_rate * 0.1},
        {"params": cfm_params, "lr": args.learning_rate},
        {"params": def_params, "lr": args.learning_rate},
    ], weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)

    config = {
        "recon_weight": args.recon_weight,
        "cfm_weight": args.cfm_weight,
        "def_weight": args.def_weight,
        "smooth_weight": args.smooth_weight,
        "jacobian_weight": args.jacobian_weight,
        "use_amp": not args.no_amp,
    }

    trainer = JointTrainer(
        vae=vae,
        cfm=cfm,
        def_model=def_model,
        optimizer=optimizer,
        device=device,
        config=config,
        scheduler=scheduler,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print("Stage 5: Joint Fine-tuning")
    print(f"{'='*60}")
    print(f"VAE: {args.encoder_checkpoint}")
    print(f"CFM: {args.cfm_checkpoint}")
    print(f"Def: {args.deform_checkpoint}")
    print(f"Epochs: {args.epochs}, LR: {args.learning_rate}")
    print(f"Weights: recon={args.recon_weight}, cfm={args.cfm_weight}, def={args.def_weight}")
    print(f"{'='*60}\n")

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_interval=args.save_interval,
        early_stopping_patience=args.early_stopping,
    )

    print(f"\nStage 5 complete! Best val_loss: {trainer.best_val_loss:.4f}")
    print(f"Best model: {args.output_dir}/joint_best.pt")


if __name__ == "__main__":
    main()
