"""
Stage 4: Deformation Generator Training for Multi-Modal VAE.

Trains the DeformationGenerator to produce anatomically plausible deformation
fields from the latent space of the multi-modal VAE.

The DeformationGenerator takes NC latent representations and outputs a 3D
displacement field that warps NC MRI toward AD-like patterns.

Loss components:
    1. Image similarity loss (L1 between warped NC and target AD)
    2. Smoothness loss (gradient-based regularization)
    3. Jacobian penalty (diffeomorphism constraint)

Usage:
    python scripts/train_stage4_deformation.py \
        --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \
        --decoder_checkpoint ./checkpoints/stage2_decoder/decoder_best.pt \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --output_dir ./checkpoints/stage4_def
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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

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
    from utils.config_loader import apply_yaml_defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    pre_args, _ = pre.parse_known_args()

    mapping = [
        (("input", "encoder_checkpoint"), "encoder_checkpoint"),
        (("input", "cfm_checkpoint"), "cfm_checkpoint"),
        (("input", "num_classes"), "num_classes"),
        (("model", "latent_channels"), "latent_channels"),
        (("model", "base_channels"), "base_channels"),
        (("model", "decoder_depth"), "decoder_depth"),
        (("model", "latent_spatial"), "latent_spatial"),
        (("model", "output_spatial"), "output_spatial"),
        (("model", "channel_mults"), "channel_mults"),
        (("model", "num_res_blocks"), "num_res_blocks"),
        (("training", "batch_size"), "batch_size"),
        (("training", "learning_rate"), "learning_rate"),
        (("training", "weight_decay"), "weight_decay"),
        (("training", "epochs"), "epochs"),
        (("training", "early_stopping_patience"), "early_stopping"),
        (("training", "num_gpus"), "num_gpus"),
        (("training", "use_amp"), "no_amp"),
        (("loss", "sim_weight"), "sim_weight"),
        (("loss", "smooth_weight"), "smooth_weight"),
        (("loss", "jacobian_weight"), "jacobian_weight"),
        (("output", "dir"), "output_dir"),
        (("seed",), "seed"),
    ]
    config_defaults = apply_yaml_defaults(pre_args.config, mapping) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Stage 4 Deformation Generator", parents=[pre])
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage4_def")
    parser.add_argument("--encoder_checkpoint", type=str,
                        default="./checkpoints/stage1_multimodal/vae_best.pt",
                        help="Path to Stage 1 checkpoint")
    parser.add_argument("--decoder_checkpoint", type=str,
                        default="./checkpoints/stage2_decoder/decoder_best.pt",
                        help="Optional: Path to Stage 2 decoder checkpoint")
    parser.add_argument("--cfm_checkpoint", type=str, default=None,
                        help="Optional: Path to Stage 3 CFM checkpoint")
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD)")
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--sim_weight", type=float, default=1.0)
    parser.add_argument("--smooth_weight", type=float, default=0.1)
    parser.add_argument("--jacobian_weight", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--early_stopping", type=int, default=50)
    parser.add_argument("--no_amp", action="store_true", default=False)
    # Apply YAML config defaults AFTER all add_argument calls
    # (set_defaults must come last so it isn't overridden by argparse defaults)
    parser.set_defaults(**config_defaults)
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


class DeformationTrainer:
    """Trainer for Stage 4 Deformation Generator."""

    def __init__(
        self,
        def_model: nn.Module,
        encoder: nn.Module,
        decoder: Optional[nn.Module],
        cfm_model: Optional[nn.Module],
        optimizer: AdamW,
        device: torch.device,
        config: Dict[str, Any],
        scheduler: Optional[CosineAnnealingLR] = None,
    ) -> None:
        self.def_model = def_model
        self.encoder = encoder
        self.decoder = decoder
        self.cfm_model = cfm_model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.scheduler = scheduler

        self.use_amp = config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None

        self.sim_weight = config.get("sim_weight", 1.0)
        self.smooth_weight = config.get("smooth_weight", 0.1)
        self.jacobian_weight = config.get("jacobian_weight", 0.01)

        self.stn = SpatialTransformer(mode="bilinear", padding_mode="border")
        self.smooth_loss_fn = GradientSmoothingLoss(penalty_type="l2")

        def_model.to(device)
        encoder.to(device)
        if decoder:
            decoder.to(device)
        if cfm_model:
            cfm_model.to(device)

        self.best_val_loss = float("inf")
        self.current_epoch = 0

    def encode(self, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode with encoder, return mu."""
        _, _, mu, _ = self.encoder(x_dict, return_components=True)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode with decoder."""
        if self.decoder is not None:
            return self.decoder(z)
        # If no separate decoder, use encoder's decode method
        return self.encoder.decode(z)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.def_model.train()
        self.encoder.eval()
        if self.decoder:
            self.decoder.eval()
        if self.cfm_model:
            self.cfm_model.eval()

        total_loss = 0.0
        total_sim = 0.0
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

            labels = batch.get("label", torch.tensor([])).to(self.device)
            if labels.dim() > 1:
                labels = labels.squeeze()

            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                # Encode source images
                z_source = self.encode(x_dict)

                # Optionally evolve with CFM
                z_input = z_source
                if self.cfm_model is not None:
                    batch_size = z_source.shape[0]
                    steps = self.config.get("cfm_steps", 10)
                    dt = 1.0 / steps
                    z_t = z_source
                    with torch.no_grad():
                        for i in range(steps):
                            t = torch.full((batch_size,), i * dt, device=self.device)
                            v = self.cfm_model(z_t, t, c=None)
                            z_t = z_t + v * dt
                    z_input = z_t

                # Generate deformation field
                flow = self.def_model(z_input)

                # Apply deformation to original MRI using Spatial Transformer
                # This warps NC-like patterns toward AD-like patterns
                warped_image = self.stn(t1, flow)

                # Decode the evolved latent to get reconstruction for auxiliary loss
                recon = self.decode(z_input)

                # Compute losses
                # Primary: similarity between warped and target (AD if available, else original)
                # When NC->AD progression, target is AD image from same subject or paired sample
                ad_label = args.num_classes - 1  # AD is the last class
                if (labels == ad_label).sum() > 0:  # AD subjects available
                    ad_idx = (labels == ad_label).nonzero(as_tuple=True)[0][0]
                    ad_target = t1[(labels == ad_label).nonzero(as_tuple=True)[0]]
                    sim_loss = F.l1_loss(warped_image, ad_target)
                else:
                    # Fallback: compare warped to original (deformation should be small)
                    sim_loss = F.l1_loss(warped_image, t1)
                smooth_loss = self.smooth_loss_fn(flow)
                jacobian_loss = compute_jacobian_penalty(flow, spacing=(1.0, 1.0, 1.0))

                loss = (
                    self.sim_weight * sim_loss
                    + self.smooth_weight * smooth_loss
                    + self.jacobian_weight * jacobian_loss
                )

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.def_model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.def_model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            total_sim += sim_loss.item()
            total_smooth += smooth_loss.item()
            total_jac += jacobian_loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return {
            "loss": total_loss / num_batches,
            "sim_loss": total_sim / num_batches,
            "smooth_loss": total_smooth / num_batches,
            "jacobian_loss": total_jac / num_batches,
        }

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.def_model.eval()

        total_loss = 0.0
        total_sim = 0.0
        total_smooth = 0.0
        total_jac = 0.0
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
                z = self.encode(x_dict)
                flow = self.def_model(z)

                # Apply deformation to original MRI
                warped_image = self.stn(t1, flow)
                recon = self.decode(z)

                # Similarity: warped image vs original (deformation should be anatomically plausible)
                sim_loss = F.l1_loss(warped_image, t1)
                smooth_loss = self.smooth_loss_fn(flow)
                jacobian_loss = compute_jacobian_penalty(flow, spacing=(1.0, 1.0, 1.0))

                loss = (
                    self.sim_weight * sim_loss
                    + self.smooth_weight * smooth_loss
                    + self.jacobian_weight * jacobian_loss
                )

            total_loss += loss.item()
            total_sim += sim_loss.item()
            total_smooth += smooth_loss.item()
            total_jac += jacobian_loss.item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "sim_loss": total_sim / num_batches,
            "smooth_loss": total_smooth / num_batches,
            "jacobian_loss": total_jac / num_batches,
        }

    def save_checkpoint(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "epoch": self.current_epoch,
            "def_model_state_dict": self.def_model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, filepath)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        output_dir: str,
        save_interval: int,
        early_stopping_patience: int,
    ) -> None:
        log_file = os.path.join(output_dir, "train_log.csv")
        os.makedirs(output_dir, exist_ok=True)
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
                f"Train: {train_metrics['loss']:.4f} (sim={train_metrics['sim_loss']:.4f}, "
                f"smooth={train_metrics['smooth_loss']:.4f}, jac={train_metrics['jacobian_loss']:.4f}) | "
                f"Val: {val_metrics['loss']:.4f} (sim={val_metrics['sim_loss']:.4f}) | "
                f"Time: {epoch_time:.1f}s | "
                f"Best: {self.best_val_loss:.4f} (epoch {best_epoch+1}) | "
                f"Patience: {epochs_no_improve}/{early_stopping_patience}"
            )

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "epoch", "train_loss", "train_sim", "train_smooth", "train_jac",
                        "val_loss", "val_sim", "val_smooth", "val_jac", "lr", "epoch_time", "is_best"
                    ])
                    write_header = False
                writer.writerow([
                    epoch + 1,
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['sim_loss']:.6f}",
                    f"{train_metrics['smooth_loss']:.6f}",
                    f"{train_metrics['jacobian_loss']:.6f}",
                    f"{val_metrics['loss']:.6f}",
                    f"{val_metrics['sim_loss']:.6f}",
                    f"{val_metrics['smooth_loss']:.6f}",
                    f"{val_metrics['jacobian_loss']:.6f}",
                    f"{current_lr:.8f}",
                    f"{epoch_time:.2f}",
                    "1" if is_best else "0",
                ])

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(os.path.join(output_dir, f"def_epoch_{epoch+1}.pt"))

            if is_best:
                self.save_checkpoint(os.path.join(output_dir, "def_best.pt"))
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

    # Remap 4-class labels to 3-class (SCD+MCI merged) if needed
    if args.num_classes == 3:
        from utils.config_loader import remap_labels_3class
        remap_labels_3class(data_list)
        print("Remapped labels to 3-class (NC / SCD+MCI / AD)")

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

    # Load encoder from Stage 1
    encoder = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        decoder_depth=args.decoder_depth,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
    )

    print(f"Loading encoder from {args.encoder_checkpoint}")
    ckpt = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]

    encoder_sd = encoder.state_dict()
    has_module_prefix = any(k.startswith("module.") for k in sd)
    model_has_module = any(k.startswith("module.") for k in encoder_sd)

    if has_module_prefix and not model_has_module:
        sd = {k[7:]: v for k, v in sd.items()}
    elif not has_module_prefix and model_has_module:
        sd = {f"module.{k}": v for k, v in sd.items()}

    encoder.load_state_dict(sd, strict=False)
    encoder = encoder.to(device)

    # Multi-GPU: wrap encoder with DataParallel
    from utils.multi_gpu import setup_data_parallel
    encoder = setup_data_parallel(encoder, args.num_gpus)

    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False
    print("Encoder loaded and frozen")

    # Freeze decoder too
    decoder = None

    # Load CFM if provided
    cfm_model = None
    if args.cfm_checkpoint and os.path.exists(args.cfm_checkpoint):
        print(f"Loading CFM from {args.cfm_checkpoint}")
        cfm_model = VelocityFieldNet(
            latent_channels=args.latent_channels,
            latent_spatial=(16, 16, 12),
            time_embed_dim=128,
            time_hidden_dim=256,
        ).to(device)
        cfm_ckpt = torch.load(args.cfm_checkpoint, map_location=device, weights_only=False)
        cfm_sd = cfm_ckpt.get("model_state_dict", cfm_ckpt)
        if any(k.startswith("module.") for k in cfm_sd):
            cfm_sd = {k[7:]: v for k, v in cfm_sd.items()}
        cfm_model.load_state_dict(cfm_sd, strict=False)
        cfm_model.eval()
        for param in cfm_model.parameters():
            param.requires_grad = False
        print("CFM loaded and frozen")

    # Create deformation generator
    latent_spatial = (16, 16, 12)
    def_model = DeformationGenerator(
        latent_channels=args.latent_channels,
        latent_spatial=latent_spatial,
        output_spatial=MULTI_MODAL_SPATIAL_SIZES["t1"],
        base_channels=16,
    ).to(device)

    print(f"DeformationGenerator params: {sum(p.numel() for p in def_model.parameters()) / 1e6:.1f}M")

    optimizer = AdamW(def_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    config = {
        "sim_weight": args.sim_weight,
        "smooth_weight": args.smooth_weight,
        "jacobian_weight": args.jacobian_weight,
        "cfm_steps": 10,
        "use_amp": not args.no_amp,
    }

    trainer = DeformationTrainer(
        def_model=def_model,
        encoder=encoder,
        decoder=decoder,
        cfm_model=cfm_model,
        optimizer=optimizer,
        device=device,
        config=config,
        scheduler=scheduler,
    )

    print(f"\n{'='*60}")
    print("Stage 4: Deformation Generator Training")
    print(f"{'='*60}")
    print(f"Encoder: {args.encoder_checkpoint}")
    print(f"CFM: {args.cfm_checkpoint or 'None'}")
    print(f"Latent shape: [{args.latent_channels}, 16, 16, 12]")
    print(f"Output spatial: {MULTI_MODAL_SPATIAL_SIZES['t1']}")
    print(f"Epochs: {args.epochs}, LR: {args.learning_rate}")
    print(f"{'='*60}\n")

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        early_stopping_patience=args.early_stopping,
    )

    print(f"\nStage 4 complete! Best val_loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
