"""
Stage 2: Decoder Fine-Tuning for Multi-Modal VAE.

Freeze the pretrained multi-modal encoder from Stage 1, train a stronger
decoder for high-quality T1 reconstruction. This prepares the model for
subsequent CFM training in Stage 3.

Key design:
- Encoder: completely frozen (no gradient updates)
- Decoder: trainable, focuses on reconstruction quality
- Loss: L1/L2 recon + KL (KL gradients don't affect frozen encoder)
- The encoder's latent quality is preserved, decoder learns to decode it better

Usage:
    python scripts/train_stage2_decoder.py \
        --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --output_dir ./checkpoints/stage2_decoder
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

from utils.stage23_compat import (
    normalize_fmri_batch,
    resolve_optional_modalities,
    resolve_use_demographic,
    shape_filtered_load_state_dict,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_train_transforms, get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D


def parse_args():
    from utils.config_loader import apply_yaml_defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    pre_args, _ = pre.parse_known_args()

    mapping = [
        (("input", "encoder_checkpoint"), "checkpoint"),
        (("input", "num_classes"), "num_classes"),
        (("model", "latent_channels"), "latent_channels"),
        (("model", "base_channels"), "base_channels"),
        (("model", "decoder_depth"), "decoder_depth"),
        (("model", "in_channels"), "in_channels"),
        (("model", "spatial_size"), "spatial_size"),
        (("model", "dropout_rate"), "dropout_rate"),
        (("training", "batch_size"), "batch_size"),
        (("training", "learning_rate"), "learning_rate"),
        (("training", "weight_decay"), "weight_decay"),
        (("training", "epochs"), "epochs"),
        (("training", "early_stopping_patience"), "early_stopping"),
        (("training", "num_gpus"), "num_gpus"),
        (("training", "use_amp"), "use_amp"),
        (("loss", "recon_loss_type"), "recon_loss_type"),
        (("loss", "kl_weight"), "kl_weight"),
        (("model", "fmri_t_target"), "fmri_t_target"),
        (("output", "dir"), "output_dir"),
        (("seed",), "seed"),
    ]
    config_defaults = apply_yaml_defaults(pre_args.config, mapping) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Stage 2 Decoder Fine-Tuning", parents=[pre])
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage2_decoder")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/stage1_multimodal/vae_best.pt",
                        help="Path to Stage 1 checkpoint")
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD)")
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--recon_loss_type", type=str, default="l1",
                        help="l1 or l2 reconstruction loss")
    parser.add_argument("--kl_weight", type=float, default=0.0,
                        help="KL weight (encoder frozen, so mainly regularizes latent usage)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of GPUs for DataParallel (default 2; canonical setup is 2x RTX 3090)")
    parser.add_argument("--early_stopping", type=int, default=30)
    parser.add_argument("--no_amp", action="store_true", default=False)
    parser.add_argument("--fmri_t_target", type=int, default=60,
                        help="fMRI temporal target (must match Stage 1)")
    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="Enable AMP (default OFF for canonical 2x RTX 3090 setup; "
                             "YAML use_amp: false maps to --no_amp via set_defaults)")
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


class DecoderTrainer:
    """Trainer that freezes encoder and trains decoder for reconstruction."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        scheduler,
        device: torch.device,
        use_amp: bool,
        recon_loss_type: str,
        kl_weight: float,
        output_dir: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.recon_loss_type = recon_loss_type
        self.kl_weight = kl_weight
        self.output_dir = output_dir
        self.scaler = GradScaler() if use_amp else None

        # Freeze encoder completely
        self._freeze_encoder()

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

        # (Removed dead `decoder_parts` build block — it never appended
        # anything and just printed a hard-coded summary.)

        self.best_recon_loss = float("inf")
        self.current_epoch = 0
        self._best_epoch = 0

    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters, keep decoder trainable."""
        frozen_count = 0
        # Keys that identify non-decoder (encoder-side) parameters.
        # Updated for T1-centric logvar: logvar_base, logvar_delta_nets,
        # logvar_gate_nets replace the old logvar_proj_t1.
        _freeze_keys = ["encoder", "fusion_proj", "logvar_proj", "logvar_base",
                        "logvar_delta", "logvar_gate", "t1_centric_fusion",
                        "classifier", "demo", "age_mlp", "sex_emb"]
        for name, param in self.model.named_parameters():
            if any(k in name for k in _freeze_keys):
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                param.requires_grad = True
        print(f"Frozen {frozen_count:,} encoder parameters")
        # Print what remains trainable
        trainable_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_count += param.numel()
                print(f"  Trainable: {name} ({param.numel():,} params)")
        print(f"Total trainable: {trainable_count:,}")

    def _recon_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.recon_loss_type == "l2":
            return F.mse_loss(recon, target)
        return F.l1_loss(recon, target)

    def train_epoch(self) -> dict:
        self.model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        from tqdm import tqdm
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Train", leave=False)

        for batch_idx, batch in pbar:
            t1 = batch["t1"].to(self.device)
            x_dict = {"t1": t1}

            # Use available_modalities to skip zero-filled missing modalities
            batch_avail = batch.get("available_modalities", None)
            if batch_avail and isinstance(batch_avail[0], (list, tuple)):
                avail_set = set(batch_avail[0])
                for av in batch_avail[1:]:
                    avail_set &= set(av)
            else:
                avail_set = {"fmri", "asl", "qsm", "flair"}

            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in avail_set and mod in batch and batch[mod] is not None:
                    mod_tensor = batch[mod].to(self.device)
                    if mod == "fmri":
                        mod_tensor = normalize_fmri_batch(mod_tensor)
                    x_dict[mod] = mod_tensor

            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(self.device)
                if labels.dim() > 1:
                    labels = labels.squeeze()

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=self.use_amp):
                recon, cls_logits, mu, logvar = self.model(x_dict, return_components=True)

                recon_loss = self._recon_loss(recon, x_dict["t1"])
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                loss = recon_loss + self.kl_weight * kl_loss

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

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
            pbar.set_postfix({"recon": f"{recon_loss.item():.4f}"})

            # OOM fix: release the per-batch autograd graph + activations.
            # recon (50MB at 256^3) and mu/logvar graphs linger otherwise.
            del loss, recon, cls_logits, mu, logvar, recon_loss, kl_loss
            torch.cuda.empty_cache()

        return {
            "loss": total_loss / num_batches,
            "recon_loss": total_recon / num_batches,
            "kl_loss": total_kl / num_batches,
        }

    @torch.no_grad()
    def validate_epoch(self) -> dict:
        self.model.eval()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        num_batches = 0

        from tqdm import tqdm
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Val", leave=False)

        for batch_idx, batch in pbar:
            t1 = batch["t1"].to(self.device)
            x_dict = {"t1": t1}

            # Use available_modalities to skip zero-filled missing modalities
            batch_avail = batch.get("available_modalities", None)
            if batch_avail and isinstance(batch_avail[0], (list, tuple)):
                avail_set = set(batch_avail[0])
                for av in batch_avail[1:]:
                    avail_set &= set(av)
            else:
                avail_set = {"fmri", "asl", "qsm", "flair"}

            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in avail_set and mod in batch and batch[mod] is not None:
                    mod_tensor = batch[mod].to(self.device)
                    if mod == "fmri":
                        mod_tensor = normalize_fmri_batch(mod_tensor)
                    x_dict[mod] = mod_tensor

            with autocast('cuda', enabled=self.use_amp):
                recon, cls_logits, mu, logvar = self.model(x_dict, return_components=True)
                recon_loss = self._recon_loss(recon, x_dict["t1"])
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.kl_weight * kl_loss

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
            pbar.set_postfix({"recon": f"{recon_loss.item():.4f}"})

        return {
            "loss": total_loss / num_batches,
            "recon_loss": total_recon / num_batches,
            "kl_loss": total_kl / num_batches,
        }

    def save_checkpoint(self, filepath: str) -> None:
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_recon_loss": self.best_recon_loss,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)

    def train(self, num_epochs: int, early_stopping_patience: int) -> None:
        import csv

        log_file = os.path.join(self.output_dir, "train_log.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        write_header = not os.path.exists(log_file)

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            import time
            t0 = time.time()

            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - t0

            is_best = val_metrics["recon_loss"] < self.best_recon_loss
            if is_best:
                self.best_recon_loss = val_metrics["recon_loss"]
                self._best_epoch = epoch
                patience = 0
            else:
                patience = epoch - self._best_epoch

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"LR: {current_lr:.6f} | "
                f"Train: loss={train_metrics['loss']:.4f}, recon={train_metrics['recon_loss']:.4f} | "
                f"Val: loss={val_metrics['loss']:.4f}, recon={val_metrics['recon_loss']:.4f} | "
                f"Time: {epoch_time:.1f}s | "
                f"Patience: {patience}/{early_stopping_patience}"
            )

            # Write log
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["epoch", "train_loss", "train_recon", "train_kl",
                                   "val_loss", "val_recon", "val_kl", "lr", "epoch_time", "is_best"])
                    write_header = False
                writer.writerow([
                    epoch + 1,
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['recon_loss']:.6f}",
                    f"{train_metrics['kl_loss']:.6f}",
                    f"{val_metrics['loss']:.6f}",
                    f"{val_metrics['recon_loss']:.6f}",
                    f"{val_metrics['kl_loss']:.6f}",
                    f"{current_lr:.8f}",
                    f"{epoch_time:.2f}",
                    "1" if is_best else "0",
                ])

            if is_best:
                self.save_checkpoint(os.path.join(self.output_dir, "decoder_best.pt"))
                print(f"  -> Best model saved (recon: {self.best_recon_loss:.4f})")

            if epoch - self._best_epoch >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best recon loss: {self.best_recon_loss:.4f}")
                break

        print(f"\nTraining complete. Best recon loss: {self.best_recon_loss:.4f}")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_list = load_data(args.json)
    print(f"Total samples: {len(data_list)}")

    # Store manifest index for precomputed cache lookup
    for i, item in enumerate(data_list):
        item["_manifest_idx"] = i

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

    # Precomputed cache (chunked)
    _npy_cache = None
    for _cache_candidate in ["./npy_cache", "C:/ADynamics_npy_cache"]:
        if os.path.isdir(_cache_candidate):
            _npy_cache = _cache_candidate
            break
    _precomputed_path = None
    if _npy_cache:
        _chunked = os.path.join(_npy_cache, "precomputed")
        if os.path.isdir(_chunked) and os.path.exists(os.path.join(_chunked, "index.json")):
            _precomputed_path = _chunked

    fmri_t_target = int(getattr(args, "fmri_t_target", 60))
    train_dataset = MultiModalDataset(train_data, transform=train_transforms,
                                      precomputed_path=_precomputed_path,
                                      fmri_t_target=fmri_t_target)
    val_dataset = MultiModalDataset(val_data, transform=val_transforms,
                                    precomputed_path=_precomputed_path,
                                    fmri_t_target=fmri_t_target)

    from core_data.dataset import multimodal_collate_fn
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn,
        drop_last=True,
    )

    # Load model from Stage 1 checkpoint
    fmri_t_target = int(getattr(args, "fmri_t_target", 60))
    model = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        decoder_depth=args.decoder_depth,
        optional_modalities=resolve_optional_modalities(args),
        use_fmri_deep=True,
        use_demographic_cond=resolve_use_demographic(args),
        fmri_t_target=fmri_t_target,
    )

    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = checkpoint["model_state_dict"]

    shape_filtered_load_state_dict(model, sd, strict=False, verbose=True)
    model = model.to(device)
    print("Model loaded successfully")

    # Multi-GPU: wrap encoder with DataParallel
    from utils.multi_gpu import setup_data_parallel
    model = setup_data_parallel(model, args.num_gpus)

    # Optimizer: only decoder (and fusion/logvar for latent manipulation)
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)

    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Honor YAML use_amp AND CLI --no_amp. If YAML says use_amp: false
    # (canonical 2x RTX 3090), set_defaults pre-fills args.use_amp=False;
    # CLI --no_amp would also be respected. Only turn AMP on if BOTH say so.
    use_amp = getattr(args, "use_amp", False) and not args.no_amp

    trainer = DecoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=use_amp,
        recon_loss_type=args.recon_loss_type,
        kl_weight=args.kl_weight,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print("Stage 2: Decoder Fine-Tuning")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"Recon loss type: {args.recon_loss_type}")
    print(f"KL weight: {args.kl_weight}")
    print(f"LR: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    trainer.train(args.epochs, args.early_stopping)

    print(f"\nStage 2 complete!")
    print(f"Best recon loss: {trainer.best_recon_loss:.4f}")
    print(f"\nNext: Stage 3 - Train CFM on the frozen encoder + decoder latent space")


if __name__ == "__main__":
    main()
