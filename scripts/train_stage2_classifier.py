"""
Stage 2: Classifier Fine-Tuning for Multi-Modal VAE.

Freeze the pretrained multi-modal encoder, train only the classifier head.
This validates whether the encoder's latent space is discriminative enough.

Key differences from Stage 1:
- Encoder: completely frozen (no gradient updates)
- Classifier head: trainable
- Loss: cross-entropy only
- Learning rate: lower for head-only training

Usage:
    python scripts/train_stage2_classifier.py \
        --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --output_dir ./checkpoints/stage2_classifier
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
        (("output", "dir"), "output_dir"),
        (("seed",), "seed"),
    ]
    config_defaults = apply_yaml_defaults(pre_args.config, mapping) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Stage 2 Classifier Fine-Tuning", parents=[pre])
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage2_classifier")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/stage1_multimodal/vae_best.pt",
                        help="Path to Stage 1 checkpoint (encoder weights)")
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD)")
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of GPUs for DataParallel (default 2; canonical setup is 2x RTX 3090)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--early_stopping", type=int, default=30)
    parser.add_argument("--no_amp", action="store_true", default=False)

    # Modality toggles (must match Stage 1 to load the right encoder architecture).
    # If you trained Stage 1 with --t1_only, you must pass --t1_only here too.
    from utils.stage23_compat import add_modality_args
    add_modality_args(parser)
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


class ClassifierTrainer:
    """Trainer that freezes encoder and trains classifier head only."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        scheduler,
        device: torch.device,
        use_amp: bool,
        num_classes: int,
        output_dir: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.num_classes = num_classes
        self.output_dir = output_dir
        self.scaler = GradScaler() if use_amp else None

        # Freeze entire encoder (all encoder + fusion + logvar layers)
        self._freeze_encoder()
        # Verify classifier is trainable
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

        self.best_acc = 0.0
        self.current_epoch = 0
        self._best_epoch = 0  # Initialize for early stopping tracking
        self.class_total = {c: 0 for c in range(num_classes)}  # For per-class accuracy reporting

    def _freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        frozen = 0
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
                frozen += param.numel()
            else:
                print(f"  Trainable: {name}")
        print(f"Frozen {frozen:,} encoder parameters")

    def train_epoch(self) -> dict:
        self.model.train()

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        from tqdm import tqdm
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Train", leave=False)

        for batch_idx, batch in pbar:
            t1 = batch["t1"].to(self.device)
            x_dict = {"t1": t1}
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    mod_tensor = batch[mod].to(self.device)
                    if mod == "fmri":
                        # Plan C: dataset may return 5D fMRI (preserve_temporal_dim=True).
                        # Normalize to 6D so the model's fMRITemporalEncoder gets a
                        # consistent shape regardless of preserve_temporal_dim.
                        from utils.stage23_compat import normalize_fmri_batch
                        mod_tensor = normalize_fmri_batch(mod_tensor)
                    x_dict[mod] = mod_tensor

            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(self.device)
                if labels.dim() > 1:
                    labels = labels.squeeze()

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=self.use_amp):
                # Forward pass - get only classification logits
                _, cls_logits, mu, logvar = self.model(x_dict, return_components=True)

                # Classification loss only. Guard against None labels —
                # some collate paths may yield unlabeled samples.
                if labels is not None:
                    cls_loss = F.cross_entropy(cls_logits, labels)
                else:
                    cls_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # Per-class accuracy (only meaningful when labels present)
                if labels is not None:
                    preds = cls_logits.argmax(dim=1)
                    acc = (preds == labels).float().mean()
                else:
                    preds = cls_logits.argmax(dim=1)
                    acc = torch.tensor(0.0, device=self.device)

                loss = cls_loss

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
            total_acc += acc.item() if isinstance(acc, torch.Tensor) else acc
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item() if isinstance(acc, torch.Tensor) else acc:.4f}"})

            # OOM fix: release the per-batch autograd graph + activations.
            # recon (50MB) and mu/logvar graphs linger otherwise.
            del loss, cls_logits, mu, logvar
            torch.cuda.empty_cache()

        return {
            "loss": total_loss / num_batches,
            "acc": total_acc / num_batches,
        }

    @torch.no_grad()
    def validate_epoch(self) -> dict:
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        # Per-class accuracy
        class_correct = [0] * self.num_classes
        class_total = [0] * self.num_classes

        from tqdm import tqdm
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Val", leave=False)

        for batch_idx, batch in pbar:
            t1 = batch["t1"].to(self.device)
            x_dict = {"t1": t1}
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    mod_tensor = batch[mod].to(self.device)
                    if mod == "fmri":
                        # Plan C: dataset may return 5D fMRI (preserve_temporal_dim=True).
                        # Normalize to 6D so the model's fMRITemporalEncoder gets a
                        # consistent shape regardless of preserve_temporal_dim.
                        from utils.stage23_compat import normalize_fmri_batch
                        mod_tensor = normalize_fmri_batch(mod_tensor)
                    x_dict[mod] = mod_tensor

            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(self.device)
                if labels.dim() > 1:
                    labels = labels.squeeze()

            with autocast('cuda', enabled=self.use_amp):
                _, cls_logits, _, _ = self.model(x_dict, return_components=True)
                cls_loss = F.cross_entropy(cls_logits, labels)
                preds = cls_logits.argmax(dim=1)
                acc = (preds == labels).float().mean()

                # Per-class
                for c in range(self.num_classes):
                    mask = labels == c
                    class_total[c] += mask.sum().item()
                    class_correct[c] += ((preds == labels) & mask).sum().item()

            total_loss += cls_loss.item()
            total_acc += acc.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{cls_loss.item():.4f}", "acc": f"{acc.item():.4f}"})

        avg_acc = total_acc / num_batches
        per_class_acc = {
            c: class_correct[c] / max(1, class_total[c])
            for c in range(self.num_classes)
        }

        # Cache per-class sample counts so the trainer can print them in
        # train_epoch (avoids the bug where the previous code referenced
        # self.class_total which was never populated and always 0).
        self.class_total = class_total
        self.class_correct = class_correct

        return {
            "loss": total_loss / num_batches,
            "acc": avg_acc,
            "per_class_acc": per_class_acc,
        }

    def save_checkpoint(self, filepath: str) -> None:
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_acc": self.best_acc,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)

    def train(self, num_epochs: int, early_stopping_patience: int) -> None:
        import csv

        log_file = os.path.join(self.output_dir, "train_log.csv")
        os.makedirs(self.output_dir, exist_ok=True)
        write_header = not os.path.exists(log_file)

        CLASS_NAMES_MAP = {
            3: ["NC", "SCD+MCI", "AD"],
            4: ["NC", "SCD", "MCI", "AD"],
        }
        class_names = CLASS_NAMES_MAP.get(self.num_classes, [f"Class_{i}" for i in range(self.num_classes)])

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

            is_best = val_metrics["acc"] > self.best_acc
            if is_best:
                self.best_acc = val_metrics["acc"]
                patience = 0
            else:
                patience = epoch - self._best_epoch

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"LR: {current_lr:.6f} | "
                f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f} | "
                f"Val: loss={val_metrics['loss']:.4f}, acc={val_metrics['acc']:.4f} | "
                f"Time: {epoch_time:.1f}s | "
                f"Best: {self.best_acc:.4f}"
            )

            # Per-class val accuracy
            for c, name in enumerate(class_names):
                pca = val_metrics["per_class_acc"][c]
                print(f"  {name}: {pca:.4f} ({int(val_metrics['per_class_acc'][c] * self.class_total[c])}/{self.class_total[c]})")

            # Write log
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                                   "nc_acc", "scd_acc", "mci_acc", "ad_acc", "lr", "is_best"])
                    write_header = False
                pca = val_metrics["per_class_acc"]
                writer.writerow([
                    epoch + 1,
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['acc']:.6f}",
                    f"{val_metrics['loss']:.6f}",
                    f"{val_metrics['acc']:.6f}",
                    f"{pca.get(0, 0):.6f}",
                    f"{pca.get(1, 0):.6f}",
                    f"{pca.get(2, 0):.6f}",
                    f"{pca.get(3, 0):.6f}",
                    f"{current_lr:.8f}",
                    "1" if is_best else "0",
                ])

            if is_best:
                self._best_epoch = epoch
                self.save_checkpoint(os.path.join(self.output_dir, "classifier_best.pt"))
                print(f"  -> Best model saved (acc: {self.best_acc:.4f})")

            if epoch - self._best_epoch >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best val acc: {self.best_acc:.4f}")
                break

        print(f"\nTraining complete. Best val acc: {self.best_acc:.4f}")


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
        num_workers=0, pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn,
        drop_last=True,  # avoid uneven last batch (replication bug if batch < num_gpus)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn
    )

    # Load model from Stage 1 checkpoint
    from utils.stage23_compat import resolve_optional_modalities, resolve_use_demographic
    optional_modalities = resolve_optional_modalities(args)
    print(f"[Modality switches] optional={optional_modalities}")
    model = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        decoder_depth=args.decoder_depth,
        optional_modalities=optional_modalities,
        use_demographic_cond=resolve_use_demographic(args),
    )

    print(f"Loading encoder from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = checkpoint["model_state_dict"]

    # Handle DataParallel prefix AND filter shape mismatches.
    # Plan C may have introduced modules (e.g. fMRITemporalEncoder inside
    # optional_encoders.fmri) that the downstream stage doesn't build.
    # shape_filtered_load_state_dict silently drops those.
    from utils.stage23_compat import shape_filtered_load_state_dict
    shape_filtered_load_state_dict(model, sd, strict=False, verbose=True)

    model = model.to(device)
    print(f"Model loaded from Stage 1 checkpoint")

    # Multi-GPU: wrap encoder with DataParallel
    from utils.multi_gpu import setup_data_parallel
    import torch as _torch
    print(f"[DEBUG] args.num_gpus = {args.num_gpus}, cuda.device_count = {_torch.cuda.device_count() if _torch.cuda.is_available() else 0}")
    model = setup_data_parallel(model, args.num_gpus)

    # Optimizer: only classifier head
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Honor config: use_amp must come from YAML; --no_amp is a CLI override
    use_amp = getattr(args, "use_amp", False) and not args.no_amp

    # Get class totals for per-class reporting
    val_labels = [d.get("label", 0) for d in val_data]

    trainer = ClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=use_amp,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print("Stage 2: Classifier Fine-Tuning")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.learning_rate}")
    print(f"{'='*60}\n")

    trainer.train(args.epochs, args.early_stopping)

    print(f"\nStage 2 complete!")
    print(f"Best val accuracy: {trainer.best_acc:.4f}")
    if trainer.best_acc < 0.60:
        print("WARNING: Low accuracy - encoder latent may not be discriminative.")
        print("Consider going back to Stage 1 with different settings.")
    elif trainer.best_acc >= 0.75:
        print("SUCCESS: Encoder latent supports good classification!")
        print("Proceed to Stage 3: freeze encoder, train decoder for reconstruction.")


if __name__ == "__main__":
    main()
