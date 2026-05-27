"""
Stage 1 Multi-Modal VAE Training for ADynamics.

Trains a multi-modal VAE that encodes T1 (required) + optional modalities (fMRI, ASL, QSM, FLAIR)
into a unified latent space, then decodes back to T1 reconstruction while training
a disease classifier.

Usage:
    python scripts/train_stage1_multimodal.py \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --batch_size 2 \
        --epochs 300 \
        --learning_rate 0.0002 \
        --output_dir ./checkpoints/stage1_multimodal
"""

# Must be at very top - before any other imports
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DataParallel

# Additional warning suppression for main process
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*allow_smaller.*")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_train_transforms, get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from engine.trainer_vae import MultiModalVAETrainer
from models.vae3d import MultiModalVAE3D


class MultiModalDataParallel(DataParallel):
    """DataParallel wrapper that properly handles dict inputs for MultiModalVAE3D."""

    def forward(self, x_dict, **kwargs):
        """Scatter dict inputs across GPUs and gather outputs."""
        if not self.device_ids or len(self.device_ids) == 1:
            return self.module(x_dict, **kwargs)

        batch_size = next(iter(x_dict.values())).shape[0]
        num_gpus = len(self.device_ids)

        # Handle case where batch_size < num_gpus by replicating
        if batch_size < num_gpus:
            replicas = self.replicate(self.module, self.device_ids[:batch_size])
            outputs = []
            for i in range(batch_size):
                device = self.device_ids[i]
                sub_dict = {k: v.to(device) for k, v in x_dict.items()}
                outputs.append(replicas[i](sub_dict, **kwargs))
            gathered = []
            for component_tuple in zip(*outputs):
                gathered.append(self.gather(component_tuple, self.output_device))
            return tuple(gathered)

        # Normal case: batch_size >= num_gpus
        replicas = self.replicate(self.module, self.device_ids)
        chunk_size = batch_size // num_gpus
        remainder = batch_size % num_gpus
        outputs = []
        start = 0
        for i, replica in enumerate(replicas):
            end = start + chunk_size + (1 if i < remainder else 0)
            sub_dict = {k: v[start:end].to(self.device_ids[i]) for k, v in x_dict.items()}
            outputs.append(replica(sub_dict, **kwargs))
            start = end

        gathered = []
        for component_tuple in zip(*outputs):
            gathered.append(self.gather(component_tuple, self.output_device))
        return tuple(gathered)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Multi-Modal VAE Training")

    # Data
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json",
                        help="Path to dataset JSON manifest")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage1_multimodal",
                        help="Output directory for checkpoints")

    # Model
    parser.add_argument("--latent_channels", type=int, default=32,
                        help="Latent channels per modality encoder")
    parser.add_argument("--base_channels", type=int, default=16,
                        help="Base channel count for encoder")
    parser.add_argument("--decoder_depth", type=int, default=4,
                        help="Decoder depth (4 for full upsampling)")
    parser.add_argument("--dropout_rate", type=float, default=0.2,
                        help="Modality dropout rate during training")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of disease classes (4: NC, SCD, MCI, AD)")

    # Training
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--cls_weight", type=float, default=1.0,
                        help="Classification loss weight")
    parser.add_argument("--kl_weight", type=float, default=0.1,
                        help="KL divergence loss weight")
    parser.add_argument("--recon_loss_type", type=str, default="l1",
                        help="Reconstruction loss type (l1 or l2)")

    # Hardware
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Checkpoint save interval")
    parser.add_argument("--early_stopping", type=int, default=100,
                        help="Early stopping patience")

    # AMP
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--no_amp", action="store_true", default=False,
                        help="Disable AMP")

    return parser.parse_args()


def load_data(json_path: str) -> list:
    """Load and validate multi-modal dataset."""
    import nibabel as nib
    from monai.transforms import LoadImaged, EnsureChannelFirstd, Orientationd, CropForegroundd, Spacingd, ScaleIntensityRangePercentilesd, ResizeWithPadOrCropd, Compose

    with open(json_path, "r") as f:
        data = json.load(f)

    # Quick validation transforms (skip expensive CropForegroundd/Spacingd for speed)
    quick_transforms = Compose([
        LoadImaged(keys=['t1'], reader='NibabelReader'),
        EnsureChannelFirstd(keys=['t1']),
        Orientationd(keys=['t1'], axcodes='RAS'),
    ])

    valid_data = []
    corrupted_t1 = 0
    for item in data:
        # T1 is required
        t1_path = item.get("t1")
        if not t1_path or not os.path.exists(t1_path):
            continue

        # Validate T1 file dimensions and data content (catch corrupted [0,0,0] files and all-zero files)
        try:
            img = nib.load(t1_path)
            shape = img.shape
            if any(s == 0 for s in shape):
                corrupted_t1 += 1
                continue
            # Check data is not all zeros (CropForegroundd would produce empty output)
            img_data = img.get_fdata()
            if img_data.min() == img_data.max():  # All zeros or constant value
                corrupted_t1 += 1
                continue
            # Quick transform check (loadable by MONAI)
            data_dict = {'t1': str(t1_path)}
            quick_transforms(data_dict)
        except Exception:
            corrupted_t1 += 1
            continue

        # Check at least one optional modality exists
        has_optional = False
        for mod in ["fmri", "asl", "qsm", "flair"]:
            path = item.get(mod)
            if path and os.path.exists(path):
                has_optional = True
                break

        # For now, accept samples with T1 even if no optional modalities
        valid_data.append(item)

    if corrupted_t1 > 0:
        print(f"Warning: Skipped {corrupted_t1} corrupted T1 files")
    print(f"Loaded {len(valid_data)} valid multi-modal samples")
    return valid_data


def main():
    args = parse_args()

    # Handle AMP flag
    use_amp = args.use_amp and not args.no_amp

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_list = load_data(args.json)
    print(f"Total samples: {len(data_list)}")

    # Transforms
    train_transforms = get_multimodal_train_transforms()
    val_transforms = get_multimodal_val_transforms()

    # Split data
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        data_list, test_size=0.15, stratify=[d.get("label", 0) for d in data_list], random_state=42
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Datasets
    train_dataset = MultiModalDataset(train_data, transform=train_transforms)
    val_dataset = MultiModalDataset(val_data, transform=val_transforms)

    # Dataloaders
    from core_data.dataset import multimodal_collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn
    )

    # Model
    model = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        decoder_depth=args.decoder_depth,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
    )

    # Multi-GPU support with custom DataParallel for dict inputs
    if args.num_gpus > 1:
        model = MultiModalDataParallel(model, device_ids=list(range(args.num_gpus)))

    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Config
    config = {
        "cls_weight": args.cls_weight,
        "kl_weight": args.kl_weight,
        "recon_loss_type": args.recon_loss_type,
        "use_amp": use_amp,
    }

    # Trainer
    trainer = MultiModalVAETrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        scheduler=scheduler,
    )

    # Resume from checkpoint
    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Train
    print(f"\n{'='*60}")
    print("Starting Multi-Modal VAE Training")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Latent channels: {args.latent_channels}")
    print(f"Base channels: {args.base_channels}")
    print(f"Classification weight: {args.cls_weight}")
    print(f"Modality dropout rate: {args.dropout_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    history = trainer.train(
        num_epochs=args.epochs,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
        early_stopping_patience=args.early_stopping,
    )

    print("\nTraining complete!")
    print(f"Best val_cls_acc: {trainer.best_cls_acc:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
