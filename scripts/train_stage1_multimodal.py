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

# Additional warning suppression for main process
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*allow_smaller.*")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_train_transforms, get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from engine.trainer_vae import MultiModalVAETrainer
from models.vae3d import MultiModalVAE3D
from utils.multi_gpu import setup_data_parallel


def _load_yaml_defaults(config_path: str) -> dict:
    """Load YAML config and map nested keys to argparse argument names."""
    from utils.config_loader import apply_yaml_defaults
    mapping = [
        (("data", "json"), "json"),
        (("data", "num_classes"), "num_classes"),
        (("model", "latent_channels"), "latent_channels"),
        (("model", "base_channels"), "base_channels"),
        (("model", "decoder_depth"), "decoder_depth"),
        (("model", "dropout_rate"), "dropout_rate"),
        (("model", "use_attention"), "use_attention"),
        (("model", "attention_heads"), "attention_heads"),
        (("training", "batch_size"), "batch_size"),
        (("training", "learning_rate"), "learning_rate"),
        (("training", "weight_decay"), "weight_decay"),
        (("training", "epochs"), "epochs"),
        (("training", "early_stopping_patience"), "early_stopping"),
        (("training", "save_interval"), "save_interval"),
        (("training", "num_gpus"), "num_gpus"),
        (("training", "use_amp"), "use_amp"),
        (("loss", "recon_loss_type"), "recon_loss_type"),
        (("loss", "cls_weight"), "cls_weight"),
        (("loss", "kl_weight"), "kl_weight"),
        (("loss", "kl_warmup_epochs"), "kl_warmup_epochs"),
        (("loss", "free_bits"), "free_bits"),
        (("loss", "contrastive_weight"), "contrastive_weight"),
        (("loss", "gradient_weight"), "gradient_weight"),
        (("loss", "ssim_weight"), "ssim_weight"),
        (("output", "dir"), "output_dir"),
        (("seed",), "seed"),
    ]
    return apply_yaml_defaults(config_path, mapping)


def parse_args():
    # Pre-parse config file (if provided) to set defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    pre_args, _ = pre.parse_known_args()

    config_defaults = {}
    if pre_args.config and os.path.exists(pre_args.config):
        config_defaults = _load_yaml_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description="Stage 1 Multi-Modal VAE Training", parents=[pre])

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

    # Encoder: multi-axis 3D attention (from NeuroQuant, CVPR Findings 2026)
    parser.add_argument("--use_attention", action="store_true", default=True,
                        help="Insert multi-axis 3D attention into encoder (default ON)")
    parser.add_argument("--no_attention", action="store_true", default=False,
                        help="Disable multi-axis 3D attention (revert to plain ResNet)")
    parser.add_argument("--attention_levels", type=str, default="3",
                        help="Comma-separated 0-indexed stage numbers for attention, e.g. '3' or '2,3'")
    parser.add_argument("--attention_heads", type=int, default=8,
                        help="Number of heads per axial attention block (auto-reduced if it doesn't divide channels)")

    # Training
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--cls_weight", type=float, default=2.0,
                        help="Classification loss weight (higher = more discriminative latent)")
    parser.add_argument("--kl_weight", type=float, default=0.1,
                        help="KL divergence loss weight")
    parser.add_argument("--kl_warmup_epochs", type=int, default=20,
                        help="Epochs for KL weight annealing from 0 to kl_weight")
    parser.add_argument("--free_bits", type=float, default=0.0,
                        help="Free bits per latent dimension (minimum KL, prevents collapse)")
    parser.add_argument("--recon_loss_type", type=str, default="l1",
                        help="Reconstruction loss type (l1 or l2)")
    parser.add_argument("--contrastive_weight", type=float, default=0.0,
                        help="Ordinal contrastive loss weight (0=disabled, try 0.05)")
    parser.add_argument("--gradient_weight", type=float, default=0.0,
                        help="Gradient/texture loss weight (0=disabled, try 0.1)")
    parser.add_argument("--ssim_weight", type=float, default=0.0,
                        help="SSIM loss weight (0=disabled, try 0.1)")

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

    # Apply YAML config defaults AFTER all add_argument calls
    # (set_defaults must come last so it isn't overridden by argparse defaults)
    parser.set_defaults(**config_defaults)

    return parser.parse_args()


def load_data(json_path: str, num_classes: int = 4) -> list:
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

    # Conditionally remap labels: only when num_classes=3 (merge SCD+MCI)
    # When num_classes=4, keep all 4 stages: NC=0, SCD=1, MCI=2, AD=3
    if num_classes == 3:
        for item in valid_data:
            label = item.get("label", 0)
            if label in [1, 2]:  # SCD or MCI -> merged class
                item["label"] = 1
            elif label == 3:  # AD -> class 2
                item["label"] = 2
            # NC (0) stays 0
        print("Remapped labels to 3-class (NC / SCD+MCI / AD)")
    else:
        print(f"Keeping labels as {num_classes}-class (NC / SCD / MCI / AD)")

    if corrupted_t1 > 0:
        print(f"Warning: Skipped {corrupted_t1} corrupted T1 files")
    print(f"Loaded {len(valid_data)} valid multi-modal samples")

    # Print class distribution
    from collections import Counter
    label_counts = Counter(item.get("label", 0) for item in valid_data)
    if num_classes == 3:
        class_names = ["NC", "SCD+MCI", "AD"]
    else:
        class_names = ["NC", "SCD", "MCI", "AD"]
    for c in range(num_classes):
        print(f"  {class_names[c]}: {label_counts.get(c, 0)}")

    return valid_data


def main():
    args = parse_args()

    # Handle AMP flag
    use_amp = args.use_amp and not args.no_amp

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_list = load_data(args.json, num_classes=args.num_classes)
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
        collate_fn=multimodal_collate_fn,
        drop_last=True,  # avoid uneven last batch (replication bug if batch < num_gpus)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn,
        drop_last=True,
    )

    # Model
    # Parse attention_levels: accept either CLI comma-separated string ("2,3")
    # or YAML list ([2, 3]) or single int / str.
    use_attention = args.use_attention and not args.no_attention
    al_raw = args.attention_levels
    if isinstance(al_raw, (list, tuple)):
        attn_levels = tuple(int(x) for x in al_raw)
    else:
        try:
            attn_levels = tuple(int(x.strip()) for x in str(al_raw).split(",") if str(x).strip())
        except (ValueError, AttributeError):
            raise ValueError(f"--attention_levels must be comma-separated ints, got {al_raw!r}")
    print(f"[Encoder] use_attention={use_attention}  attention_levels={attn_levels}  attention_heads={args.attention_heads}")
    model = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        decoder_depth=args.decoder_depth,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
        use_attention=use_attention,
        attention_levels=attn_levels,
        attention_heads=args.attention_heads,
    )

    # Multi-GPU support via shared utils (replaces buggy local DataParallel)
    print(f"[DEBUG] args.num_gpus = {args.num_gpus}, cuda.device_count = {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    model = setup_data_parallel(model, args.num_gpus)

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
        "kl_warmup_epochs": args.kl_warmup_epochs,
        "free_bits": args.free_bits,
        "recon_loss_type": args.recon_loss_type,
        "contrastive_weight": args.contrastive_weight,
        "gradient_weight": args.gradient_weight,
        "ssim_weight": args.ssim_weight,
        "num_classes": args.num_classes,
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
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = checkpoint["model_state_dict"]

        # Handle DataParallel prefix
        model_ref = model.module if hasattr(model, "module") else model
        model_sd = model_ref.state_dict()
        has_dp = any(k.startswith("module.") for k in sd)

        if has_dp:
            sd = {k[7:]: v for k, v in sd.items()}

        # Filter: only load keys that exist and shape matches
        filtered_sd = {}
        skipped = []
        for k, v in sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                filtered_sd[k] = v
            else:
                skipped.append(k)

        # Load into underlying model (bypass DataParallel)
        model_ref.load_state_dict(filtered_sd, strict=False)
        print(f"  Loaded {len(filtered_sd)} params, skipped {len(skipped)}")
        if skipped:
            print(f"  Skipped (shape mismatch): {skipped}")

        trainer.current_epoch = checkpoint.get("epoch", 0)
        trainer.best_val_loss = float("inf")
        trainer.best_cls_acc = 0.0

        # Optimizer state: always skip when resuming with different num_classes
        # (old optimizer has stale buffers for classifier head that waste GPU memory)
        if skipped:
            print("  Optimizer state skipped (classifier changed, starting fresh)")
        elif "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("  Optimizer state restored")
            except Exception as e:
                print(f"  Optimizer state incompatible, starting fresh: {e}")

        # Free checkpoint from memory
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

    try:
        history = trainer.train(
            num_epochs=args.epochs,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
            early_stopping_patience=args.early_stopping,
        )

        print("\nTraining complete!")
        print(f"Best val_cls_acc: {trainer.best_cls_acc:.4f}")
        print(f"Checkpoints saved to: {args.output_dir}")
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"TRAINING FAILED: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
