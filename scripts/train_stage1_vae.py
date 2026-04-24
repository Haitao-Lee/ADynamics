"""
ADynamics Stage 1: VAE Training Script

Trains the 3D VAE model for learning compressed latent representations
of T1-weighted MRI scans.

HD Configuration:
    - Input spatial size: (256, 256, 192)
    - Latent spatial size: (16, 16, 12)
    - Default batch_size: 1 (GPU memory efficient for HD)

Usage:
    python scripts/train_stage1_vae.py --config configs/vae_train.yaml
    python scripts/train_stage1_vae.py --data_dir ./data --epochs 300
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data.dataset import get_train_val_test_dataloaders, create_dummy_dataset
from core_data.transforms import get_train_transforms, get_val_transforms
from engine.trainer_vae import VAETrainer
from models.vae3d import ADynamicsVAE3D


# HD configuration defaults
HD_SPATIAL_SIZE = (256, 256, 192)
HD_LATENT_SPATIAL = (16, 16, 12)
HD_LATENT_CHANNELS = 64


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_data_list_from_directory(
    data_dir: str,
    extensions: tuple = (".nii", ".nii.gz"),
) -> List[Dict[str, Any]]:
    """
    Scan directory for MRI files and create data list.

    Assumes directory structure:
        data_dir/
        ├── NC/
        ├── SCD/
        ├── MCI/
        └── AD/

    Args:
        data_dir: Root directory containing MRI data
        extensions: Tuple of valid file extensions

    Returns:
        List of dictionaries with "image" and "label" keys
    """
    data_dir = Path(data_dir)
    data_list = []

    stage_map = {"NC": 0, "SCD": 1, "MCI": 2, "AD": 3}

    for stage_name, stage_label in stage_map.items():
        stage_dir = data_dir / stage_name
        if not stage_dir.exists():
            print(f"Warning: Stage directory not found: {stage_dir}")
            continue

        for ext in extensions:
            for mri_file in stage_dir.glob(f"*{ext}"):
                data_list.append({
                    "image": str(mri_file),
                    "label": stage_label,
                })

    # Force deterministic ordering for reproducible splits (architect directive #3)
    data_list.sort(key=lambda x: x["image"])

    print(f"Found {len(data_list)} MRI files across all stages")
    return data_list


def prepare_data_list_from_json(
    json_path: str,
    modality: str = "t1",
) -> List[Dict[str, Any]]:
    """
    Load data list from a JSON dataset file.

    Expected JSON structure:
        [
            {
                "patient_id": "0_0_sub001",
                "center": "center0 (China-Japan)",
                "label": 3,
                "t1": "path/to/t1.nii.gz",
                "fmri": "path/to/fmri.nii.gz",
                "qsm": "path/to/qsm.nii.gz",
                "asl": "path/to/asl.nii.gz",
                "flair": "path/to/flair.nii.gz"
            },
            ...
        ]

    Supported modalities: "t1", "fmri", "qsm", "asl", "flair"

    Args:
        json_path: Path to the JSON dataset file
        modality: Which modality to use for training (default: "t1")

    Returns:
        List of dictionaries with "image" and "label" keys

    Raises:
        FileNotFoundError: If JSON file does not exist
        ValueError: If modality is not supported
    """
    import json

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON dataset file not found: {json_path}")

    supported_modalities = ["t1", "fmri", "qsm", "asl", "flair"]
    if modality not in supported_modalities:
        raise ValueError(f"Unsupported modality: {modality}. Use one of {supported_modalities}")

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    data_list = []
    skipped = 0

    for item in dataset:
        label = item.get("label")
        if label is None:
            skipped += 1
            continue

        image_path = item.get(modality)
        if not image_path or not Path(image_path).exists():
            skipped += 1
            continue

        data_list.append({
            "image": str(image_path),
            "label": int(label),
            "patient_id": item.get("patient_id", ""),
            "center": item.get("center", ""),
        })

    # Force deterministic ordering for reproducible splits
    data_list.sort(key=lambda x: x["image"])

    if skipped > 0:
        print(f"Warning: Skipped {skipped} entries (missing label or file)")

    print(f"Found {len(data_list)} samples from JSON ({modality})")
    return data_list


def create_dummy_data_list(
    num_samples: int = 20,
    spatial_size: tuple = HD_SPATIAL_SIZE,
) -> List[Dict[str, Any]]:
    """Create a dummy data list for testing without real MRI data."""
    return create_dummy_dataset(
        spatial_size=spatial_size,
        num_samples=num_samples,
    )


def log_reconstruction_to_tensorboard(
    writer: SummaryWriter,
    images: torch.Tensor,
    recons: torch.Tensor,
    epoch: int,
    tag: str = "Reconstruction",
) -> None:
    """
    Log middle axial slice of image/recon comparison to TensorBoard.

    Args:
        writer: TensorBoard summary writer
        images: Original images [B, 1, D, H, W]
        recons: Reconstructed images [B, 1, D, H, W]
        epoch: Current epoch number
        tag: Tag for TensorBoard
    """
    # Take first sample from batch
    orig = images[0, 0]  # [D, H, W]
    recon = recons[0, 0]  # [D, H, W]

    # Middle axial slice
    d_mid = orig.shape[0] // 2
    orig_slice = orig[d_mid].cpu().detach()  # [H, W]
    recon_slice = recon[d_mid].cpu().detach()  # [H, W]

    # Stack horizontally: original | reconstruction
    comparison = torch.cat([orig_slice, recon_slice], dim=1)  # [H, 2*W]

    # Clamp to valid range [0, 1] for visualization
    comparison = torch.clamp(comparison, 0.0, 1.0)

    writer.add_image(f"{tag}/axial_mid", comparison.unsqueeze(0), epoch)


def main():
    parser = argparse.ArgumentParser(
        description="Train ADynamics 3D VAE for Stage 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vae_train.yaml",
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to MRI data directory (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--use_dummy_data",
        action="store_true",
        help="Use dummy data instead of real MRI data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (cuda/cpu)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--spatial_size",
        type=int,
        nargs=3,
        default=None,
        help="Spatial size (D, H, W). Defaults to HD (256, 256, 192)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default='./dataset_manifest_merged.json',
        help="Path to JSON dataset file (alternative to --data_dir)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="t1",
        choices=["t1", "fmri", "qsm", "asl", "flair"],
        help="Modality to use when loading from JSON (default: t1)",
    )

    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        # HD-optimized default config
        config = {
            "batch_size": 1,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "epochs": 300,
            "spatial_size": list(HD_SPATIAL_SIZE),
            "in_channels": 1,
            "latent_channels": HD_LATENT_CHANNELS,
            "kl_weight": 0.0001,
            "data_dir": "./data",
            "output_dir": "./checkpoints/stage1_vae",
            "val_split": 0.2,
            "num_workers": 4,
            "use_amp": True,
            "checkpoint": {
                "save_interval": 50,
                "save_best": True,
            },
            "logging": {
                "log_interval": 10,
            },
        }
        print("Using default HD configuration")

    # Override config with command line arguments
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.spatial_size is not None:
        config["spatial_size"] = list(args.spatial_size)

    spatial_size = tuple(config["spatial_size"])

    print("\n" + "=" * 60)
    print("ADynamics Stage 1: 3D VAE Training (HD)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"HD Spatial size: {spatial_size}")
    print(f"Latent spatial: {HD_LATENT_SPATIAL}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['output_dir']}")

    os.makedirs(config["output_dir"], exist_ok=True)

    # TensorBoard writer
    tb_dir = os.path.join(config["output_dir"], "tensorboard")
    writer = SummaryWriter(tb_dir)

    # Prepare data
    data_list = []
    if args.use_dummy_data:
        print("\nUsing dummy data for training")
        data_list = create_dummy_data_list(num_samples=20, spatial_size=spatial_size)
    elif args.data_dir is not None:
        print(f"\nLoading data from: {config['data_dir']}")
        if os.path.exists(config["data_dir"]):
            data_list = prepare_data_list_from_directory(config["data_dir"])
        else:
            print(f"Warning: Data directory not found: {config['data_dir']}")
            print("Using dummy data for training")
            data_list = create_dummy_data_list(num_samples=20, spatial_size=spatial_size)
    elif args.json is not None:
        # Load from JSON dataset file
        print(f"\nLoading data from JSON: {args.json}")
        print(f"Using modality: {args.modality}")
        data_list = prepare_data_list_from_json(args.json, modality=args.modality)

    if len(data_list) == 0:
        raise ValueError("No data found. Please provide valid data or use --use_dummy_data")

    # Deterministic sort for reproducible splits (architect directive #3)
    data_list.sort(key=lambda x: x["image"])

    train_transforms = get_train_transforms(spatial_size=spatial_size)
    val_transforms = get_val_transforms(spatial_size=spatial_size)

    print("\nCreating dataloaders...")
    train_loader, val_loader, _ = get_train_val_test_dataloaders(
        data_list=data_list,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        test_transforms=val_transforms,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 4),
        train_split=1.0 - config.get("val_split", 0.2),
        val_split=config.get("val_split", 0.2),
        shuffle=True,
        seed=config.get("seed", 42),
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Extract fixed visualization batch from val_loader (before training to avoid performance overhead)
    viz_batch = None
    for batch in val_loader:
        viz_batch = batch["image"].to(args.device)
        break

    # Create model
    print("\nInitializing 3D VAE model...")
    model = ADynamicsVAE3D(
        spatial_size=spatial_size,
        in_channels=config.get("in_channels", 1),
        latent_channels=config.get("latent_channels", HD_LATENT_CHANNELS),
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-5),
        betas=(0.9, 0.999),
    )

    # Create scheduler
    num_epochs = config["epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=config.get("learning_rate", 1e-4) * 0.01,
    )

    # Trainer config with AMP enabled
    trainer_config = {
        "kl_weight": config.get("kl_weight", 0.0001),
        "recon_loss_type": config.get("recon_loss_type", "l1"),
        "use_amp": config.get("use_amp", True),
    }

    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        config=trainer_config,
        scheduler=scheduler,
    )

    # Resume from checkpoint if specified (architect directive #2)
    if args.resume is not None:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        else:
            print(f"\nWarning: Resume checkpoint not found: {args.resume}")
            print("Starting training from scratch.")

    # Train
    print("\nStarting training...")
    print("=" * 60)

    # Hook for TensorBoard logging every 10 epochs
    original_train_epoch = trainer.train_epoch
    use_amp = trainer_config.get("use_amp", True)

    def train_epoch_with_tb(current_kl_weight: float) -> Dict[str, float]:
        metrics = original_train_epoch(current_kl_weight)

        # Log reconstruction every 10 epochs using pre-extracted viz_batch
        if (trainer.current_epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=use_amp):
                    recon, mu, logvar = model(viz_batch)
                log_reconstruction_to_tensorboard(
                    writer, viz_batch, recon, trainer.current_epoch + 1
                )
            model.train()

        return metrics

    trainer.train_epoch = train_epoch_with_tb

    history = trainer.train(
        num_epochs=num_epochs,
        save_interval=config.get("checkpoint", {}).get("save_interval", 50),
        output_dir=config["output_dir"],
    )

    writer.close()

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Checkpoints saved to: {config['output_dir']}")
    print(f"TensorBoard logs saved to: {tb_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
