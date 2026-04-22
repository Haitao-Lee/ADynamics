"""
ADynamics Stage 1: VAE Training Script

Trains the 3D VAE model for learning compressed latent representations
of T1-weighted MRI scans.

Usage:
    python scripts/train_stage1_vae.py --config configs/vae_train.yaml

The script will:
    1. Load configuration from YAML
    2. Create train/validation dataloaders
    3. Initialize the 3D VAE model
    4. Train for specified epochs with checkpointing
    5. Save the best model based on validation loss
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.optim as AdamW
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data.dataset import get_dataloader
from core_data.transforms import get_train_transforms, get_val_transforms
from engine.trainer_vae import VAETrainer
from models.vae3d import ADynamicsVAE3D


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_data_list_from_directory(
    data_dir: str,
    extensions: tuple = (".nii", ".nii.gz"),
) -> List[Dict[str, Any]]:
    """
    Scan directory for MRI files and create data list.

    Assumes directory structure with subdirectories for each disease stage:
        data_dir/
        ├── NC/
        │   ├── sub-NC001_T1.nii.gz
        │   └── ...
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

    # Disease stage mapping
    stage_map = {
        "NC": 0,
        "SCD": 1,
        "MCI": 2,
        "AD": 3,
    }

    # Scan subdirectories
    for stage_name, stage_label in stage_map.items():
        stage_dir = data_dir / stage_name
        if not stage_dir.exists():
            print(f"Warning: Stage directory not found: {stage_dir}")
            continue

        # Find all MRI files
        for ext in extensions:
            for mri_file in stage_dir.glob(f"*{ext}"):
                data_list.append({
                    "image": str(mri_file),
                    "label": stage_label,
                })

    print(f"Found {len(data_list)} MRI files across all stages")
    return data_list


def create_dummy_data_list(num_samples: int = 20) -> List[Dict[str, Any]]:
    """
    Create a dummy data list for testing without real MRI data.

    Args:
        num_samples: Number of dummy samples to create

    Returns:
        List of data dictionaries
    """
    from core_data.dataset import create_dummy_dataset

    return create_dummy_dataset(
        spatial_size=(128, 128, 128),
        num_samples=num_samples,
    )


def main():
    """
    Main training function for Stage 1 VAE.
    """
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

    args = parser.parse_args()

    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        # Use default config if file doesn't exist
        config = {
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "epochs": 300,
            "spatial_size": [128, 128, 128],
            "in_channels": 1,
            "latent_channels": 64,
            "kl_weight": 0.0001,
            "data_dir": "./data",
            "output_dir": "./checkpoints/stage1_vae",
            "val_split": 0.2,
            "num_workers": 4,
            "checkpoint": {
                "save_interval": 50,
                "save_best": True,
            },
            "logging": {
                "log_interval": 10,
            },
        }
        print("Using default configuration")

    # Override config with command line arguments
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    print("\n" + "=" * 60)
    print("ADynamics Stage 1: 3D VAE Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Output directory: {config['output_dir']}")

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Prepare data
    if args.use_dummy_data:
        print("\nUsing dummy data for training")
        data_list = create_dummy_data_list(num_samples=20)
    else:
        print(f"\nLoading data from: {config['data_dir']}")
        if os.path.exists(config["data_dir"]):
            data_list = prepare_data_list_from_directory(config["data_dir"])
        else:
            print(f"Warning: Data directory not found: {config['data_dir']}")
            print("Using dummy data for training")
            data_list = create_dummy_data_list(num_samples=20)

    if len(data_list) == 0:
        raise ValueError("No data found. Please provide valid data or use --use_dummy_data")

    # Create transforms
    spatial_size = tuple(config["spatial_size"])
    train_transforms = get_train_transforms(spatial_size=spatial_size)
    val_transforms = get_val_transforms(spatial_size=spatial_size)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = get_dataloader(
        data_list=data_list,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 4),
        val_split=config.get("val_split", 0.2),
        shuffle=True,
        seed=config.get("seed", 42),
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    print("\nInitializing 3D VAE model...")
    model = ADynamicsVAE3D(
        spatial_size=spatial_size,
        in_channels=config.get("in_channels", 1),
        latent_channels=config.get("latent_channels", 64),
    )

    # Create optimizer
    optimizer = AdamW(
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

    # Create trainer
    trainer = VAETrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        config={
            "kl_weight": config.get("kl_weight", 0.0001),
            "recon_loss_type": config.get("recon_loss_type", "l1"),
        },
        scheduler=scheduler,
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    print("=" * 60)

    history = trainer.train(
        num_epochs=num_epochs,
        save_interval=config.get("checkpoint", {}).get("save_interval", 50),
        output_dir=config["output_dir"],
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Checkpoints saved to: {config['output_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
