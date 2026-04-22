"""
ADynamics Stage 2 & 3: Classifier + CFM Training Script

Combines Stage 2 (Disease Classifier) and Stage 3 (Conditional Flow Matching)
training in a single script. The script:
1. Loads pretrained VAE encoder (frozen)
2. Extracts latent features from MRI data
3. Trains the Disease Classifier (Stage 2)
4. Trains the Velocity Field Network (Stage 3)

Usage:
    # Train both stages
    python scripts/train_stage2_cfm.py --config configs/cfm_train.yaml --stage 2
    python scripts/train_stage2_cfm.py --config configs/cfm_train.yaml --stage 3

    # Train with specific VAE checkpoint
    python scripts/train_stage2_cfm.py --vae_checkpoint path/to/vae_best.pt --stage 3

    # Resume training from checkpoint
    python scripts/train_stage2_cfm.py --resume path/to/checkpoint.pt --stage 3
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as AdamW
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data.dataset import create_dummy_dataset, get_dataloader
from core_data.transforms import get_train_transforms, get_val_transforms
from engine.trainer_cfm import CFMTrainer
from models.classifier import DiseaseClassifier, classifier_ce_loss, classifier_accuracy
from models.vae3d import ADynamicsVAE3D
from models.vector_field import VelocityFieldNet


class LatentDataset(Dataset):
    """
    Dataset that extracts latents from MRI using a frozen VAE encoder.

    Stores (image_path, label, condition) and extracts latents on-the-fly
    or caches them for faster training.
    """

    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        vae_model: nn.Module,
        transform=None,
        device: torch.device = torch.device("cpu"),
        cache_latents: bool = False,
    ) -> None:
        """
        Initialize latent dataset.

        Args:
            data_list: List of dicts with "image" (path), "label" (int), optionally "condition" (float)
            vae_model: Frozen VAE model for encoding
            transform: Optional MONAI transforms for preprocessing
            device: Device to run VAE encoding
            cache_latents: Whether to cache latents in memory
        """
        self.data_list = data_list
        self.vae_model = vae_model
        self.transform = transform
        self.device = device
        self.cache_latents = cache_latents

        # Cache for latents
        if cache_latents:
            self.latent_cache = [None] * len(data_list)

        # Set VAE to eval mode
        self.vae_model.eval()

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample.

        Returns:
            Dictionary with "latent", "label", and optionally "condition"
        """
        item = self.data_list[idx]

        # Check cache
        if self.cache_latents and self.latent_cache[idx] is not None:
            return self.latent_cache[idx]

        # Load and preprocess image
        import nibabel as nib

        nii = nib.load(item["image"])
        image = nii.get_fdata().astype(np.float32)
        # shape: [D, H, W]

        # Add channel and batch dims
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        # shape: [1, 1, D, H, W]

        # Apply transforms if provided
        if self.transform is not None:
            # For single image
            image = self.transform({"image": image})["image"]

        # Move to device and encode
        image = image.to(self.device)

        with torch.no_grad():
            # Extract latent using VAE encoder
            mu, _ = self.vae_model.encode(image)
            # shape: [1, latent_channels, D', H', W']

            # Use mean as latent representation
            latent = mu[0]  # shape: [latent_channels, D', H', W']
            label = item["label"]

            # Condition (e.g., normalized age)
            condition = item.get("condition", None)
            if condition is not None:
                condition = torch.tensor([condition], dtype=torch.float32)

        result = {
            "latent": latent,
            "label": label,
        }

        if condition is not None:
            result["condition"] = condition

        # Cache if enabled
        if self.cache_latents:
            self.latent_cache[idx] = result

        return result


class LatentDatasetPrecomputed(Dataset):
    """
    Dataset for precomputed latents (faster training after one encoding pass).
    """

    def __init__(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize with precomputed latents.

        Args:
            latents: Precomputed latents [N, C, D, H, W]
            labels: Disease labels [N]
            conditions: Optional clinical conditions [N, num_conditions]
        """
        self.latents = latents
        self.labels = labels
        self.conditions = conditions

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        result = {
            "latent": self.latents[idx],
            "label": self.labels[idx],
        }
        if self.conditions is not None:
            result["condition"] = self.conditions[idx]
        return result


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_data_list_from_directory(
    data_dir: str,
    extensions: tuple = (".nii", ".nii.gz"),
) -> List[Dict[str, Any]]:
    """
    Scan directory for MRI files and create data list.

    Expected structure:
        data_dir/
        ├── NC/
        ├── SCD/
        ├── MCI/
        └── AD/
    """
    data_dir = Path(data_dir)
    data_list = []

    stage_map = {"NC": 0, "SCD": 1, "MCI": 2, "AD": 3}

    for stage_name, stage_label in stage_map.items():
        stage_dir = data_dir / stage_name
        if not stage_dir.exists():
            continue

        for ext in extensions:
            for mri_file in stage_dir.glob(f"*{ext}"):
                # Generate dummy condition (normalized age ~70)
                condition = np.random.normal(70, 10) / 100.0  # Normalized to ~0.7

                data_list.append({
                    "image": str(mri_file),
                    "label": stage_label,
                    "condition": condition,
                })

    return data_list


def encode_dataset_to_latents(
    data_list: List[Dict[str, Any]],
    vae_model: nn.Module,
    transform,
    device: torch.device,
    batch_size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Encode entire dataset to latents using frozen VAE.

    Args:
        data_list: List of data dictionaries
        vae_model: Frozen VAE encoder
        transform: MONAI transforms
        device: Device to run encoding
        batch_size: Batch size for encoding

    Returns:
        Tuple of (latents, labels, conditions)
    """
    vae_model.eval()

    latents_list = []
    labels_list = []
    conditions_list = []

    for i in range(0, len(data_list), batch_size):
        batch_items = data_list[i:i + batch_size]

        images = []
        labels = []
        conditions = []

        for item in batch_items:
            import nibabel as nib

            nii = nib.load(item["image"])
            image = nii.get_fdata().astype(np.float32)
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

            if transform is not None:
                image = transform({"image": image})["image"]

            images.append(image)
            labels.append(item["label"])
            conditions.append(item.get("condition", 0.7))

        # Batch
        images = torch.cat(images, dim=0).to(device)
        labels = torch.tensor(labels)
        conditions = torch.tensor(conditions, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            mu, _ = vae_model.encode(images)
            latents_list.append(mu.cpu())
            labels_list.append(labels)
            conditions_list.append(conditions)

    latents = torch.cat(latents_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    conditions = torch.cat(conditions_list, dim=0)

    return latents, labels, conditions


def train_classifier(
    latents: torch.Tensor,
    labels: torch.Tensor,
    conditions: Optional[torch.Tensor],
    config: Dict[str, Any],
    device: torch.device,
) -> DiseaseClassifier:
    """
    Train the disease classifier (Stage 2).

    Args:
        latents: VAE latents [N, C, D, H, W]
        labels: Disease labels [N]
        conditions: Clinical conditions [N, 1]
        config: Training configuration
        device: Device to train on

    Returns:
        Trained DiseaseClassifier
    """
    print("\n" + "=" * 60)
    print("Stage 2: Training Disease Classifier")
    print("=" * 60)

    # Calculate latent dimension
    latent_dim = latents.shape[1] * latents.shape[2] * latents.shape[3] * latents.shape[4]
    print(f"Latent dimension: {latent_dim}")

    # Create classifier
    classifier = DiseaseClassifier(
        latent_dim=latent_dim,
        hidden_dims=[2048, 1024, 512],
        num_classes=4,
        dropout=config.get("classifier", {}).get("dropout", 0.3),
    ).to(device)

    # Create optimizer
    lr = config.get("classifier", {}).get("learning_rate", 1e-4)
    wd = config.get("classifier", {}).get("weight_decay", 1e-5)
    optimizer = AdamW(classifier.parameters(), lr=lr, weight_decay=wd)

    # Training
    epochs = config.get("classifier", {}).get("epochs", 200)
    batch_size = config.get("batch_size", 32)

    # Simple train/val split
    n_samples = len(latents)
    n_val = int(n_samples * 0.2)
    indices = torch.randperm(n_samples)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    # Create dataloaders
    train_dataset = LatentDatasetPrecomputed(
        latents[train_idx], labels[train_idx],
        conditions[train_idx] if conditions is not None else None
    )
    val_dataset = LatentDatasetPrecomputed(
        latents[val_idx], labels[val_idx],
        conditions[val_idx] if conditions is not None else None
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    best_acc = 0.0
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch in train_loader:
            z = batch["latent"].to(device)
            y = batch["label"].to(device)

            # Forward
            logits = classifier(z)
            loss = classifier_ce_loss(logits, y)
            acc = classifier_accuracy(logits, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        # Validation
        classifier.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                z = batch["latent"].to(device)
                y = batch["label"].to(device)

                logits = classifier(z)
                loss = classifier_ce_loss(logits, y)
                acc = classifier_accuracy(logits, y)

                val_loss += loss.item()
                val_acc += acc.item()
                val_batches += 1

        val_loss /= val_batches
        val_acc /= val_batches

        # Log
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {avg_loss:.4f} Acc: {avg_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
            )

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(classifier.state_dict(), "classifier_best.pt")

    print(f"Best validation accuracy: {best_acc:.4f}")
    return classifier


def train_cfm(
    latents: torch.Tensor,
    labels: torch.Tensor,
    conditions: Optional[torch.Tensor],
    config: Dict[str, Any],
    device: torch.device,
) -> VelocityFieldNet:
    """
    Train the CFM velocity field network (Stage 3).

    Args:
        latents: VAE latents [N, C, D, H, W]
        labels: Disease labels [N]
        conditions: Clinical conditions [N, num_conditions]
        config: Training configuration
        device: Device to train on

    Returns:
        Trained VelocityFieldNet
    """
    print("\n" + "=" * 60)
    print("Stage 3: Training CFM Velocity Field Network")
    print("=" * 60)

    # Extract latent dimensions
    latent_channels = latents.shape[1]
    latent_spatial = tuple(latents.shape[2:])
    print(f"Latent shape: [{latent_channels}, {latent_spatial[0]}, {latent_spatial[1]}, {latent_spatial[2]}]")

    # Create velocity field network
    cfm_config = config.get("cfm", {})
    vector_field = VelocityFieldNet(
        latent_channels=latent_channels,
        latent_spatial=latent_spatial,
        time_embed_dim=cfm_config.get("time_embed_dim", 128),
        time_hidden_dim=cfm_config.get("time_hidden_dim", 256),
        cond_embed_dim=cfm_config.get("cond_embed_dim", 64),
        cond_hidden_dim=cfm_config.get("cond_hidden_dim", 128),
        num_conditions=cfm_config.get("num_conditions", 1),
        base_channels=cfm_config.get("base_channels", 64),
        channel_mults=tuple(cfm_config.get("channel_mults", [1, 2, 4])),
        num_res_blocks=cfm_config.get("num_res_blocks", 2),
    ).to(device)

    # Create optimizer
    lr = cfm_config.get("learning_rate", 1e-4)
    wd = cfm_config.get("weight_decay", 1e-5)
    optimizer = AdamW(vector_field.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    epochs = cfm_config.get("epochs", 500)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Train/val split
    n_samples = len(latents)
    n_val = int(n_samples * 0.2)
    indices = torch.randperm(n_samples)
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    # Create dataloaders
    train_dataset = LatentDatasetPrecomputed(
        latents[train_idx], labels[train_idx],
        conditions[train_idx] if conditions is not None else None
    )
    val_dataset = LatentDatasetPrecomputed(
        latents[val_idx], labels[val_idx],
        conditions[val_idx] if conditions is not None else None
    )

    batch_size = cfm_config.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create trainer
    trainer = CFMTrainer(
        model=vector_field,
        optimizer=optimizer,
        device=device,
        config={
            "velocity_loss_weight": cfm_config.get("velocity_loss_weight", 1.0),
        },
        scheduler=scheduler,
    )

    # Train
    output_dir = config.get("output_dir", "./checkpoints/stage3_cfm")
    os.makedirs(output_dir, exist_ok=True)

    history = trainer.train(
        latent_loader_train=train_loader,
        latent_loader_val=val_loader,
        num_epochs=epochs,
        save_interval=config.get("checkpoint", {}).get("save_interval", 50),
        output_dir=output_dir,
    )

    print(f"\nTraining completed. Best val loss: {trainer.best_val_loss:.4f}")

    return vector_field


def main():
    parser = argparse.ArgumentParser(
        description="Train ADynamics Stage 2 (Classifier) and Stage 3 (CFM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cfm_train.yaml",
        help="Path to training configuration YAML",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[2, 3],
        required=True,
        help="Stage to train: 2 for classifier, 3 for CFM",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained VAE checkpoint (required for both stages)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to MRI data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
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
        help="Device to use (cuda/cpu)",
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
    else:
        # Default config
        config = {
            "batch_size": 32,
            "cfm": {
                "latent_channels": 64,
                "latent_spatial": [8, 8, 8],
                "time_embed_dim": 128,
                "cond_embed_dim": 64,
                "epochs": 500,
            },
            "classifier": {
                "epochs": 200,
                "dropout": 0.3,
            },
            "output_dir": "./checkpoints/stage2_cfm",
        }

    # Override config
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir

    device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # Prepare data
    if args.use_dummy_data:
        print("\nUsing dummy data for training")
        data_list = create_dummy_dataset(
            spatial_size=(128, 128, 128),
            num_samples=40,
        )
    else:
        data_dir = config.get("data_dir", "./data")
        if os.path.exists(data_dir):
            data_list = prepare_data_list_from_directory(data_dir)
        else:
            print(f"Data directory not found: {data_dir}")
            print("Using dummy data")
            data_list = create_dummy_dataset(
                spatial_size=(128, 128, 128),
                num_samples=40,
            )

    print(f"Total samples: {len(data_list)}")

    # Load VAE
    vae = ADynamicsVAE3D(
        spatial_size=(128, 128, 128),
        in_channels=1,
        latent_channels=config.get("cfm", {}).get("latent_channels", 64),
    )

    if args.vae_checkpoint and os.path.exists(args.vae_checkpoint):
        print(f"Loading VAE from {args.vae_checkpoint}")
        state_dict = torch.load(args.vae_checkpoint, map_location=device)
        vae.load_state_dict(state_dict["model_state_dict"])
    else:
        print("Warning: No VAE checkpoint provided. Using untrained VAE.")

    vae = vae.to(device)
    vae.eval()  # Freeze VAE

    # Extract latents
    print("\nExtracting latents from VAE...")
    transforms = get_train_transforms(spatial_size=(128, 128, 128))
    latents, labels, conditions = encode_dataset_to_latents(
        data_list, vae, transforms, device, batch_size=8
    )
    print(f"Latents shape: {latents.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Conditions shape: {conditions.shape}")

    # Train requested stage
    if args.stage == 2:
        classifier = train_classifier(latents, labels, conditions, config, device)
        torch.save(classifier.state_dict(), os.path.join(config["output_dir"], "classifier_final.pt"))
        print(f"\nClassifier saved to {config['output_dir']}/classifier_final.pt")

    elif args.stage == 3:
        vector_field = train_cfm(latents, labels, conditions, config, device)

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
