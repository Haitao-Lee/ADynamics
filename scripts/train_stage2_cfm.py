"""
ADynamics Stage 2 & 3: Classifier + CFM Training Script

Combines Stage 2 (Disease Classifier) and Stage 3 (Conditional Flow Matching)
training in a single script. The script:
1. Loads pretrained VAE encoder (frozen)
2. Extracts latent features from MRI data using proper MONAI transforms
3. Trains the Disease Classifier (Stage 2) with AdaptiveAvgPool3d
4. Trains the Velocity Field Network (Stage 3)

HD Configuration:
    - Input spatial size: (256, 256, 192)
    - Latent spatial size: (16, 16, 12)
    - Uses proper RAS orientation via MONAI transforms

Usage:
    python scripts/train_stage2_cfm.py --config configs/cfm_train.yaml --stage 2
    python scripts/train_stage2_cfm.py --config configs/cfm_train.yaml --stage 3
    python scripts/train_stage2_cfm.py --vae_checkpoint path/to/vae_best.pt --stage 3
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data.dataset import (
    create_dummy_dataset,
    get_train_val_test_dataloaders,
    cleanup_dummy_dataset,
)
from core_data.transforms import get_train_transforms, get_val_transforms
from engine.trainer_cfm import CFMTrainer
from models.classifier import DiseaseClassifier, classifier_ce_loss, classifier_accuracy
from models.vae3d import ADynamicsVAE3D
from models.vector_field import VelocityFieldNet


# HD configuration
HD_SPATIAL_SIZE = (256, 256, 192)
HD_LATENT_SPATIAL = (16, 16, 12)
HD_LATENT_CHANNELS = 64


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

    If participants.csv exists, reads real age. Otherwise uses random age with warning.

    Args:
        data_dir: Root directory containing stage subdirectories
        extensions: File extensions to look for

    Returns:
        List of data dictionaries with "image", "label", and optionally "condition" (age)
    """
    data_dir = Path(data_dir)
    data_list = []

    stage_map = {"NC": 0, "SCD": 1, "MCI": 2, "AD": 3}

    # Check for participants.csv
    csv_path = data_dir / "participants.csv"
    use_real_age = False
    if csv_path.exists():
        print(f"  Found participants.csv, reading real age data...")
        try:
            participants_df = pd.read_csv(csv_path)
            if "age" in participants_df.columns and "filename" in participants_df.columns:
                use_real_age = True
                age_dict = dict(zip(participants_df["filename"], participants_df["age"]))
            elif "age" in participants_df.columns:
                # Use row index as filename
                use_real_age = True
                age_dict = {}
                for idx, row in participants_df.iterrows():
                    age_dict[row.get("filename", f"image_{idx}") if "filename" in participants_df.columns else str(idx)] = row["age"]
        except Exception as e:
            warnings.warn(f"  Could not read participants.csv: {e}, using random ages")
            use_real_age = False
    else:
        warnings.warn("  No participants.csv found. Using random ages (NOT for real experiments!)")

    for stage_name, stage_label in stage_map.items():
        stage_dir = data_dir / stage_name
        if not stage_dir.exists():
            continue

        for ext in extensions:
            for mri_file in stage_dir.glob(f"*{ext}"):
                if use_real_age:
                    filename = mri_file.name
                    if filename in age_dict:
                        age = age_dict[filename]
                        condition = age / 100.0
                    else:
                        condition = np.random.normal(70, 10) / 100.0
                else:
                    condition = np.random.normal(70, 10) / 100.0

                data_list.append({
                    "image": str(mri_file),
                    "label": stage_label,
                    "condition": condition,
                })

    return data_list


def encode_dataset_to_latents(
    data_list: List[Dict[str, Any]],
    vae_model: nn.Module,
    spatial_size: Tuple[int, int, int],
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Encode entire dataset to latents using frozen VAE with proper MONAI transforms.

    Uses DataLoader with multi-threading for efficient batch loading.
    Ensures RAS orientation and 1mm resampling via get_val_transforms.

    Args:
        data_list: List of data dictionaries
        vae_model: Frozen VAE encoder
        spatial_size: HD spatial size (256, 256, 192)
        device: Device to run encoding
        batch_size: Batch size for encoding
        num_workers: Number of workers for DataLoader

    Returns:
        Tuple of (latents, labels, conditions)
    """
    vae_model.eval()

    # Get transforms with proper RAS orientation and 1mm resampling
    transforms = get_val_transforms(spatial_size=spatial_size)

    # Create a simple Dataset that applies MONAI transforms
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, data_list, transform):
            self.data_list = data_list
            self.transform = transform

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            item = self.data_list[idx]
            # Apply MONAI transforms (handles LoadImaged, Orientation, Spacing, etc.)
            result = self.transform(item)
            return result

    dataset = TransformDataset(data_list, transforms)

    # Use DataLoader with multiple workers for parallel I/O
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    latents_list = []
    labels_list = []
    conditions_list = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"]
            conditions = batch.get("condition", None)
            if conditions is not None and conditions.dim() == 1:
                conditions = conditions.unsqueeze(-1)

            # Extract latent using VAE encoder
            mu, _ = vae_model.encode(images)
            latents_list.append(mu.cpu())
            labels_list.append(labels)
            if conditions is not None:
                conditions_list.append(conditions.cpu())

    latents = torch.cat(latents_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    conditions = torch.cat(conditions_list, dim=0) if conditions_list else None

    return latents, labels, conditions


def train_classifier(
    latents: torch.Tensor,
    labels: torch.Tensor,
    conditions: Optional[torch.Tensor],
    config: Dict[str, Any],
    device: torch.device,
) -> DiseaseClassifier:
    """
    Train the disease classifier (Stage 2) with AdaptiveAvgPool3d.

    Uses the updated DiseaseClassifier that takes vae_latent_channels and pooling_size
    instead of hardcoded latent_dim to prevent OOM with HD latents.

    Args:
        latents: VAE latents [N, C, 16, 16, 12] for HD
        labels: Disease labels [N]
        conditions: Clinical conditions [N, 1]
        config: Training configuration
        device: Device to train on

    Returns:
        Trained DiseaseClassifier
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 60)
    print("Stage 2: Training Disease Classifier with AdaptiveAvgPool3d")
    print("=" * 60)

    latent_channels = latents.shape[1]
    print(f"Latent shape: [{latent_channels}, {HD_LATENT_SPATIAL[0]}, {HD_LATENT_SPATIAL[1]}, {HD_LATENT_SPATIAL[2]}]")

    # Create classifier with pooling-based architecture (no hardcoded latent_dim)
    classifier = DiseaseClassifier(
        vae_latent_channels=latent_channels,
        pooling_size=HD_LATENT_SPATIAL,
        hidden_dims=[512, 256, 128],
        num_classes=4,
        dropout=config.get("classifier", {}).get("dropout", 0.3),
    ).to(device)

    lr = config.get("classifier", {}).get("learning_rate", 1e-4)
    wd = config.get("classifier", {}).get("weight_decay", 1e-5)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=wd)

    epochs = config.get("classifier", {}).get("epochs", 200)
    batch_size = config.get("batch_size", 32)

    # Stratified train/val split using sklearn
    n_samples = len(latents)
    labels_np = labels.cpu().numpy()
    indices = np.arange(n_samples)

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels_np,
        random_state=42,
    )

    train_dataset = LatentDatasetPrecomputed(
        latents[train_idx], labels[train_idx],
        conditions[train_idx] if conditions is not None else None
    )
    val_dataset = LatentDatasetPrecomputed(
        latents[val_idx], labels[val_idx],
        conditions[val_idx] if conditions is not None else None
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    best_acc = 0.0
    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch in train_loader:
            z = batch["latent"].to(device)
            y = batch["label"].to(device)

            logits = classifier(z)
            loss = classifier_ce_loss(logits, y)
            acc = classifier_accuracy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

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

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {avg_loss:.4f} Acc: {avg_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
            )

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
        latents: VAE latents [N, C, 16, 16, 12] for HD
        labels: Disease labels [N]
        conditions: Clinical conditions [N, num_conditions]
        config: Training configuration
        device: Device to train on

    Returns:
        Trained VelocityFieldNet
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 60)
    print("Stage 3: Training CFM Velocity Field Network (HD)")
    print("=" * 60)

    latent_channels = latents.shape[1]
    latent_spatial = tuple(latents.shape[2:])
    print(f"Latent shape: [{latent_channels}, {latent_spatial[0]}, {latent_spatial[1]}, {latent_spatial[2]}]")

    cfm_config = config.get("cfm", {})
    vector_field = VelocityFieldNet(
        latent_channels=latent_channels,
        latent_spatial=HD_LATENT_SPATIAL,
        time_embed_dim=cfm_config.get("time_embed_dim", 128),
        time_hidden_dim=cfm_config.get("time_hidden_dim", 256),
        cond_embed_dim=cfm_config.get("cond_embed_dim", 64),
        cond_hidden_dim=cfm_config.get("cond_hidden_dim", 128),
        num_conditions=cfm_config.get("num_conditions", 1),
        base_channels=cfm_config.get("base_channels", 64),
        channel_mults=tuple(cfm_config.get("channel_mults", [1, 2, 4])),
        num_res_blocks=cfm_config.get("num_res_blocks", 2),
    ).to(device)

    lr = cfm_config.get("learning_rate", 1e-4)
    wd = cfm_config.get("weight_decay", 1e-5)
    optimizer = torch.optim.AdamW(vector_field.parameters(), lr=lr, weight_decay=wd)

    epochs = cfm_config.get("epochs", 500)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Stratified train/val split using sklearn
    n_samples = len(latents)
    labels_np = labels.cpu().numpy()
    indices = np.arange(n_samples)

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels_np,
        random_state=42,
    )

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

    trainer = CFMTrainer(
        model=vector_field,
        optimizer=optimizer,
        device=device,
        config={
            "velocity_loss_weight": cfm_config.get("velocity_loss_weight", 1.0),
            "batch_size": batch_size,
        },
        scheduler=scheduler,
    )

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

    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = {
            "batch_size": 32,
            "cfm": {
                "latent_channels": HD_LATENT_CHANNELS,
                "latent_spatial": list(HD_LATENT_SPATIAL),
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

    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir

    device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    print(f"HD Configuration: spatial_size={HD_SPATIAL_SIZE}, latent_spatial={HD_LATENT_SPATIAL}")

    # Prepare data
    if args.use_dummy_data:
        print("\nUsing dummy data for training")
        data_list = create_dummy_dataset(
            spatial_size=HD_SPATIAL_SIZE,
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
                spatial_size=HD_SPATIAL_SIZE,
                num_samples=40,
            )

    print(f"Total samples: {len(data_list)}")

    # Load VAE with HD configuration
    vae = ADynamicsVAE3D(
        spatial_size=HD_SPATIAL_SIZE,
        in_channels=1,
        latent_channels=HD_LATENT_CHANNELS,
        base_channels=32,
    )

    if args.vae_checkpoint and os.path.exists(args.vae_checkpoint):
        print(f"Loading VAE from {args.vae_checkpoint}")
        state_dict = torch.load(args.vae_checkpoint, map_location=device)
        vae.load_state_dict(state_dict["model_state_dict"])
    else:
        print("Warning: No VAE checkpoint provided. Using untrained VAE.")

    vae = vae.to(device)
    vae.eval()

    # Extract latents with proper MONAI transforms and multi-threaded loading
    print("\nExtracting latents from VAE (using MONAI transforms + DataLoader)...")
    latents, labels, conditions = encode_dataset_to_latents(
        data_list, vae, HD_SPATIAL_SIZE, device, batch_size=8, num_workers=4
    )
    print(f"Latents shape: {latents.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Conditions shape: {conditions.shape if conditions is not None else 'None'}")

    # Cleanup dummy data
    if args.use_dummy_data:
        cleanup_dummy_dataset(data_list)

    # Train requested stage
    if args.stage == 2:
        classifier = train_classifier(latents, labels, conditions, config, device)
        os.makedirs(config["output_dir"], exist_ok=True)
        torch.save(classifier.state_dict(), os.path.join(config["output_dir"], "classifier_final.pt"))
        print(f"\nClassifier saved to {config['output_dir']}/classifier_final.pt")

    elif args.stage == 3:
        vector_field = train_cfm(latents, labels, conditions, config, device)

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()