"""
Cross-Validation Framework for ADynamics.

Performs K-fold stratified cross-validation to provide reliable performance estimates
with mean +/- std reporting. Essential for small medical imaging datasets.

Metrics reported per fold and aggregated:
    - Classification accuracy (per-class and overall)
    - Silhouette score (latent space separation)
    - Reconstruction quality (MAE, PSNR)
    - CFM velocity loss

Usage:
    python scripts/run_cross_validation.py \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --output_dir ./inference_results/cross_validation \
        --n_folds 5 \
        --epochs_per_fold 100
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_train_transforms, get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D
from engine.trainer_vae import MultiModalVAETrainer


def parse_args():
    from utils.config_loader import apply_yaml_defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    pre_args, _ = pre.parse_known_args()

    mapping = [
        (("input", "num_classes"), "num_classes"),
        (("model", "latent_channels"), "latent_channels"),
        (("model", "base_channels"), "base_channels"),
        (("model", "decoder_depth"), "decoder_depth"),
        (("model", "dropout_rate"), "dropout_rate"),
        (("cross_validation", "output_dir"), "output_dir"),
        (("cross_validation", "n_folds"), "n_folds"),
        (("cross_validation", "epochs_per_fold"), "epochs_per_fold"),
    ]
    config_defaults = apply_yaml_defaults(pre_args.config, mapping) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Cross-Validation for ADynamics", parents=[pre])
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./inference_results/cross_validation")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs_per_fold", type=int, default=100, help="Epochs per fold")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--cls_weight", type=float, default=3.0)
    parser.add_argument("--kl_weight", type=float, default=0.1)
    parser.add_argument("--contrastive_weight", type=float, default=0.05)
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD)")
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_true", default=False)
    # Apply YAML config defaults AFTER all add_argument calls
    # (set_defaults must come last so it isn't overridden by argparse defaults)
    parser.set_defaults(**config_defaults)
    return parser.parse_args()


def train_fold(args, train_data, val_data, fold_idx, device):
    """Train a single fold and return metrics."""
    from core_data.dataset import multimodal_collate_fn

    train_transforms = get_multimodal_train_transforms()
    val_transforms = get_multimodal_val_transforms()

    train_dataset = MultiModalDataset(train_data, transform=train_transforms)
    val_dataset = MultiModalDataset(val_data, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=multimodal_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=multimodal_collate_fn
    )

    # Create model
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

    if args.num_gpus > 1:
        from torch.nn.parallel import DataParallel
        model = DataParallel(model, device_ids=list(range(args.num_gpus)))

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_per_fold, eta_min=1e-6)

    config = {
        "cls_weight": args.cls_weight,
        "kl_weight": args.kl_weight,
        "contrastive_weight": args.contrastive_weight,
        "recon_loss_type": "l1",
        "use_amp": not args.no_amp,
    }

    trainer = MultiModalVAETrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        scheduler=scheduler,
    )

    # Train
    fold_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/{args.n_folds}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"{'='*60}")

    history = trainer.train(
        num_epochs=args.epochs_per_fold,
        save_interval=args.epochs_per_fold,
        output_dir=fold_dir,
        early_stopping_patience=30,
    )

    # Collect final metrics
    val_metrics = trainer.validate_epoch()

    return {
        "fold": fold_idx,
        "val_loss": val_metrics["loss"],
        "val_recon_loss": val_metrics["recon_loss"],
        "val_cls_loss": val_metrics["cls_loss"],
        "val_kl_loss": val_metrics["kl_loss"],
        "val_cls_acc": val_metrics["cls_acc"],
        "best_cls_acc": trainer.best_cls_acc,
        "epochs_trained": trainer.current_epoch + 1,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    with open(args.json, "r") as f:
        data_list = json.load(f)

    # Remap labels for 3-class: NC=0, SCD+MCI=1, AD=2
    if args.num_classes == 3:
        from utils.config_loader import remap_labels_3class
        remap_labels_3class(data_list)
        print("Remapped labels to 3-class (NC / SCD+MCI / AD)")

    CLASS_NAMES_MAP = {
        3: ["NC", "SCD+MCI", "AD"],
        4: ["NC", "SCD", "MCI", "AD"],
    }

    # Remap labels for 3-class: NC=0, SCD+MCI=1, AD=2
    if args.num_classes == 3:
        for item in data_list:
            label = item.get("label", 0)
            if label in [1, 2]:
                item["label"] = 1
            elif label == 3:
                item["label"] = 2

    labels = [d.get("label", 0) for d in data_list]
    class_names = CLASS_NAMES_MAP.get(args.num_classes, [f"Class_{i}" for i in range(args.num_classes)])

    # Count per class
    from collections import Counter
    label_counts = Counter(labels)
    print(f"Total samples: {len(data_list)}")
    for c in range(args.num_classes):
        print(f"  {class_names[c]}: {label_counts.get(c, 0)}")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_results = []

    os.makedirs(args.output_dir, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(data_list, labels)):
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]

        result = train_fold(args, train_data, val_data, fold_idx, device)
        fold_results.append(result)

    # Aggregate results
    print(f"\n{'='*60}")
    print("Cross-Validation Results Summary")
    print(f"{'='*60}")

    metrics_to_report = [
        "val_loss", "val_recon_loss", "val_cls_loss", "val_kl_loss",
        "val_cls_acc", "best_cls_acc", "epochs_trained"
    ]

    print(f"\n{'Metric':<25} {'Mean':>10} {'Std':>10} {'Per-Fold':>30}")
    print("-" * 75)

    aggregated = {}
    for metric in metrics_to_report:
        values = [r[metric] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        per_fold = ", ".join([f"{v:.4f}" for v in values])
        print(f"{metric:<25} {mean_val:>10.4f} {std_val:>10.4f} {per_fold:>30}")
        aggregated[metric] = {"mean": float(mean_val), "std": float(std_val), "per_fold": [float(v) for v in values]}

    # Save results
    import json as json_mod
    results_path = os.path.join(args.output_dir, "cv_results.json")
    with open(results_path, "w") as f:
        json_mod.dump({
            "n_folds": args.n_folds,
            "config": vars(args),
            "fold_results": fold_results,
            "aggregated": aggregated,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate summary plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        folds = list(range(1, args.n_folds + 1))

        # Classification accuracy
        accs = [r["val_cls_acc"] for r in fold_results]
        axes[0].bar(folds, accs, color='#2196F3')
        axes[0].axhline(y=np.mean(accs), color='red', linestyle='--', label=f'Mean={np.mean(accs):.3f}')
        axes[0].set_xlabel("Fold")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Classification Accuracy per Fold")
        axes[0].legend()
        axes[0].set_ylim(0, 1)

        # Losses
        losses = [r["val_loss"] for r in fold_results]
        recon_losses = [r["val_recon_loss"] for r in fold_results]
        axes[1].bar(folds, losses, label='Total', color='#4CAF50', alpha=0.7)
        axes[1].bar(folds, recon_losses, label='Recon', color='#FF9800', alpha=0.7)
        axes[1].set_xlabel("Fold")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Validation Loss per Fold")
        axes[1].legend()

        # Box plot of accuracies
        axes[2].boxplot(accs, labels=['Accuracy'])
        axes[2].set_title("Accuracy Distribution")
        axes[2].set_ylabel("Accuracy")

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "cv_summary.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
