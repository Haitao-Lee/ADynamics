"""
Ablation Experiment Framework for ADynamics.

Runs systematic ablation studies to quantify the contribution of each component:
    1. KL loss on/off
    2. Multi-modal vs T1-only
    3. Contrastive loss on/off
    4. CFM direction constraint (forward-only vs bidirectional)
    5. Rectified flow regularization on/off
    6. Deformation generator on/off

Each ablation changes ONE variable while keeping others at their best known values.

Usage:
    # Run all ablations
    python scripts/run_ablation.py --json ./core_data/dataset_manifest_merged_v2.json

    # Run specific ablation
    python scripts/run_ablation.py --ablation kl_weight --json ./core_data/dataset_manifest_merged_v2.json
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import json
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    # Stage 1 parameters
    cls_weight: float = 3.0
    kl_weight: float = 0.1
    contrastive_weight: float = 0.05
    base_channels: int = 32
    # Stage 3 parameters
    cfm_forward_only: bool = True
    cfm_rectified_flow: bool = False
    # Training
    epochs: int = 50  # Shorter for ablation
    batch_size: int = 2
    learning_rate: float = 0.00005


# Define ablation experiments
ABLATIONS = {
    "kl_weight": [
        AblationConfig("no_kl", "KL weight = 0 (no latent regularization)", kl_weight=0.0),
        AblationConfig("kl_low", "KL weight = 0.01", kl_weight=0.01),
        AblationConfig("kl_medium", "KL weight = 0.1 (default)", kl_weight=0.1),
        AblationConfig("kl_high", "KL weight = 0.5", kl_weight=0.5),
    ],
    "contrastive": [
        AblationConfig("no_contrastive", "No contrastive loss", contrastive_weight=0.0),
        AblationConfig("contrastive_low", "Contrastive weight = 0.01", contrastive_weight=0.01),
        AblationConfig("contrastive_medium", "Contrastive weight = 0.05 (default)", contrastive_weight=0.05),
        AblationConfig("contrastive_high", "Contrastive weight = 0.2", contrastive_weight=0.2),
    ],
    "cls_weight": [
        AblationConfig("cls_low", "Classification weight = 1.0", cls_weight=1.0),
        AblationConfig("cls_medium", "Classification weight = 2.0", cls_weight=2.0),
        AblationConfig("cls_high", "Classification weight = 3.0 (default)", cls_weight=3.0),
        AblationConfig("cls_very_high", "Classification weight = 5.0", cls_weight=5.0),
    ],
    "encoder_capacity": [
        AblationConfig("small_encoder", "Base channels = 16", base_channels=16),
        AblationConfig("medium_encoder", "Base channels = 32 (default)", base_channels=32),
        AblationConfig("large_encoder", "Base channels = 48", base_channels=48),
    ],
    "cfm_direction": [
        AblationConfig("bidirectional", "CFM with random direction pairs", cfm_forward_only=False),
        AblationConfig("forward_only", "CFM with forward-only pairs (default)", cfm_forward_only=True),
    ],
    "rectified_flow": [
        AblationConfig("no_rectified", "Standard CFM (no rectification)", cfm_rectified_flow=False),
        AblationConfig("with_rectified", "CFM with rectified flow regularization", cfm_rectified_flow=True),
    ],
}


def run_stage1_ablation(config: AblationConfig, args, device):
    """Run Stage 1 training with given ablation config."""
    from core_data.dataset import MultiModalDataset, multimodal_collate_fn
    from core_data.transforms import get_multimodal_train_transforms, get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
    from models.vae3d import MultiModalVAE3D
    from engine.trainer_vae import MultiModalVAETrainer
    from sklearn.model_selection import train_test_split
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    with open(args.json, "r") as f:
        data_list = json.load(f)

    train_transforms = get_multimodal_train_transforms()
    val_transforms = get_multimodal_val_transforms()

    train_data, val_data = train_test_split(
        data_list, test_size=0.15,
        stratify=[d.get("label", 0) for d in data_list],
        random_state=42
    )

    train_dataset = MultiModalDataset(train_data, transform=train_transforms)
    val_dataset = MultiModalDataset(val_data, transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, collate_fn=multimodal_collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, collate_fn=multimodal_collate_fn
    )

    model = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=config.base_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        decoder_depth=args.decoder_depth,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
    )

    if args.num_gpus > 1:
        from torch.nn.parallel import DataParallel
        model = DataParallel(model, device_ids=list(range(args.num_gpus)))

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    train_config = {
        "cls_weight": config.cls_weight,
        "kl_weight": config.kl_weight,
        "contrastive_weight": config.contrastive_weight,
        "recon_loss_type": "l1",
        "use_amp": not args.no_amp,
    }

    trainer = MultiModalVAETrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=train_config,
        scheduler=scheduler,
    )

    output_dir = os.path.join(args.output_dir, config.name)
    os.makedirs(output_dir, exist_ok=True)

    history = trainer.train(
        num_epochs=config.epochs,
        save_interval=config.epochs,
        output_dir=output_dir,
        early_stopping_patience=20,
    )

    val_metrics = trainer.validate_epoch()

    return {
        "name": config.name,
        "description": config.description,
        "val_loss": val_metrics["loss"],
        "val_recon_loss": val_metrics["recon_loss"],
        "val_cls_loss": val_metrics["cls_loss"],
        "val_kl_loss": val_metrics["kl_loss"],
        "val_cls_acc": val_metrics["cls_acc"],
        "best_cls_acc": trainer.best_cls_acc,
        "epochs_trained": trainer.current_epoch + 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation Experiments")
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./inference_results/ablation")
    parser.add_argument("--ablation", type=str, default="all",
                        help="Ablation to run: all, kl_weight, contrastive, cls_weight, encoder_capacity, cfm_direction, rectified_flow")
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD)")
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_amp", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which ablations to run
    if args.ablation == "all":
        ablation_keys = list(ABLATIONS.keys())
    else:
        ablation_keys = [args.ablation]

    all_results = {}

    for ablation_key in ablation_keys:
        if ablation_key not in ABLATIONS:
            print(f"Unknown ablation: {ablation_key}")
            print(f"Available: {list(ABLATIONS.keys())}")
            continue

        configs = ABLATIONS[ablation_key]
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation_key}")
        print(f"{'='*60}")

        ablation_results = []
        for config in configs:
            print(f"\n--- {config.name}: {config.description} ---")
            result = run_stage1_ablation(config, args, device)
            ablation_results.append(result)
            print(f"  Result: val_cls_acc={result['val_cls_acc']:.4f}, best_cls_acc={result['best_cls_acc']:.4f}")

        all_results[ablation_key] = ablation_results

    # Summary
    print(f"\n{'='*60}")
    print("Ablation Summary")
    print(f"{'='*60}")

    for ablation_key, results in all_results.items():
        print(f"\n--- {ablation_key} ---")
        print(f"{'Config':<25} {'Description':<40} {'Val Acc':>10} {'Best Acc':>10}")
        print("-" * 85)
        for r in results:
            print(f"{r['name']:<25} {r['description']:<40} {r['val_cls_acc']:>10.4f} {r['best_cls_acc']:>10.4f}")

    # Save results
    import json as json_mod
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json_mod.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate comparison plots
    try:
        import matplotlib.pyplot as plt

        n_ablations = len(all_results)
        if n_ablations == 0:
            return

        fig, axes = plt.subplots(1, n_ablations, figsize=(5 * n_ablations, 5))
        if n_ablations == 1:
            axes = [axes]

        for idx, (ablation_key, results) in enumerate(all_results.items()):
            names = [r["name"] for r in results]
            accs = [r["best_cls_acc"] for r in results]

            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(accs)))
            bars = axes[idx].bar(range(len(names)), accs, color=colors)
            axes[idx].set_xticks(range(len(names)))
            axes[idx].set_xticklabels(names, rotation=45, ha='right')
            axes[idx].set_ylabel("Best Classification Accuracy")
            axes[idx].set_title(f"Ablation: {ablation_key}")
            axes[idx].set_ylim(0, 1)

            # Add value labels
            for bar, acc in zip(bars, accs):
                axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                             f'{acc:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "ablation_summary.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
