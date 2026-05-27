"""
Stage 2a Classification Validation.

Loads trained classifier checkpoint, computes per-class accuracy,
confusion matrix, and classification report on validation set.

Usage:
    python scripts/run_cls_validation.py \
        --checkpoint ./checkpoints/stage2_classifier/classifier_best.pt \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --output_dir ./inference_results/cls_validation
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
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D


CLASS_NAMES = ["NC", "SCD", "MCI", "AD"]


def parse_args():
    parser = argparse.ArgumentParser(description="Classification Validation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./inference_results/cls_validation")
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_data(json_path):
    import nibabel as nib
    from monai.transforms import LoadImaged, EnsureChannelFirstd, Orientationd, Compose

    with open(json_path, "r") as f:
        data = json.load(f)

    quick = Compose([
        LoadImaged(keys=['t1'], reader='NibabelReader'),
        EnsureChannelFirstd(keys=['t1']),
        Orientationd(keys=['t1'], axcodes='RAS'),
    ])

    valid = []
    for item in data:
        t1_path = item.get("t1")
        if not t1_path or not os.path.exists(t1_path):
            continue
        try:
            img = nib.load(t1_path)
            if any(s == 0 for s in img.shape):
                continue
            if img.get_fdata().min() == img.get_fdata().max():
                continue
            quick({'t1': str(t1_path)})
        except Exception:
            continue
        valid.append(item)
    return valid


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data_list = load_data(args.json)
    from sklearn.model_selection import train_test_split
    _, val_data = train_test_split(
        data_list, test_size=0.15,
        stratify=[d.get("label", 0) for d in data_list],
        random_state=42,
    )
    print(f"Validation samples: {len(val_data)}")

    val_transforms = get_multimodal_val_transforms()
    val_dataset = MultiModalDataset(val_data, transform=val_transforms)
    from core_data.dataset import multimodal_collate_fn
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=multimodal_collate_fn,
    )

    # Load model
    model = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        decoder_depth=args.decoder_depth,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    model_sd = model.state_dict()
    if any(k.startswith("module.") for k in sd) and not any(k.startswith("module.") for k in model_sd):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            t1 = batch["t1"].to(device)
            x_dict = {"t1": t1}
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    x_dict[mod] = batch[mod].to(device)

            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(device).squeeze()

            _, cls_logits, _, _ = model(x_dict, return_components=True)
            probs = F.softmax(cls_logits, dim=1)
            preds = cls_logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    print("\nClassification Report:")
    print(report)

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"Saved: {args.output_dir}/confusion_matrix.png")

    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = all_labels == i
        if mask.sum() > 0:
            acc = (all_preds[mask] == i).mean()
            per_class_acc[name] = float(acc)
            print(f"  {name}: {acc:.4f} ({mask.sum()} samples)")

    # Save metrics
    import json as json_mod
    metrics = {
        "overall_acc": float((all_preds == all_labels).mean()),
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(all_labels),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json_mod.dump(metrics, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
