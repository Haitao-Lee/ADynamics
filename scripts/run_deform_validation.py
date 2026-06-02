"""
Stage 4 Deformation Validation.

Validates the deformation generator:
- Jacobian determinant analysis (folding detection)
- Deformation field smoothness
- Visual comparison: original vs warped vs target
- Deformation magnitude statistics

Usage:
    python scripts/run_deform_validation.py \
        --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \
        --deform_checkpoint ./checkpoints/stage4_def/def_best.pt \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --output_dir ./inference_results/deform_validation
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
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D
from models.spatial_transform import (
    DeformationGenerator,
    SpatialTransformer,
    compute_determinant_jacobian,
)


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
        (("deform_validation", "output_dir"), "output_dir"),
        (("deform_validation", "num_samples"), "num_samples"),
    ]
    config_defaults = apply_yaml_defaults(pre_args.config, mapping) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Deformation Validation", parents=[pre])
    parser.set_defaults(**config_defaults)
    parser.add_argument("--encoder_checkpoint", type=str, required=True)
    parser.add_argument("--deform_checkpoint", type=str, required=True)
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./inference_results/deform_validation")
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD)")
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

    # Load encoder
    encoder = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        decoder_depth=args.decoder_depth,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
    )
    ckpt = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    model_sd = encoder.state_dict()
    if any(k.startswith("module.") for k in sd) and not any(k.startswith("module.") for k in model_sd):
        sd = {k[7:]: v for k, v in sd.items()}
    encoder.load_state_dict(sd, strict=False)
    encoder = encoder.to(device).eval()

    # Load deformation generator
    def_gen = DeformationGenerator(
        latent_channels=args.latent_channels,
        latent_spatial=(16, 16, 12),
        output_spatial=MULTI_MODAL_SPATIAL_SIZES["t1"],
        base_channels=16,
    )
    def_ckpt = torch.load(args.deform_checkpoint, map_location=device, weights_only=False)
    def_sd = def_ckpt.get("def_model_state_dict", def_ckpt.get("model_state_dict", def_ckpt))
    if any(k.startswith("module.") for k in def_sd):
        def_sd = {k[7:]: v for k, v in def_sd.items()}
    def_gen.load_state_dict(def_sd, strict=False)
    def_gen = def_gen.to(device).eval()

    stn = SpatialTransformer(mode="bilinear", padding_mode="border")

    # Load data
    data_list = load_data(args.json)
    from sklearn.model_selection import train_test_split
    _, val_data = train_test_split(
        data_list, test_size=0.15,
        stratify=[d.get("label", 0) for d in data_list],
        random_state=42,
    )

    val_transforms = get_multimodal_val_transforms()
    val_dataset = MultiModalDataset(val_data, transform=val_transforms)
    from core_data.dataset import multimodal_collate_fn
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=multimodal_collate_fn,
    )

    # Process samples
    print(f"Processing {args.num_samples} samples...")
    all_jac_stats = []
    all_flow_stats = []
    sample_idx = 0

    with torch.no_grad():
        for batch in val_loader:
            if sample_idx >= args.num_samples:
                break

            t1 = batch["t1"].to(device)
            x_dict = {"t1": t1}
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    x_dict[mod] = batch[mod].to(device)

            labels = batch.get("label", torch.tensor([0]))

            # Encode
            _, _, mu, _ = encoder(x_dict, return_components=True)

            # Generate deformation
            flow = def_gen(mu)

            # Compute Jacobian determinant
            det = compute_determinant_jacobian(flow, spacing=(1.0, 1.0, 1.0))
            det_np = det.cpu().numpy()

            jac_stats = {
                "sample": sample_idx,
                "label": int(labels[0]) if labels.dim() == 1 else int(labels[0][0]),
                "det_mean": float(det_np.mean()),
                "det_min": float(det_np.min()),
                "det_max": float(det_np.max()),
                "n_negative": int((det_np < 0).sum()),
                "n_near_zero": int((det_np < 0.01).sum()),
                "total_voxels": int(det_np.size),
            }
            all_jac_stats.append(jac_stats)

            # Flow statistics
            flow_np = flow.cpu().numpy()
            flow_stats = {
                "sample": sample_idx,
                "flow_mean": float(np.abs(flow_np).mean()),
                "flow_max": float(np.abs(flow_np).max()),
                "flow_std": float(flow_np.std()),
            }
            all_flow_stats.append(flow_stats)

            # Visualize middle slice
            if sample_idx < 5:
                mid_d = t1.shape[2] // 2
                original = t1[0, 0, mid_d].cpu().numpy()

                # Apply deformation
                warped = stn(t1, flow)
                warped_np = warped[0, 0, mid_d].cpu().numpy()

                # Flow magnitude
                flow_mag = torch.norm(flow[0], dim=0)[mid_d].cpu().numpy()

                # Jacobian determinant slice
                det_slice = det_np[0, max(0, mid_d - det_np.shape[1] // 2)]

                fig, axes = plt.subplots(2, 2, figsize=(14, 12))

                im0 = axes[0, 0].imshow(original, cmap='gray')
                axes[0, 0].set_title('Original T1')
                plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

                im1 = axes[0, 1].imshow(warped_np, cmap='gray')
                axes[0, 1].set_title('Warped T1')
                plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

                im2 = axes[1, 0].imshow(flow_mag, cmap='hot')
                axes[1, 0].set_title('Flow Magnitude')
                plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

                im3 = axes[1, 1].imshow(det_slice, cmap='RdBu_r', vmin=-2, vmax=2)
                axes[1, 1].set_title(f'Jacobian Det (neg={jac_stats["n_negative"]})')
                plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

                for ax in axes.flat:
                    ax.axis('off')

                plt.suptitle(f'Sample {sample_idx} (label={jac_stats["label"]})')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"deform_sample_{sample_idx}.png"), dpi=150)
                plt.close()

            sample_idx += 1

    # Summary statistics
    print("\n=== Jacobian Determinant Statistics ===")
    det_means = [s["det_mean"] for s in all_jac_stats]
    det_mins = [s["det_min"] for s in all_jac_stats]
    n_negatives = [s["n_negative"] for s in all_jac_stats]

    print(f"  Det mean: {np.mean(det_means):.4f} +/- {np.std(det_means):.4f}")
    print(f"  Det min:  {np.mean(det_mins):.4f} +/- {np.std(det_mins):.4f}")
    print(f"  Negative voxels: {np.mean(n_negatives):.1f} +/- {np.std(n_negatives):.1f}")

    print("\n=== Flow Magnitude Statistics ===")
    flow_means = [s["flow_mean"] for s in all_flow_stats]
    flow_maxs = [s["flow_max"] for s in all_flow_stats]
    print(f"  Flow mean: {np.mean(flow_means):.4f} +/- {np.std(flow_means):.4f}")
    print(f"  Flow max:  {np.mean(flow_maxs):.4f} +/- {np.std(flow_maxs):.4f}")

    # Save metrics
    import json as json_mod
    metrics = {
        "jacobian_stats": all_jac_stats,
        "flow_stats": all_flow_stats,
        "summary": {
            "det_mean_avg": float(np.mean(det_means)),
            "det_min_avg": float(np.mean(det_mins)),
            "n_negative_avg": float(np.mean(n_negatives)),
            "flow_mean_avg": float(np.mean(flow_means)),
            "flow_max_avg": float(np.mean(flow_maxs)),
        },
    }
    with open(os.path.join(args.output_dir, "deform_metrics.json"), "w") as f:
        json_mod.dump(metrics, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
