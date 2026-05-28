"""
Reconstruction Quality Validation for Multi-Modal VAE.

Computes reconstruction metrics on validation set:
- MAE (Mean Absolute Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity)
- Per-slice visual comparison (original, recon, diff)

Usage:
    python scripts/run_recon_validation.py \
        --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --output_dir ./inference_results/recon_validation \
        --num_samples 10
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


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruction Validation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./inference_results/recon_validation")
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
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


def compute_psnr(img1, img2):
    """Compute PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(img1.max() / np.sqrt(mse))


def compute_ssim(img1, img2, window_size=11):
    """Compute SSIM using sliding window."""
    from scipy.ndimage import uniform_filter
    C1 = (0.01 * (img1.max() - img1.min())) ** 2
    C2 = (0.03 * (img1.max() - img1.min())) ** 2

    mu1 = uniform_filter(img1, size=window_size)
    mu2 = uniform_filter(img2, size=window_size)
    sigma1_sq = uniform_filter(img1 ** 2, size=window_size) - mu1 ** 2
    sigma2_sq = uniform_filter(img2 ** 2, size=window_size) - mu2 ** 2
    sigma12 = uniform_filter(img1 * img2, size=window_size) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


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
    model = model.to(device).eval()

    # Compute metrics
    print(f"Computing reconstruction metrics on {args.num_samples} samples...")
    all_mae = []
    all_psnr = []
    all_ssim = []
    sample_idx = 0

    CLASS_NAMES_MAP = {
        3: ["NC", "SCD+MCI", "AD"],
        4: ["NC", "SCD", "MCI", "AD"],
    }
    CLASS_NAMES = CLASS_NAMES_MAP.get(args.num_classes, [f"Class_{i}" for i in range(args.num_classes)])

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

            recon, _, _, _ = model(x_dict, return_components=True)

            orig_np = t1[0, 0].cpu().numpy()
            recon_np = recon[0, 0].cpu().numpy()

            mae = float(np.abs(orig_np - recon_np).mean())
            psnr = compute_psnr(orig_np, recon_np)
            ssim = compute_ssim(orig_np, recon_np)

            all_mae.append(mae)
            all_psnr.append(psnr)
            all_ssim.append(ssim)

            label = int(labels[0]) if labels.dim() == 1 else int(labels[0][0])

            # Visualize middle slice
            if sample_idx < min(10, args.num_samples):
                mid = orig_np.shape[0] // 2
                diff = np.abs(orig_np[mid] - recon_np[mid])

                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                axes[0].imshow(orig_np[mid], cmap='gray', vmin=0, vmax=1)
                axes[0].set_title(f'Original ({CLASS_NAMES[label]})')
                axes[0].axis('off')

                axes[1].imshow(recon_np[mid], cmap='gray', vmin=0, vmax=1)
                axes[1].set_title(f'Reconstruction (MAE={mae:.4f})')
                axes[1].axis('off')

                im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
                axes[2].set_title(f'Diff (PSNR={psnr:.1f}, SSIM={ssim:.4f})')
                axes[2].axis('off')
                plt.colorbar(im, ax=axes[2], fraction=0.046)

                plt.suptitle(f'Sample {sample_idx}')
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, f"recon_sample_{sample_idx}.png"), dpi=150)
                plt.close()

            sample_idx += 1

    # Summary
    print(f"\n{'='*60}")
    print("RECONSTRUCTION QUALITY SUMMARY")
    print(f"{'='*60}")
    print(f"Samples: {len(all_mae)}")
    print(f"MAE:  {np.mean(all_mae):.6f} +/- {np.std(all_mae):.6f}")
    print(f"PSNR: {np.mean(all_psnr):.2f} +/- {np.std(all_psnr):.2f} dB")
    print(f"SSIM: {np.mean(all_ssim):.4f} +/- {np.std(all_ssim):.4f}")

    # Save metrics
    metrics = {
        "n_samples": len(all_mae),
        "mae_mean": float(np.mean(all_mae)),
        "mae_std": float(np.std(all_mae)),
        "psnr_mean": float(np.mean(all_psnr)),
        "psnr_std": float(np.std(all_psnr)),
        "ssim_mean": float(np.mean(all_ssim)),
        "ssim_std": float(np.std(all_ssim)),
        "per_sample": [
            {"mae": m, "psnr": p, "ssim": s}
            for m, p, s in zip(all_mae, all_psnr, all_ssim)
        ],
    }
    with open(os.path.join(args.output_dir, "recon_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
