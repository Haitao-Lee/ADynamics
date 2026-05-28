"""
Comprehensive Latent Space Analysis for Multi-Modal VAE.

Analyzes the encoder's latent space quality:
- PCA (2D + 3D) visualization
- t-SNE visualization
- Silhouette score (class separation)
- Per-class latent statistics (mean, std, intra-class distance)
- Inter-class vs intra-class distance ratio
- Latent dimension variance analysis

Usage:
    python scripts/run_latent_analysis.py --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt
    python scripts/run_latent_analysis.py --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt --output_dir ./inference_results/latent_analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D

CLASS_NAMES_MAP = {
    3: ["NC", "SCD+MCI", "AD"],
    4: ["NC", "SCD", "MCI", "AD"],
}
COLORS_MAP = {
    3: ['#2ecc71', '#f39c12', '#e74c3c'],
    4: ['#2ecc71', '#3498db', '#f39c12', '#e74c3c'],
}
MARKERS_MAP = {
    3: ['o', '^', 'D'],
    4: ['o', 's', '^', 'D'],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Latent Space Analysis")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/stage1_multimodal/vae_best.pt")
    parser.add_argument("--json", type=str,
                        default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str,
                        default="./inference_results/latent_analysis")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--decoder_depth", type=int, default=4)
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


def load_model(checkpoint_path, device, args):
    model = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=4,
        decoder_depth=args.decoder_depth,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    model_sd = model.state_dict()
    if any(k.startswith("module.") for k in sd) and not any(k.startswith("module.") for k in model_sd):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def encode_dataset(model, dataloader, device, max_samples=500):
    latents = []
    labels = []

    with torch.no_grad():
        total = 0
        for batch in dataloader:
            if total >= max_samples:
                break
            t1 = batch["t1"].to(device)
            x_dict = {"t1": t1}
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    x_dict[mod] = batch[mod].to(device)

            _, _, mu, _ = model(x_dict, return_components=True)
            pooled = torch.nn.functional.adaptive_avg_pool3d(mu, (1, 1, 1)).view(mu.size(0), -1)
            latents.append(pooled.cpu().numpy())
            labels.append(batch["label"].numpy())
            total += t1.shape[0]

    latents = np.concatenate(latents, axis=0)[:max_samples]
    labels = np.concatenate(labels, axis=0)[:max_samples]
    return latents, labels


def plot_pca_2d(latents, labels, output_path):
    pca = PCA(n_components=2, random_state=42)
    latents_2d = pca.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (name, color, marker) in enumerate(zip(CLASS_NAMES, COLORS, MARKERS)):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                      c=color, marker=marker, s=60, alpha=0.6,
                      edgecolors='white', linewidths=0.5,
                      label=f'{name} (n={mask.sum()})')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Latent Space - PCA')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    return pca


def plot_tsne(latents, labels, output_path, perplexity=30):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(latents) - 1))
    latents_2d = tsne.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, (name, color, marker) in enumerate(zip(CLASS_NAMES, COLORS, MARKERS)):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                      c=color, marker=marker, s=60, alpha=0.6,
                      edgecolors='white', linewidths=0.5,
                      label=f'{name} (n={mask.sum()})')

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Latent Space - t-SNE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def compute_class_distances(latents, labels):
    """Compute intra-class and inter-class distances."""
    class_means = {}
    class_stds = {}
    intra_distances = {}

    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        if mask.sum() == 0:
            continue
        class_latents = latents[mask]
        class_means[name] = class_latents.mean(axis=0)
        class_stds[name] = class_latents.std(axis=0)

        dists = np.linalg.norm(class_latents - class_means[name], axis=1)
        intra_distances[name] = float(dists.mean())

    inter_distances = {}
    names = list(class_means.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = np.linalg.norm(class_means[names[i]] - class_means[names[j]])
            inter_distances[f"{names[i]}-{names[j]}"] = float(d)

    return intra_distances, inter_distances, class_means, class_stds


def plot_silhouette(latents, labels, output_path):
    """Plot silhouette analysis per sample."""
    sil_scores = silhouette_samples(latents, labels)
    sil_avg = silhouette_score(latents, labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    y_lower = 10

    for i, name in enumerate(CLASS_NAMES):
        mask = labels == i
        if mask.sum() == 0:
            continue
        class_scores = sil_scores[mask]
        class_scores.sort()

        size = len(class_scores)
        y_upper = y_lower + size

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, class_scores,
                         alpha=0.7, color=COLORS[i])
        ax.text(-0.05, y_lower + 0.5 * size, name, fontweight='bold')
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color='red', linestyle='--', label=f'Average: {sil_avg:.4f}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Samples (sorted by class)')
    ax.set_title(f'Silhouette Analysis (avg={sil_avg:.4f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    return sil_avg


def plot_variance_analysis(latents, labels, output_path):
    """Plot per-dimension variance analysis."""
    overall_var = latents.var(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].bar(range(len(overall_var)), overall_var, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('Variance')
    axes[0].set_title('Per-Dimension Variance (Overall)')
    axes[0].grid(True, alpha=0.3, axis='y')

    x = np.arange(min(32, latents.shape[1]))
    width = 0.2
    for i, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        mask = labels == i
        if mask.sum() == 0:
            continue
        class_mean = latents[mask].mean(axis=0)[:len(x)]
        axes[1].bar(x + i * width, class_mean, width, label=name, color=color, alpha=0.7)

    axes[1].set_xlabel('Latent Dimension')
    axes[1].set_ylabel('Mean Value')
    axes[1].set_title('Per-Class Mean Latent Value (first 32 dims)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    global CLASS_NAMES, COLORS, MARKERS
    args = parse_args()
    CLASS_NAMES = CLASS_NAMES_MAP.get(args.num_classes, [f"Class_{i}" for i in range(args.num_classes)])
    COLORS = COLORS_MAP.get(args.num_classes, ['#2ecc71', '#3498db', '#f39c12', '#e74c3c'][:args.num_classes])
    MARKERS = MARKERS_MAP.get(args.num_classes, ['o', 's', '^', 'D'][:args.num_classes])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device, args)

    print(f"Loading dataset from {args.json}")
    data_list = load_data(args.json)
    print(f"Valid samples: {len(data_list)}")

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
    dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=0, collate_fn=multimodal_collate_fn,
    )

    print("Encoding validation samples...")
    latents, labels = encode_dataset(model, dataloader, device, max_samples=args.num_samples)
    print(f"Encoded {len(latents)} samples, latent dim: {latents.shape[1]}")

    print("\n=== Silhouette Analysis ===")
    sil_path = os.path.join(args.output_dir, "silhouette_analysis.png")
    sil_avg = plot_silhouette(latents, labels, sil_path)

    print("\n=== Class Distance Analysis ===")
    intra, inter, class_means, class_stds = compute_class_distances(latents, labels)

    print("Intra-class distances (mean distance to class center):")
    for name, d in intra.items():
        print(f"  {name}: {d:.4f}")

    print("\nInter-class distances (between class centers):")
    for pair, d in inter.items():
        print(f"  {pair}: {d:.4f}")

    if intra and inter:
        avg_intra = np.mean(list(intra.values()))
        avg_inter = np.mean(list(inter.values()))
        ratio = avg_intra / avg_inter if avg_inter > 0 else float('inf')
        print(f"\nIntra/Inter ratio: {ratio:.4f} (lower = better separation)")

    print("\n=== PCA Visualization ===")
    pca_path = os.path.join(args.output_dir, "latent_pca.png")
    pca = plot_pca_2d(latents, labels, pca_path)

    print("\n=== t-SNE Visualization ===")
    tsne_path = os.path.join(args.output_dir, "latent_tsne.png")
    plot_tsne(latents, labels, tsne_path)

    print("\n=== Variance Analysis ===")
    var_path = os.path.join(args.output_dir, "variance_analysis.png")
    plot_variance_analysis(latents, labels, var_path)

    metrics = {
        "n_samples": len(latents),
        "latent_dim": latents.shape[1],
        "silhouette_score": float(sil_avg),
        "intra_class_distances": intra,
        "inter_class_distances": inter,
        "intra_inter_ratio": float(ratio) if intra and inter else None,
        "pca_explained_variance": pca.explained_variance_ratio_[:10].tolist(),
        "label_counts": {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(args.num_classes)},
        "per_class_mean_norm": {
            name: float(np.linalg.norm(class_means[name]))
            for name in class_means
        },
    }

    with open(os.path.join(args.output_dir, "latent_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("LATENT SPACE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Samples: {len(latents)}, Latent dim: {latents.shape[1]}")
    print(f"Silhouette Score: {sil_avg:.4f}")

    if sil_avg > 0.3:
        print("  -> Good class separation!")
    elif sil_avg > 0.1:
        print("  -> Moderate separation, could be improved")
    elif sil_avg > 0:
        print("  -> Weak separation, encoder needs improvement")
    else:
        print("  -> No separation, encoder does not distinguish classes")

    if intra and inter:
        print(f"Intra/Inter ratio: {ratio:.4f}")
        if ratio < 0.5:
            print("  -> Excellent: classes are well-clustered relative to separation")
        elif ratio < 1.0:
            print("  -> Good: intra-class spread is smaller than inter-class distance")
        else:
            print("  -> Poor: classes overlap more than they separate")

    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - silhouette_analysis.png")
    print(f"  - latent_pca.png")
    print(f"  - latent_tsne.png")
    print(f"  - variance_analysis.png")
    print(f"  - latent_metrics.json")


if __name__ == "__main__":
    main()
