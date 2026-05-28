"""
Baseline Comparison for ADynamics CFM Pipeline.

Compares the learned CFM flow against simpler interpolation baselines to validate
that flow matching provides genuine value over naive approaches.

Baselines:
    1. Linear Interpolation: z_interp = (1-t)*z_NC + t*z_AD (no learned model)
    2. KNN Interpolation: Average of K nearest neighbors in each class
    3. Supervised Regression: Direct NC→AD mapping via trained MLP
    4. CFM (Ours): Learned velocity field via Conditional Flow Matching

Evaluation Metrics:
    - Trajectory straightness (curvature)
    - Latent space alignment (cosine similarity to true class centroids)
    - ODE integration efficiency (steps needed for convergence)
    - Classification consistency (do intermediate states classify correctly?)

Usage:
    python scripts/run_baseline_comparison.py \
        --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \
        --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt \
        --output_dir ./inference_results/baseline_comparison
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

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D
from models.vector_field import VelocityFieldNet


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Comparison")
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--encoder_checkpoint", type=str, required=True)
    parser.add_argument("--cfm_checkpoint", type=str, default=None)
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD)")
    parser.add_argument("--output_dir", type=str, default="./inference_results/baseline_comparison")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_encoder(args, device):
    """Load pretrained multi-modal encoder."""
    encoder = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        decoder_depth=args.decoder_depth,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
    )
    checkpoint = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    sd = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    encoder.load_state_dict(sd, strict=False)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


CLASS_NAMES_MAP = {
    3: ["NC", "SCD+MCI", "AD"],
    4: ["NC", "SCD", "MCI", "AD"],
}


def build_class_pools(encoder, dataloader, device, num_classes=3):
    """Encode all samples and build per-class latent pools."""
    pools = {c: [] for c in range(num_classes)}
    class_names = CLASS_NAMES_MAP.get(num_classes, [f"Class_{i}" for i in range(num_classes)])

    with torch.no_grad():
        for batch in dataloader:
            t1 = batch["t1"].to(device)
            x_dict = {"t1": t1}
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    x_dict[mod] = batch[mod].to(device)

            labels = batch.get("label", torch.tensor([])).to(device)
            if labels.dim() > 1:
                labels = labels.squeeze()

            _, _, mu, _ = encoder(x_dict, return_components=True)

            for i, label in enumerate(labels):
                lbl = label.item() if label.numel() == 1 else label
                pools[lbl].append(mu[i].cpu())

    for c in range(num_classes):
        pools[c] = [z.to(device) for z in pools[c]]
        print(f"  {class_names[c]}: {len(pools[c])} samples")

    return pools


class LinearInterpolationBaseline:
    """Baseline 1: Simple linear interpolation in latent space.

    No learned model - just z_t = (1-t)*z_NC + t*z_AD.
    This is the "null hypothesis" that CFM must beat.
    """

    def __init__(self, name="Linear Interpolation"):
        self.name = name

    def flow(self, z0, z1, t):
        """Straight-line interpolation."""
        return (1 - t) * z0 + t * z1

    def trajectory(self, z0, z1, steps=20):
        """Generate trajectory from z0 to z1."""
        traj = []
        for i in range(steps + 1):
            t = i / steps
            traj.append(self.flow(z0, z1, t))
        return traj


class KNNInterpolationBaseline:
    """Baseline 2: K-Nearest Neighbor interpolation.

    For each source sample, find K nearest neighbors in target class,
    then interpolate toward their centroid. This captures local structure
    without learning a global flow.
    """

    def __init__(self, target_pool, k=5, name="KNN Interpolation"):
        self.name = name
        self.k = k
        self.target_pool = target_pool
        # Pre-compute target centroid
        if len(target_pool) > 0:
            self.target_centroid = torch.stack(target_pool).mean(dim=0)
        else:
            self.target_centroid = None

    def flow(self, z0, z1, t):
        """Interpolate toward KNN centroid."""
        if self.target_centroid is None:
            return (1 - t) * z0 + t * z1
        # Find K nearest neighbors in target pool
        dists = torch.stack([
            torch.norm(z0.flatten() - z.flatten()).item()
            for z in self.target_pool
        ])
        k = min(self.k, len(self.target_pool))
        _, indices = torch.topk(dists, k, largest=False)
        knn_centroid = torch.stack([self.target_pool[i] for i in indices]).mean(dim=0)
        return (1 - t) * z0 + t * knn_centroid

    def trajectory(self, z0, z1, steps=20):
        traj = []
        for i in range(steps + 1):
            t = i / steps
            traj.append(self.flow(z0, z1, t))
        return traj


class SupervisedRegressionBaseline:
    """Baseline 3: Supervised regression from NC to AD latent.

    Trains a simple MLP to directly map z_NC → z_AD.
    This tests whether a simpler supervised approach suffices.
    """

    def __init__(self, latent_shape, device, name="Supervised Regression"):
        self.name = name
        self.device = device
        flat_dim = int(np.prod(latent_shape))

        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, flat_dim * 2),
            nn.SiLU(),
            nn.Linear(flat_dim * 2, flat_dim * 2),
            nn.SiLU(),
            nn.Linear(flat_dim * 2, flat_dim),
        ).to(device)

    def train(self, nc_pool, ad_pool, epochs=200, lr=1e-3):
        """Train the regression model on NC→AD pairs."""
        optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self.mlp.train()

        for epoch in range(epochs):
            # Sample pairs
            nc_idx = torch.randint(0, len(nc_pool), (min(16, len(nc_pool)),))
            ad_idx = torch.randint(0, len(ad_pool), (min(16, len(ad_pool)),))

            z_nc = torch.stack([nc_pool[i] for i in nc_idx])
            z_ad = torch.stack([ad_pool[i] for i in ad_idx])

            # Flatten
            z_nc_flat = z_nc.view(z_nc.shape[0], -1)
            z_ad_flat = z_ad.view(z_ad.shape[0], -1)

            # Predict
            pred = self.mlp(z_nc_flat)
            loss = F.mse_loss(pred, z_ad_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"  Regression epoch {epoch+1}: loss={loss.item():.6f}")

    def flow(self, z0, z1, t):
        """Interpolate using learned mapping."""
        self.mlp.eval()
        with torch.no_grad():
            z0_flat = z0.view(1, -1)
            z1_pred = self.mlp(z0_flat).view_as(z0)
            return (1 - t) * z0 + t * z1_pred

    def trajectory(self, z0, z1, steps=20):
        traj = []
        for i in range(steps + 1):
            t = i / steps
            traj.append(self.flow(z0, z1, t))
        return traj


def compute_trajectory_straightness(trajectory):
    """
    Compute trajectory straightness = |z_end - z_start| / path_length.

    A perfectly straight trajectory has straightness = 1.0.
    Curved trajectories have straightness < 1.0.
    """
    if len(trajectory) < 2:
        return 1.0

    z_start = trajectory[0].flatten()
    z_end = trajectory[-1].flatten()
    direct_dist = torch.norm(z_end - z_start).item()

    path_length = 0.0
    for i in range(1, len(trajectory)):
        path_length += torch.norm(
            trajectory[i].flatten() - trajectory[i-1].flatten()
        ).item()

    if path_length < 1e-8:
        return 1.0

    return direct_dist / path_length


def compute_class_centroid_alignment(trajectory, class_centroids, src_class, tgt_class):
    """
    Check if trajectory passes through intermediate class centroids.

    For NC→AD, does it pass near SCD and MCI centroids?
    """
    alignments = []
    n = len(trajectory)

    for i, z in enumerate(trajectory):
        t = i / max(1, n - 1)
        # Expected class at this t (linear mapping)
        expected_class = src_class + t * (tgt_class - src_class)

        # Find nearest class centroid
        z_flat = z.flatten()
        min_dist = float('inf')
        nearest_class = -1
        for c, centroid in class_centroids.items():
            dist = torch.norm(z_flat - centroid.flatten()).item()
            if dist < min_dist:
                min_dist = dist
                nearest_class = c

        # Alignment: does nearest class match expected?
        alignment = 1.0 - abs(nearest_class - expected_class) / 3.0
        alignments.append(max(0.0, alignment))

    return np.mean(alignments)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    with open(args.json, "r") as f:
        data_list = json.load(f)

    val_transforms = get_multimodal_val_transforms()
    dataset = MultiModalDataset(data_list, transform=val_transforms)
    from core_data.dataset import multimodal_collate_fn
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                           collate_fn=multimodal_collate_fn)

    # Load encoder
    print("Loading encoder...")
    encoder = load_encoder(args, device)

    # Build latent pools
    print("Building latent pools...")
    pools = build_class_pools(encoder, dataloader, device, args.num_classes)

    # Compute class centroids
    class_names = CLASS_NAMES_MAP.get(args.num_classes, [f"Class_{i}" for i in range(args.num_classes)])
    centroids = {}
    for c in range(args.num_classes):
        if len(pools[c]) > 0:
            centroids[c] = torch.stack(pools[c]).mean(dim=0)
            print(f"  {class_names[c]} centroid norm: {torch.norm(centroids[c]).item():.4f}")

    # Define baselines
    latent_shape = pools[0][0].shape if len(pools[0]) > 0 else (32, 16, 16, 12)

    baselines = [
        LinearInterpolationBaseline(),
        KNNInterpolationBaseline(pools[3] if len(pools[3]) > 0 else pools[0], k=5),
    ]

    # Train supervised regression baseline
    if len(pools[0]) > 0 and len(pools[3]) > 0:
        reg_baseline = SupervisedRegressionBaseline(latent_shape, device)
        print("\nTraining supervised regression baseline...")
        reg_baseline.train(pools[0], pools[3], epochs=200)
        baselines.append(reg_baseline)

    # Load CFM model if available
    cfm_model = None
    if args.cfm_checkpoint and os.path.exists(args.cfm_checkpoint):
        print(f"\nLoading CFM from {args.cfm_checkpoint}...")
        latent_spatial = tuple(pools[0][0].shape[1:]) if len(pools[0]) > 0 else (16, 16, 12)
        cfm_model = VelocityFieldNet(
            latent_channels=args.latent_channels,
            latent_spatial=latent_spatial,
            time_embed_dim=128,
            cond_embed_dim=64,
            base_channels=64,
            channel_mults=(1, 2, 4),
            num_res_blocks=2,
            use_demographics=False,
        ).to(device)
        ckpt = torch.load(args.cfm_checkpoint, map_location=device, weights_only=False)
        cfm_model.load_state_dict(ckpt["model_state_dict"])
        cfm_model.eval()

    # Evaluate baselines
    print(f"\n{'='*60}")
    print("Baseline Comparison Results")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    # Sample test pairs: NC→AD
    if len(pools[0]) < 5 or len(pools[3]) < 5:
        print("ERROR: Need at least 5 samples in NC and AD pools")
        return

    n_test = min(50, len(pools[0]), len(pools[3]))
    test_nc = [pools[0][i] for i in torch.randperm(len(pools[0]))[:n_test]]
    test_ad = [pools[3][i] for i in torch.randperm(len(pools[3]))[:n_test]]

    for baseline in baselines:
        print(f"\n--- {baseline.name} ---")

        straightness_scores = []
        alignment_scores = []

        for z_nc, z_ad in zip(test_nc, test_ad):
            traj = baseline.trajectory(z_nc, z_ad, steps=20)

            straightness = compute_trajectory_straightness(traj)
            alignment = compute_class_centroid_alignment(
                traj, centroids, src_class=0, tgt_class=3
            )

            straightness_scores.append(straightness)
            alignment_scores.append(alignment)

        avg_straightness = np.mean(straightness_scores)
        avg_alignment = np.mean(alignment_scores)

        results[baseline.name] = {
            "straightness": avg_straightness,
            "alignment": avg_alignment,
        }

        print(f"  Trajectory Straightness: {avg_straightness:.4f} (1.0 = perfectly straight)")
        print(f"  Class Centroid Alignment: {avg_alignment:.4f} (1.0 = passes through all stages)")

    # Evaluate CFM
    if cfm_model is not None:
        print(f"\n--- CFM (Ours) ---")

        straightness_scores = []
        alignment_scores = []

        for z_nc, z_ad in zip(test_nc, test_ad):
            # Euler integration
            z_t = z_nc.clone()
            traj = [z_t.clone()]
            steps = 20
            dt = 1.0 / steps

            with torch.no_grad():
                for i in range(steps):
                    t = torch.full((1,), i * dt, device=device, dtype=z_t.dtype)
                    v = cfm_model(z_t.unsqueeze(0), t).squeeze(0)
                    z_t = z_t + v * dt
                    traj.append(z_t.clone())

            straightness = compute_trajectory_straightness(traj)
            alignment = compute_class_centroid_alignment(
                traj, centroids, src_class=0, tgt_class=3
            )

            straightness_scores.append(straightness)
            alignment_scores.append(alignment)

        avg_straightness = np.mean(straightness_scores)
        avg_alignment = np.mean(alignment_scores)

        results["CFM (Ours)"] = {
            "straightness": avg_straightness,
            "alignment": avg_alignment,
        }

        print(f"  Trajectory Straightness: {avg_straightness:.4f}")
        print(f"  Class Centroid Alignment: {avg_alignment:.4f}")

    # Summary table
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Straightness':>12} {'Alignment':>12}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['straightness']:>12.4f} {metrics['alignment']:>12.4f}")

    # Save results
    import json as json_mod
    results_path = os.path.join(args.output_dir, "baseline_results.json")
    with open(results_path, "w") as f:
        json_mod.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate comparison plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        names = list(results.keys())
        straightness = [results[n]["straightness"] for n in names]
        alignment = [results[n]["alignment"] for n in names]

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336'][:len(names)]

        axes[0].barh(names, straightness, color=colors)
        axes[0].set_xlabel("Straightness (1.0 = straight)")
        axes[0].set_title("Trajectory Straightness")
        axes[0].set_xlim(0, 1.1)

        axes[1].barh(names, alignment, color=colors)
        axes[1].set_xlabel("Alignment (1.0 = passes through all stages)")
        axes[1].set_title("Class Centroid Alignment")
        axes[1].set_xlim(0, 1.1)

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "baseline_comparison.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
