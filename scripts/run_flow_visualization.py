"""
Stage 3 CFM Flow Visualization.

Visualizes disease progression trajectories in latent space:
- NC ->?AD trajectory via ODE integration
- Velocity field magnitude over time
- Latent trajectory PCA projection
- Trajectory smoothness metrics

Usage:
    python scripts/run_flow_visualization.py \
        --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \
        --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --output_dir ./inference_results/flow_vis
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
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D
from models.vector_field import VelocityFieldNet

CLASS_NAMES_MAP = {
    3: ["NC", "SCD+MCI", "AD"],
    4: ["NC", "SCD", "MCI", "AD"],
}


def parse_args():
    from utils.config_loader import apply_yaml_defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    pre_args, _ = pre.parse_known_args()

    mapping = [
        (("input", "encoder_checkpoint"), "encoder_checkpoint"),
        (("input", "num_classes"), "num_classes"),
        (("model", "latent_channels"), "latent_channels"),
        (("model", "base_channels"), "base_channels"),
        (("model", "decoder_depth"), "decoder_depth"),
        (("model", "cfm_base_channels"), "cfm_base_channels"),
        (("model", "time_embed_dim"), "time_embed_dim"),
        (("model", "cond_embed_dim"), "cond_embed_dim"),
        (("flow_visualization", "output_dir"), "output_dir"),
        (("flow_visualization", "ode_steps"), "ode_steps"),
        (("flow_visualization", "num_samples"), "num_samples"),
    ]
    config_defaults = apply_yaml_defaults(pre_args.config, mapping) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="CFM Flow Visualization", parents=[pre])
    parser.add_argument("--encoder_checkpoint", type=str, required=True)
    parser.add_argument("--cfm_checkpoint", type=str, required=True)
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./inference_results/flow_vis")
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--cfm_base_channels", type=int, default=64)
    parser.add_argument("--time_embed_dim", type=int, default=128)
    parser.add_argument("--ode_steps", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD)")
    parser.add_argument("--device", type=str, default="cuda")
    # Apply YAML config defaults AFTER all add_argument calls
    # (set_defaults must come last so it isn't overridden by argparse defaults)
    parser.set_defaults(**config_defaults)
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


@torch.no_grad()
def integrate_ode(z0, cfm_model, steps=20):
    """Euler ODE integration: z0 ->?z1."""
    z_t = z0.clone()
    dt = 1.0 / steps
    trajectory = [z_t.clone().cpu()]
    velocities = []

    for i in range(steps):
        t = torch.full((z0.shape[0],), i * dt, device=z0.device)
        v_t = cfm_model(z_t, t)
        z_t = z_t + v_t * dt
        trajectory.append(z_t.clone().cpu())
        velocities.append(v_t.norm().item())

    return z_t, trajectory, velocities


def main():
    global CLASS_NAMES
    args = parse_args()
    CLASS_NAMES = CLASS_NAMES_MAP.get(args.num_classes, [f"Class_{i}" for i in range(args.num_classes)])
    ad_label = args.num_classes - 1  # AD is the last class

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

    # Load CFM
    cfm = VelocityFieldNet(
        latent_channels=args.latent_channels,
        latent_spatial=(16, 16, 12),
        time_embed_dim=args.time_embed_dim,
        base_channels=args.cfm_base_channels,
    )
    cfm_ckpt = torch.load(args.cfm_checkpoint, map_location=device, weights_only=False)
    cfm_sd = cfm_ckpt.get("model_state_dict", cfm_ckpt)
    if any(k.startswith("module.") for k in cfm_sd):
        cfm_sd = {k[7:]: v for k, v in cfm_sd.items()}
    cfm.load_state_dict(cfm_sd, strict=False)
    cfm = cfm.to(device).eval()

    # Load data
    data_list = load_data(args.json)

    # Remap labels for 3-class: NC=0, SCD+MCI=1, AD=2
    if args.num_classes == 3:
        from utils.config_loader import remap_labels_3class
        remap_labels_3class(data_list)
        print("Remapped labels to 3-class (NC / SCD+MCI / AD)")

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

    # Encode all samples
    print("Encoding validation samples...")
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if len(all_latents) >= args.num_samples:
                break
            t1 = batch["t1"].to(device)
            x_dict = {"t1": t1}
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    x_dict[mod] = batch[mod].to(device)

            _, _, mu, _ = encoder(x_dict, return_components=True)
            pooled = torch.nn.functional.adaptive_avg_pool3d(mu, (1, 1, 1)).view(mu.size(0), -1)
            all_latents.append(pooled.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    all_latents = np.concatenate(all_latents, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Remap labels for 3-class: NC=0, SCD+MCI=1, AD=2
    if args.num_classes == 3:
        all_labels = np.where(np.isin(all_labels, [1, 2]), 1, all_labels)
        all_labels = np.where(all_labels == 3, 2, all_labels)

    # Sample NC sources and run ODE
    nc_mask = all_labels == 0
    ad_mask = all_labels == ad_label

    if nc_mask.sum() == 0:
        print("ERROR: No NC samples found!")
        return

    nc_latents = torch.from_numpy(all_latents[nc_mask][:min(10, nc_mask.sum())]).float().to(device)

    print(f"Running ODE integration ({args.ode_steps} steps)...")
    z_final, trajectory, velocities = integrate_ode(nc_latents, cfm, steps=args.ode_steps)

    # 1. Velocity magnitude over time
    fig, ax = plt.subplots(figsize=(10, 5))
    time_points = np.linspace(0, 1, len(velocities))
    ax.plot(time_points, velocities, 'b-o', linewidth=2)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Velocity Magnitude ||v||')
    ax.set_title('CFM Velocity Magnitude Over Time')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "velocity_magnitude.png"), dpi=150)
    plt.close()
    print(f"Saved: velocity_magnitude.png")

    # 2. Trajectory smoothness
    traj_diffs = []
    for i in range(1, len(trajectory)):
        diff = (trajectory[i] - trajectory[i-1]).norm().item()
        traj_diffs.append(diff)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(traj_diffs)), traj_diffs, 'r-o', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('||z_{t+1} - z_t||')
    ax.set_title('Trajectory Step Size (Smoothness)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "trajectory_smoothness.png"), dpi=150)
    plt.close()
    print(f"Saved: trajectory_smoothness.png")

    # 3. PCA projection of trajectories
    # Combine all latents + trajectory points for PCA
    traj_flat = []
    for z in trajectory:
        pooled = torch.nn.functional.adaptive_avg_pool3d(z, (1, 1, 1)).view(z.size(0), -1)
        traj_flat.append(pooled.cpu().numpy())
    traj_flat = np.concatenate(traj_flat, axis=0)

    all_points = np.concatenate([all_latents, traj_flat], axis=0)
    pca = PCA(n_components=2, random_state=42)
    all_2d = pca.fit_transform(all_points)

    n_orig = len(all_latents)
    orig_2d = all_2d[:n_orig]
    traj_2d = all_2d[n_orig:]

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    markers = ['o', 's', '^', 'D']

    for i, name in enumerate(CLASS_NAMES):
        mask = all_labels == i
        if mask.sum() > 0:
            ax.scatter(orig_2d[mask, 0], orig_2d[mask, 1],
                      c=colors[i], marker=markers[i], s=60, alpha=0.5,
                      label=f'{name} (n={mask.sum()})')

    # Plot trajectories
    n_traj = len(nc_latents)
    for j in range(min(5, n_traj)):
        traj_j = traj_2d[j::n_traj]
        ax.plot(traj_j[:, 0], traj_j[:, 1], 'k-', alpha=0.3, linewidth=1)
        ax.scatter(traj_j[0, 0], traj_j[0, 1], c='black', marker='x', s=100, linewidths=3)
        ax.scatter(traj_j[-1, 0], traj_j[-1, 1], c='red', marker='*', s=200, linewidths=3)

    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('CFM Disease Progression Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "flow_trajectories.png"), dpi=150)
    plt.close()
    print(f"Saved: flow_trajectories.png")

    # Save metrics
    import json as json_mod

    # Compute trajectory straightness
    straightness_scores = []
    for j in range(len(nc_latents)):
        z_start = trajectory[0][j].flatten()
        z_end = trajectory[-1][j].flatten()
        direct_dist = torch.norm(z_end - z_start).item()

        path_length = 0.0
        for i in range(1, len(trajectory)):
            path_length += torch.norm(
                trajectory[i][j].flatten() - trajectory[i-1][j].flatten()
            ).item()

        if path_length > 1e-8:
            straightness_scores.append(direct_dist / path_length)

    # Compute class centroid distances
    class_centroids = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = all_labels == i
        if mask.sum() > 0:
            class_centroids[name] = float(np.mean(all_latents[mask]))

    metrics = {
        "ode_steps": args.ode_steps,
        "velocity_mean": float(np.mean(velocities)),
        "velocity_std": float(np.std(velocities)),
        "trajectory_smoothness_mean": float(np.mean(traj_diffs)),
        "trajectory_smoothness_std": float(np.std(traj_diffs)),
        "trajectory_straightness_mean": float(np.mean(straightness_scores)) if straightness_scores else 0.0,
        "trajectory_straightness_std": float(np.std(straightness_scores)) if straightness_scores else 0.0,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "n_nc_sources": len(nc_latents),
    }
    with open(os.path.join(args.output_dir, "flow_metrics.json"), "w") as f:
        json_mod.dump(metrics, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("Flow Visualization Summary")
    print(f"{'='*60}")
    print(f"ODE Steps: {args.ode_steps}")
    print(f"NC Sources: {len(nc_latents)}")
    print(f"Velocity: {metrics['velocity_mean']:.4f} +/- {metrics['velocity_std']:.4f}")
    print(f"Smoothness: {metrics['trajectory_smoothness_mean']:.4f} +/- {metrics['trajectory_smoothness_std']:.4f}")
    print(f"Straightness: {metrics['trajectory_straightness_mean']:.4f} +/- {metrics['trajectory_straightness_std']:.4f}")
    print(f"  (1.0 = perfectly straight trajectory)")
    print(f"PCA Explained Variance: {pca.explained_variance_ratio_[0]:.3f}, {pca.explained_variance_ratio_[1]:.3f}")

    print(f"\nResults saved to {args.output_dir}")
    print(f"  - velocity_magnitude.png")
    print(f"  - trajectory_smoothness.png")
    print(f"  - flow_trajectories.png")
    print(f"  - flow_metrics.json")


if __name__ == "__main__":
    main()
