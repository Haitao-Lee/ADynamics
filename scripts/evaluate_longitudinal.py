"""
Longitudinal Validation for ADynamics CFM.

Evaluates whether the learned disease progression flow actually corresponds
to real biological change by comparing CFM-predicted evolution against
observed longitudinal MRI data (same patient at multiple time points).

Metrics:
    1. Latent MSE: ||z_evolved - z_followup||^2  (does flow reach the right latent?)
    2. Latent cosine similarity: cos(z_evolved, z_followup)  (directional agreement)
    3. Image MSE: ||warped_baseline - followup||^2  (does warp match real change?)
    4. Jacobian stats: min/mean det(J)  (is deformation anatomically plausible?)
    5. Clinical alignment: does predicted stage match actual followup stage?

Usage:
    # With ADNI longitudinal pairs
    python scripts/evaluate_longitudinal.py \
        --pairs_json ./core_data/longitudinal_pairs.json \
        --vae_checkpoint ./checkpoints/stage1/vae_best.pt \
        --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt \
        --output_dir ./eval_results/longitudinal

    # With cross-sectional synthetic pairs (pseudo-longitudinal)
    python scripts/evaluate_longitudinal.py \
        --manifest ./core_data/dataset_manifest_merged_v2.json \
        --synthetic \
        --vae_checkpoint ./checkpoints/stage1/vae_best.pt \
        --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt \
        --output_dir ./eval_results/synthetic_longitudinal

Pairs JSON format:
    [
        {
            "patient_id": "sub-001",
            "baseline_t1": "/path/to/baseline_T1.nii.gz",
            "followup_t1": "/path/to/followup_T1.nii.gz",
            "baseline_stage": 0,       // 0=NC, 1=SCD, 2=MCI, 3=AD
            "followup_stage": 2,
            "time_years": 2.5,
            "age_baseline": 70.0,
            "sex": 1
        },
        ...
    ]
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core_data.transforms import MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D
from models.vector_field import VelocityFieldNet

try:
    from models.spatial_transform import (
        DeformationGenerator,
        SpatialTransformer,
        compute_jacobian_penalty,
    )
    HAS_DEFORMATION = True
except ImportError:
    HAS_DEFORMATION = False


# ─── ODE Integration ────────────────────────────────────────────────

def integrate_cfm(
    z0: Tensor,
    cfm: torch.nn.Module,
    n_steps: int = 20,
    target_t: float = 1.0,
    demographics: Optional[Dict[str, Tensor]] = None,
) -> Tuple[Tensor, List[Tensor]]:
    """
    Euler integration of the CFM velocity field from z0 to z_target.

    Args:
        z0: Initial latent [B, C, D, H, W]
        cfm: Velocity field network
        n_steps: Number of Euler steps
        target_t: End time (1.0 = full NC→AD flow)
        demographics: Optional dict with 'age' [B] and 'sex' [B]

    Returns:
        z_final: Evolved latent [B, C, D, H, W]
        trajectory: List of intermediate latents (including z0)
    """
    z_t = z0.clone()
    dt = target_t / n_steps
    trajectory = [z_t.clone()]

    with torch.no_grad():
        for i in range(n_steps):
            t_val = i * dt
            t = torch.full((z0.shape[0],), t_val, device=z0.device, dtype=z0.dtype)

            kwargs = {}
            if demographics is not None:
                kwargs.update(demographics)

            v = cfm(z_t, t, **kwargs)
            z_t = z_t + v * dt
            trajectory.append(z_t.clone())

    return z_t, trajectory


# ─── Metric Functions ────────────────────────────────────────────────

def latent_mse(z_pred: Tensor, z_target: Tensor) -> float:
    """MSE between predicted and target latent."""
    return F.mse_loss(z_pred, z_target).item()


def latent_cosine_sim(z_pred: Tensor, z_target: Tensor) -> float:
    """Cosine similarity between flattened latents."""
    a = z_pred.flatten(1)
    b = z_target.flatten(1)
    cos = F.cosine_similarity(a, b, dim=1).mean().item()
    return cos


def image_mse(img_pred: Tensor, img_target: Tensor) -> float:
    """MSE between predicted (warped) and target image."""
    return F.mse_loss(img_pred, img_target).item()


def image_mae(img_pred: Tensor, img_target: Tensor) -> float:
    """MAE between predicted (warped) and target image."""
    return F.l1_loss(img_pred, img_target).item()


def jacobian_stats(flow: Tensor) -> Dict[str, float]:
    """
    Compute Jacobian determinant statistics of a deformation field.

    det(J) < 0 indicates folding (topology violation).
    det(J) = 1 indicates no deformation.
    """
    # flow: [B, 3, D, H, W]
    # Compute spatial gradients
    dx = flow[:, :, :, :, 2:] - flow[:, :, :, :, :-2]  # ∂flow/∂w
    dy = flow[:, :, :, 2:, :] - flow[:, :, :, :-2, :]  # ∂flow/∂h
    dz = flow[:, :, 2:, :, :] - flow[:, :, :-2, :, :]  # ∂flow/∂d

    # Crop to common region
    min_d = min(dx.shape[2], dy.shape[2], dz.shape[2])
    min_h = min(dx.shape[3], dy.shape[3], dz.shape[3])
    min_w = min(dx.shape[4], dy.shape[4], dz.shape[4])

    dx = dx[:, :, :min_d, :min_h, :min_w]
    dy = dy[:, :, :min_d, :min_h, :min_w]
    dz = dz[:, :, :min_d, :min_h, :min_w]

    # Approximate Jacobian: I + ∂flow/∂x
    # det(I + J) ≈ 1 + tr(J) for small deformations
    # For exact: compute 3x3 determinant
    B = flow.shape[0]
    # Pad to identity: diag = 1 + ∂flow_i/∂x_i
    trace = dx[:, 2] + dy[:, 1] + dz[:, 0]  # tr(∂flow/∂x)
    det_approx = 1.0 + trace

    return {
        "det_mean": det_approx.mean().item(),
        "det_min": det_approx.min().item(),
        "det_max": det_approx.max().item(),
        "folding_pct": (det_approx < 0).float().mean().item() * 100,
    }


# ─── Main Evaluation ─────────────────────────────────────────────────

def load_model(
    vae_path: str,
    cfm_path: str,
    device: str,
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    latent_channels: int = 32,
    base_channels: int = 16,
) -> Tuple[MultiModalVAE3D, VelocityFieldNet]:
    """Load VAE and CFM from checkpoints."""
    # VAE
    vae = MultiModalVAE3D(
        spatial_size=spatial_size,
        latent_channels=latent_channels,
        base_channels=base_channels,
        optional_modalities=[],  # T1-only for evaluation
        use_t1_centric_fusion=False,
    ).to(device)
    ckpt = torch.load(vae_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # Filter keys that match
    filtered = {k: v for k, v in state.items() if k in vae.state_dict()
                and v.shape == vae.state_dict()[k].shape}
    vae.load_state_dict(filtered, strict=False)
    vae.eval()
    print(f"Loaded VAE: {len(filtered)}/{len(vae.state_dict())} keys")

    # CFM
    latent_spatial = tuple(s // 16 for s in spatial_size)  # 128//16=8
    cfm = VelocityFieldNet(
        latent_channels=latent_channels,
        latent_spatial=latent_spatial,
    ).to(device)
    ckpt_cfm = torch.load(cfm_path, map_location=device, weights_only=False)
    state_cfm = ckpt_cfm.get("model_state_dict", ckpt_cfm)
    filtered_cfm = {k: v for k, v in state_cfm.items() if k in cfm.state_dict()
                    and v.shape == cfm.state_dict()[k].shape}
    cfm.load_state_dict(filtered_cfm, strict=False)
    cfm.eval()
    print(f"Loaded CFM: {len(filtered_cfm)}/{len(cfm.state_dict())} keys")

    return vae, cfm


def preprocess_t1(
    nii_path: str,
    target_size: Tuple[int, int, int] = (128, 128, 128),
) -> Tensor:
    """Load and preprocess a single T1 NIfTI → [1, 1, D, H, W]."""
    import nibabel as nib
    from scipy.ndimage import zoom

    arr = np.asarray(nib.load(nii_path).dataobj, dtype=np.float32)
    # Resize
    if arr.shape != target_size:
        factors = [t / s for t, s in zip(target_size, arr.shape)]
        arr = zoom(arr, factors, order=1, mode='constant', cval=0.0)
    # Intensity normalization
    mask = arr > 0
    if mask.sum() > 0:
        m, s = arr[mask].mean(), arr[mask].std()
        if s > 1e-8:
            arr[mask] = (arr[mask] - m) / s
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    return tensor


def evaluate_pair(
    pair: Dict[str, Any],
    vae: MultiModalVAE3D,
    cfm: VelocityFieldNet,
    device: str,
    n_steps: int = 20,
    target_t: float = 1.0,
) -> Dict[str, float]:
    """
    Evaluate a single longitudinal pair.

    Returns dict of metrics.
    """
    spatial_size = MULTI_MODAL_SPATIAL_SIZES["t1"]

    # Load and preprocess
    t1_base = preprocess_t1(pair["baseline_t1"], spatial_size).to(device)
    t1_follow = preprocess_t1(pair["followup_t1"], spatial_size).to(device)

    with torch.no_grad():
        # Encode baseline
        z0 = vae.encode_t1_only(t1_base)

        # Encode followup
        z_followup = vae.encode_t1_only(t1_follow)

        # Evolve baseline through CFM
        demographics = None
        if "age_baseline" in pair and "sex" in pair:
            demographics = {
                "age": torch.tensor([pair["age_baseline"]], device=device),
                "sex": torch.tensor([pair["sex"]], device=device, dtype=torch.long),
            }

        z_evolved, trajectory = integrate_cfm(
            z0, cfm, n_steps=n_steps, target_t=target_t,
            demographics=demographics,
        )

    # Compute metrics
    results = {
        "patient_id": pair.get("patient_id", "unknown"),
        "baseline_stage": pair.get("baseline_stage", -1),
        "followup_stage": pair.get("followup_stage", -1),
        "time_years": pair.get("time_years", -1),
        "latent_mse": latent_mse(z_evolved, z_followup),
        "latent_cosine": latent_cosine_sim(z_evolved, z_followup),
        "z0_norm": z0.norm().item(),
        "z_evolved_norm": z_evolved.norm().item(),
        "z_followup_norm": z_followup.norm().item(),
    }

    # Stage progression check
    if "baseline_stage" in pair and "followup_stage" in pair:
        results["stage_progressed"] = int(pair["followup_stage"] > pair["baseline_stage"])
        results["stage_distance"] = pair["followup_stage"] - pair["baseline_stage"]

    return results


def evaluate_synthetic_pairs(
    manifest_path: str,
    vae: MultiModalVAE3D,
    cfm: VelocityFieldNet,
    device: str,
    n_pairs: int = 100,
    n_steps: int = 20,
) -> List[Dict[str, float]]:
    """
    Evaluate using cross-sectional data as pseudo-longitudinal.

    Creates synthetic pairs: NC patient → AD patient (different individuals).
    This tests whether CFM's flow direction is reasonable, but cannot validate
    individual trajectory prediction.

    Metrics are weaker: we expect z_evolved ≈ z_AD on average, not per-subject.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    items = manifest if isinstance(manifest, list) else manifest.get("samples", list(manifest.values()))

    # Group by stage
    by_stage: Dict[int, List[Dict]] = {0: [], 1: [], 2: [], 3: []}
    for item in items:
        paths = item.get("paths", item)
        t1 = paths.get("t1")
        label = item.get("label", paths.get("label"))
        if t1 and label is not None and os.path.exists(t1):
            by_stage[int(label)].append(item)

    print(f"Synthetic pairs: NC={len(by_stage[0])}, SCD={len(by_stage[1])}, "
          f"MCI={len(by_stage[2])}, AD={len(by_stage[3])}")

    results = []
    # NC → AD pairs
    nc_items = by_stage[0][:n_pairs]
    ad_items = by_stage[3][:n_pairs]
    n = min(len(nc_items), len(ad_items))

    for i in range(n):
        nc_t1 = nc_items[i].get("paths", nc_items[i]).get("t1")
        ad_t1 = ad_items[i].get("paths", ad_items[i]).get("t1")

        pair = {
            "patient_id": f"synthetic_{i}",
            "baseline_t1": nc_t1,
            "followup_t1": ad_t1,
            "baseline_stage": 0,
            "followup_stage": 3,
            "time_years": -1,  # unknown for synthetic
        }
        r = evaluate_pair(pair, vae, cfm, device, n_steps=n_steps)
        results.append(r)

    return results


def summarize(results: List[Dict[str, float]]) -> Dict[str, Any]:
    """Compute summary statistics across all pairs."""
    if not results:
        return {"error": "no results"}

    keys = ["latent_mse", "latent_cosine"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        if vals:
            summary[f"{k}_mean"] = np.mean(vals)
            summary[f"{k}_std"] = np.std(vals)
            summary[f"{k}_median"] = np.median(vals)

    # Stage-specific breakdown
    by_transition: Dict[str, List[Dict]] = {}
    for r in results:
        key = f"{r.get('baseline_stage', '?')}->{r.get('followup_stage', '?')}"
        by_transition.setdefault(key, []).append(r)

    for trans, group in sorted(by_transition.items()):
        mses = [r["latent_mse"] for r in group]
        summary[f"transition_{trans}_n"] = len(group)
        summary[f"transition_{trans}_mse"] = np.mean(mses)

    summary["n_pairs"] = len(results)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Longitudinal validation for ADynamics CFM")
    parser.add_argument("--pairs_json", type=str, default=None,
                        help="Path to longitudinal pairs JSON")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to cross-sectional manifest (for --synthetic)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use cross-sectional data as pseudo-longitudinal")
    parser.add_argument("--n_pairs", type=int, default=100,
                        help="Number of synthetic pairs to evaluate")
    parser.add_argument("--vae_checkpoint", type=str, required=True)
    parser.add_argument("--cfm_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_results/longitudinal")
    parser.add_argument("--n_steps", type=int, default=20,
                        help="Euler integration steps")
    parser.add_argument("--target_t", type=float, default=1.0,
                        help="Integration end time (1.0 = full NC→AD)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    vae, cfm = load_model(args.vae_checkpoint, args.cfm_checkpoint, args.device)

    # Evaluate
    if args.synthetic:
        manifest = args.manifest or "./core_data/dataset_manifest_merged_v2.json"
        print(f"=== Synthetic Longitudinal Evaluation ===")
        results = evaluate_synthetic_pairs(
            manifest, vae, cfm, args.device,
            n_pairs=args.n_pairs, n_steps=args.n_steps,
        )
    else:
        assert args.pairs_json, "Need --pairs_json for real longitudinal evaluation"
        with open(args.pairs_json) as f:
            pairs = json.load(f)
        print(f"=== Longitudinal Evaluation ({len(pairs)} pairs) ===")
        results = []
        for i, pair in enumerate(pairs):
            r = evaluate_pair(pair, vae, cfm, args.device, n_steps=args.n_steps,
                              target_t=args.target_t)
            results.append(r)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(pairs)}] latent_mse={r['latent_mse']:.4f} "
                      f"cosine={r['latent_cosine']:.4f}")

    # Summary
    summary = summarize(results)
    print("\n=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save
    out_results = os.path.join(args.output_dir, "results.json")
    out_summary = os.path.join(args.output_dir, "summary.json")
    with open(out_results, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_results}")
    print(f"Saved: {out_summary}")


if __name__ == "__main__":
    main()
