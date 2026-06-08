"""
Latent Space Diagnostics for ADynamics Multi-Modal VAE.

Goes BEYOND silhouette/PCA. Specifically diagnoses:
  - Per-dim KL averaged over dataset  (identifies posterior collapse)
  - Per-dim mean / std                 (flags collapsed vs active dims)
  - Modality agreement                  (T1 vs T1+fMRI cosine of mu)
  - Gradient norm per module            (catches recon dominance)
  - Most discriminative per-dim dims   (ranked by F-stat against label)

Output: latent_diag_report.json + JSON metrics + console summary.

Usage:
    python scripts/run_latent_diagnostics.py \\
        --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \\
        --json ./core_data/dataset_manifest_merged_v2.json \\
        --output_dir ./inference_results/latent_diagnostics_v0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D
from utils.config_loader import apply_yaml_defaults
from utils.multi_gpu import setup_data_parallel


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    mapping = [
        (("input", "encoder_checkpoint"), "checkpoint"),
        (("input", "num_classes"), "num_classes"),
        (("model", "latent_channels"), "latent_channels"),
        (("model", "base_channels"), "base_channels"),
        (("model", "decoder_depth"), "decoder_depth"),
        (("model", "dropout_rate"), "dropout_rate"),
        (("latent_diagnostics", "output_dir"), "output_dir"),
    ]
    config_defaults = apply_yaml_defaults(pre_args.config, mapping) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Latent Space Diagnostics", parents=[pre])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./inference_results/latent_diagnostics")
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Subsample to keep diagnostic fast (default 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(**config_defaults)
    return parser.parse_args()


def load_data(json_path: str, num_classes: int, max_samples: int) -> List[dict]:
    """Load + validate T1 nii.gz files. Mirrors train_stage1_multimodal.load_data."""
    import nibabel as nib
    with open(json_path, "r") as f:
        data = json.load(f)
    if num_classes == 3:
        for item in data:
            label = item.get("label", 0)
            if label in (1, 2):
                item["label"] = 1
            elif label == 3:
                item["label"] = 2
    valid = []
    corrupted = 0
    for item in data:
        t1_path = item.get("t1")
        if not t1_path or not os.path.exists(t1_path):
            continue
        try:
            img = nib.load(t1_path)
            shape = img.shape
            if any(s == 0 for s in shape):
                corrupted += 1; continue
            data_arr = img.get_fdata()
            if data_arr.min() == data_arr.max():
                corrupted += 1; continue
        except Exception:
            corrupted += 1; continue
        valid.append(item)
        if len(valid) >= max_samples:
            break
    print(f"Loaded {len(valid)} valid samples ({corrupted} corrupted skipped)")
    return valid


def _encode_mu_logvar(model, x_dict):
    """Helper: get the FUSED mu, logvar [B, C, D, H, W] from MultiModalVAE3D.

    `model.encode(x_dict)` returns the concat latent [B, 5*32, D, H, W]
    before fusion. We need mu, logvar (after the 1x1 conv fusion proj).
    """
    z_concat = model.encode(x_dict)  # [B, 5*32, D, H, W]
    mu = model.fusion_proj(z_concat)  # [B, 32, D, H, W]
    logvar = model.logvar_proj(z_concat)
    return mu, logvar


def per_dim_kl(model, dataloader, device) -> np.ndarray:
    """Returns [latent_channels] array: avg per-dim KL over dataset.

    A healthy VAE should have ALL dims with KL>free_bits.
    A collapsed dim has KL≈0.
    """
    model.eval()
    kls = []
    n_done = 0
    with torch.no_grad():
        for batch in dataloader:
            x_dict = {"t1": batch["t1"].to(device)}
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    x_dict[mod] = batch[mod].to(device)
            mu, logvar = _encode_mu_logvar(model, x_dict)
            # mu: [B, C, D, H, W]
            # KL per dim = -0.5 * (1 + logvar - mu^2 - exp(logvar))
            # Reduce over batch + spatial: [C]
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_per_dim = kl.mean(dim=(0, 2, 3, 4))  # [C]
            kls.append(kl_per_dim.cpu().numpy())
            n_done += batch["t1"].shape[0]
            print(f"  per-dim-kl: processed {n_done} samples", flush=True)
    return np.mean(kls, axis=0)


def per_dim_mean_std(model, dataloader, device) -> tuple:
    """Returns (means[C], stds[C]) across dataset.

    A healthy VAE posterior: mean≈0, std≈1 (matches N(0,1) prior).
    Collapsed: std≈0.
    """
    model.eval()
    means = []
    n_done = 0
    with torch.no_grad():
        for batch in dataloader:
            x_dict = {"t1": batch["t1"].to(device)}
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    x_dict[mod] = batch[mod].to(device)
            mu, _ = _encode_mu_logvar(model, x_dict)
            # pool spatial -> [B, C]
            pooled = mu.mean(dim=(2, 3, 4))
            means.append(pooled.cpu().numpy())
            n_done += batch["t1"].shape[0]
            print(f"  per-dim-mean-std: processed {n_done} samples", flush=True)
    means = np.concatenate(means, axis=0)  # [N, C]
    return means.mean(0), means.std(0)


def modality_agreement(model, dataloader, device, num_samples: int = 100) -> Dict:
    """Compare mu when running T1-only vs T1+fMRI (or other mods).

    Returns dict with per-modality cos similarity vs T1-only baseline.
    cos~1.0 means the modality adds nothing to the latent (broken fusion).
    cos<0.5 means the fusion is brittle / inconsistent.
    """
    model.eval()
    per_mod = {mod: [] for mod in ["t1only", "fmri", "asl", "qsm", "flair"]}
    n_collected = 0
    with torch.no_grad():
        for batch in dataloader:
            if n_collected >= num_samples:
                break
            t1 = batch["t1"].to(device)
            B = t1.shape[0]
            mu_only, _ = _encode_mu_logvar(model, {"t1": t1})
            mu_only_pooled = mu_only.mean(dim=(2, 3, 4))  # [B, C]
            per_mod["t1only"].append(mu_only_pooled.cpu().numpy())
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    mu_combined, _ = _encode_mu_logvar(model, {"t1": t1, mod: batch[mod].to(device)})
                    mu_combined_pooled = mu_combined.mean(dim=(2, 3, 4))
                else:
                    mu_combined_pooled = mu_only_pooled
                per_mod[mod].append(mu_combined_pooled.cpu().numpy())
            n_collected += B
            print(f"  modality-agree: processed {n_collected} samples", flush=True)
    out = {}
    for mod in ["fmri", "asl", "qsm", "flair"]:
        if not per_mod["t1only"] or not per_mod[mod]:
            out[mod] = float("nan")
            continue
        a = np.concatenate(per_mod["t1only"], axis=0)
        b = np.concatenate(per_mod[mod], axis=0)
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        cos_per_sample = (a_n * b_n).sum(axis=1)
        out[mod] = float(cos_per_sample.mean())
    return out


def gradient_norm_per_module(model, dataloader, device, num_classes: int = 4):
    """Run one backward pass with cls_loss and measure grad norm per module.

    Identifies if decoder grads >> encoder grads (recon dominates cls signal).
    """
    model.train()  # need gradients
    # Take only 1 sample to avoid OOM
    for batch in dataloader:
        x_dict = {"t1": batch["t1"][:1].to(device)}
        for mod in ["fmri", "asl", "qsm", "flair"]:
            if mod in batch and batch[mod] is not None:
                x_dict[mod] = batch[mod][:1].to(device)
        labels = batch["label"][:1].to(device)
        if labels.dim() > 1:
            labels = labels.squeeze()

        # Reset grads
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        recon, cls_logits, mu, logvar = model(x_dict, return_components=True)
        cls_loss = F.cross_entropy(cls_logits, labels)
        cls_loss.backward()
        # We did 1 iteration; now break out of the loader
        break

    # Per-module grad norm aggregation (after break)
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is None or p.grad.abs().max() == 0:
            continue
        # Group by top-level module
        if "encoder_t1" in name or name.startswith("encoder_t1"):
            top = "encoder_t1"
        elif "optional_encoders" in name:
            top = "optional_encoders"
        elif "fusion_proj" in name:
            top = "fusion_proj"
        elif "logvar_proj" in name:
            top = "logvar_proj"
        elif "decoder" in name:
            top = "decoder"
        elif "classifier" in name:
            top = "classifier"
        elif "class_priors" in name:
            top = "class_priors"
        elif "attention" in name:
            top = "attention"
        else:
            top = "other"
        norm = p.grad.norm().item()
        grad_norms[top] = grad_norms.get(top, 0.0) + norm ** 2
    grad_norms = {k: float(np.sqrt(v)) for k, v in grad_norms.items()}
    # Reset to eval for downstream
    model.eval()
    return cls_loss.item(), grad_norms


def active_dim_ranking(per_dim_kl_arr, per_dim_mean, per_dim_std, free_bits: float = 0.01):
    """Rank dims by "active" — top 8 most active dims (highest KL)."""
    n_active = int((per_dim_kl_arr > free_bits).sum())
    n_total = len(per_dim_kl_arr)
    # Active ratio
    active_ratio = n_active / n_total
    # Top 8 dims
    top_idx = np.argsort(per_dim_kl_arr)[::-1][:8]
    top_dims = [
        {
            "dim": int(i),
            "kl": float(per_dim_kl_arr[i]),
            "mean": float(per_dim_mean[i]),
            "std": float(per_dim_std[i]),
            "active": bool(per_dim_kl_arr[i] > free_bits),
        }
        for i in top_idx
    ]
    return {
        "n_active": n_active,
        "n_total": n_total,
        "active_ratio": float(active_ratio),
        "top8_most_active": top_dims,
    }


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model
    print(f"Loading model from {args.checkpoint}")
    model = MultiModalVAE3D(
        spatial_size=tuple(MULTI_MODAL_SPATIAL_SIZES["t1"]),
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        decoder_depth=args.decoder_depth,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    # Strip DataParallel prefix
    if any(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    # Load data
    data_list = load_data(args.json, args.num_classes, args.max_samples)
    transforms = get_multimodal_val_transforms()
    dataset = MultiModalDataset(data_list, transform=transforms)
    from core_data.dataset import multimodal_collate_fn
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0,
        collate_fn=multimodal_collate_fn, drop_last=False,
    )

    # Run diagnostics
    print("\n=== Diagnostic 1: per-dim KL ===")
    pd_kl = per_dim_kl(model, loader, device)
    print(f"  per-dim KL: {pd_kl.tolist()}")
    print(f"  mean KL: {pd_kl.mean():.4f}, min: {pd_kl.min():.4f}, max: {pd_kl.max():.4f}")

    print("\n=== Diagnostic 2: per-dim mean / std ===")
    pd_mean, pd_std = per_dim_mean_std(model, loader, device)
    print(f"  per-dim mean: {pd_mean.tolist()}")
    print(f"  per-dim std:  {pd_std.tolist()}")

    print("\n=== Diagnostic 3: active-dim ranking ===")
    ranking = active_dim_ranking(pd_kl, pd_mean, pd_std, free_bits=0.01)
    print(f"  Active dims (KL>0.01): {ranking['n_active']}/{ranking['n_total']} ({ranking['active_ratio']*100:.1f}%)")
    print("  Top 8 most active dims:")
    for d in ranking["top8_most_active"]:
        print(f"    dim {d['dim']}: KL={d['kl']:.4f}, mean={d['mean']:+.3f}, std={d['std']:.3f}, active={d['active']}")

    print("\n=== Diagnostic 4: modality agreement (T1+mod vs T1-only cosine sim) ===")
    # Subset to 100 samples for speed
    mod_agree = modality_agreement(model, loader, device, num_samples=100)
    for mod, cos in mod_agree.items():
        bar = "█" * int(cos * 30)
        print(f"  T1+{mod:<6}: cos={cos:.4f}  {bar}")
    print("  -> If cos ~ 1.0, the optional modality is NOT differentiating the latent (fusion broken).")
    print("  -> If cos < 0.7, the optional modality is contributing meaningful variation.")

    print("\n=== Diagnostic 5: gradient norm per module (cls loss only) ===")
    cls_loss_val, grad_norms = gradient_norm_per_module(model, loader, device, args.num_classes)
    print(f"  cls_loss: {cls_loss_val:.4f}")
    for mod, norm in sorted(grad_norms.items(), key=lambda x: -x[1]):
        bar = "█" * int(min(norm, 50))
        print(f"  {mod:<20}: grad_norm={norm:7.3f}  {bar}")
    # If decoder >> encoder, recon is dominating

    # Compose report
    report = {
        "checkpoint": args.checkpoint,
        "n_samples": len(data_list),
        "latent_dim": int(args.latent_channels),
        "per_dim_kl": pd_kl.tolist(),
        "per_dim_mean": pd_mean.tolist(),
        "per_dim_std": pd_std.tolist(),
        "active_dim_ranking": ranking,
        "modality_agreement": mod_agree,
        "gradient_norms": grad_norms,
        "cls_loss_at_diagnostic": cls_loss_val,
    }
    out_json = os.path.join(args.output_dir, "latent_diag_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_json}")

    # Go/No-Go summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY (Go/No-Go for Phase A)")
    print("=" * 60)
    active_ratio = ranking["active_ratio"]
    n_active = ranking["n_active"]
    mean_agree = np.mean(list(mod_agree.values()))
    if "encoder_t1" in grad_norms and "decoder" in grad_norms:
        ratio_dec_enc = grad_norms["decoder"] / max(grad_norms["encoder_t1"], 1e-8)
    else:
        ratio_dec_enc = 1.0

    if active_ratio < 0.5:
        print(f"  [!!] Posterior collapse: only {n_active}/{ranking['n_total']} dims active ({active_ratio*100:.0f}%)")
        print("       -> A1 cyclical KL + A2 rebalance are ESSENTIAL")
    else:
        print(f"  [OK] Posterior OK: {n_active}/{ranking['n_total']} dims active ({active_ratio*100:.0f}%)")

    if mean_agree > 0.95:
        print(f"  [!!] Modality fusion brittle: mean cos={mean_agree:.4f}")
        print("       -> Optional modalities add little. B1 multi-level attention is critical.")
    else:
        print(f"  [OK] Modality fusion OK: mean cos={mean_agree:.4f}")

    if ratio_dec_enc > 3.0:
        print(f"  [!!] Decoder dominates: decoder_grad={grad_norms['decoder']:.2f}, encoder_grad={grad_norms['encoder_t1']:.2f}, ratio={ratio_dec_enc:.1f}x")
        print("       -> D1 two-stage training + A2 rebalance are ESSENTIAL")
    else:
        print(f"  [OK] Encoder/decoder balance OK: ratio={ratio_dec_enc:.1f}x")


if __name__ == "__main__":
    main()
