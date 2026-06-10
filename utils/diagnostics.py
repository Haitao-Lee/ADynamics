"""
Deep diagnostic metrics for ADynamics Stage 1.

These go beyond val_acc and recon_loss to expose *why* the model is or
isn't learning. Each function is a pure compute step on tensors; the
trainer/validator owns the per-epoch aggregation.

Functions (all take torch tensors and return a dict of float metrics):
    - silhouette_score         : latent class separability (the headline number)
    - per_class_centroid_dist  : how far apart the 4 class centroids are in latent
    - per_class_pred_freq      : model's prediction distribution (collapse signal)
    - per_dim_latent_stats     : which latent dims are alive vs collapsed
    - grad_norm_by_module      : encoder/decoder/classifier gradient magnitude
    - recon_intensity_stats    : T1 mean/std/min/max (input data sanity check)

Why this lives in utils/ and not engine/
    Diagnostics are conceptually separable from training (could be run on a
    frozen model for offline analysis). Keeping them in utils/ means the
    run_latent_analysis.py script can also import them.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Latent class structure
# ---------------------------------------------------------------------------

@torch.no_grad()
def silhouette_score(
    mu_pooled: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Silhouette score on pooled mu (latent per sample, after spatial mean).

    Args:
        mu_pooled: [N, C] latent mean per sample (after spatial pooling).
        labels: [N] integer class labels.

    Returns:
        Silhouette in [-1, 1]. +1 = perfect clusters, 0 = overlapping,
        -1 = wrong clusters. 0.10 is a meaningful improvement over 0.00.

    Notes:
        This is a reimplementation of sklearn's silhouette_score using
        pytorch, so we don't depend on scikit-learn at training time.
        O(N^2) in memory and compute; with N=221 (val set) this is fine.
    """
    if mu_pooled.size(0) < 2:
        return 0.0
    # Move to CPU float32 for stable cosine distance computation
    X = mu_pooled.detach().to(torch.float32).cpu()
    y = labels.detach().to(torch.long).cpu()
    n = X.size(0)

    # Pairwise L2 distance matrix
    dist = torch.cdist(X, X, p=2)  # [N, N]

    sil = torch.zeros(n)
    for i in range(n):
        same = (y == y[i])
        other = ~same
        if same.sum() <= 1:
            # Only one sample of this class in the set; silhouette undefined.
            sil[i] = 0.0
            continue
        if other.sum() == 0:
            sil[i] = 0.0
            continue
        a = dist[i, same].sum() / (same.sum() - 1)  # mean dist to same class
        b = dist[i, other].min()  # nearest other-class distance (NOT mean, but min-cluster)
        # The min version is more robust to many classes (standard formulation)
        # Actually, standard uses MEAN distance to nearest cluster; we approximate
        # by mean over all "other" points (since we have 4 classes and 1 vs 3).
        b = dist[i, other].mean()
        sil[i] = (b - a) / max(a, b).clamp(min=1e-8)
    return sil.mean().item()


def per_class_centroid_distance(
    mu_pooled: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 4,
) -> Dict[str, float]:
    """Pairwise centroid distances in latent space.

    Returns:
        Dict with keys like "c01_dist" (class 0 vs 1), "c02_dist", etc.
        Useful to see if NC and AD separate (expected) but NC and SCD don't
        (which is the disease progression signal we want to learn).
    """
    out: Dict[str, float] = {}
    centroids = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            centroids.append(None)
            continue
        centroids.append(mu_pooled[mask].mean(dim=0))
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            if centroids[i] is None or centroids[j] is None:
                out[f"c{i}{j}_dist"] = 0.0
                continue
            out[f"c{i}{j}_dist"] = (centroids[i] - centroids[j]).norm().item()
    return out


def per_class_pred_frequency(
    preds: torch.Tensor,
    num_classes: int = 4,
) -> List[float]:
    """Histogram of model predictions (how often does it predict each class?).

    If the model collapses to predicting one class, this shows it.
    """
    freqs = torch.zeros(num_classes)
    for c in range(num_classes):
        freqs[c] = (preds == c).float().mean().item()
    return freqs.tolist()


# ---------------------------------------------------------------------------
# Per-dim latent statistics
# ---------------------------------------------------------------------------

def per_dim_latent_stats(
    mu_pooled: torch.Tensor,
    free_bits: float = 0.0,
) -> Dict[str, float]:
    """Per-dim statistics on pooled mu (after spatial mean).

    Args:
        mu_pooled: [N, C] (N samples, C latent dims).
        free_bits: KL floor; dims with std below sqrt(2*free_bits+eps) are
            considered collapsed.

    Returns:
        Dict with:
            "latent_std_mean"  : mean per-dim std across dims
            "latent_std_min"   : smallest per-dim std (worst dim)
            "latent_std_max"   : largest per-dim std
            "n_active_dims"    : # dims above collapse threshold
            "n_collapsed_dims" : # dims below threshold
            "top5_active_dims" : indices of 5 most active dims
            "bot5_active_dims" : indices of 5 most collapsed dims
    """
    if mu_pooled.size(0) < 2:
        return {
            "latent_std_mean": 0.0, "latent_std_min": 0.0, "latent_std_max": 0.0,
            "n_active_dims": 0, "n_collapsed_dims": 0,
            "top5_active_dims": -1, "bot5_active_dims": -1,
        }
    per_dim_std = mu_pooled.std(dim=0)  # [C]
    threshold = math.sqrt(2.0 * free_bits + 1e-6)
    active_mask = per_dim_std > threshold
    n_active = int(active_mask.sum().item())
    n_total = per_dim_std.numel()
    top5 = torch.topk(per_dim_std, k=min(5, n_total)).indices.tolist()
    bot5 = torch.topk(per_dim_std, k=min(5, n_total), largest=False).indices.tolist()
    return {
        "latent_std_mean": per_dim_std.mean().item(),
        "latent_std_min": per_dim_std.min().item(),
        "latent_std_max": per_dim_std.max().item(),
        "n_active_dims": n_active,
        "n_collapsed_dims": n_total - n_active,
        "top5_active_dims": top5,
        "bot5_active_dims": bot5,
    }


# ---------------------------------------------------------------------------
# Gradient norm by module (for diagnosing starvation/imbalance)
# ---------------------------------------------------------------------------

def grad_norm_by_module(
    model: torch.nn.Module,
    module_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Sum of grad-norm across parameters belonging to each module group.

    Args:
        model: the (raw) model with grads already attached.
        module_names: list of substrings to group by. Default groups are
            ["encoder_t1", "decoder", "classifier", "logvar", "demo"].

    Returns:
        Dict like {"encoder_t1_grad": 1.234, "decoder_grad": 0.567, ...}
    """
    if module_names is None:
        module_names = [
            "encoder_t1", "encoder_fmri", "encoder_asl",
            "encoder_qsm", "encoder_flair", "decoder",
            "classifier", "logvar", "demo",
        ]
    sums: Dict[str, float] = {f"{n}_grad": 0.0 for n in module_names}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        gn = param.grad.norm().item()
        for n in module_names:
            if n in name:
                sums[f"{n}_grad"] += gn
                break
    return sums


# ---------------------------------------------------------------------------
# Input data sanity check
# ---------------------------------------------------------------------------

def recon_intensity_stats(
    images: torch.Tensor,
    recon: torch.Tensor,
) -> Dict[str, float]:
    """Stats on the input and reconstruction (intensity, range).

    Args:
        images: [B, 1, D, H, W] input T1 (after preprocessing, in [0, 1] ideally)
        recon: [B, 1, D, H, W] reconstruction

    Returns:
        Dict with:
            "input_mean", "input_std", "input_min", "input_max"
            "recon_mean", "recon_std", "recon_min", "recon_max"
            "recon_relative_error" : mean abs error / input mean abs value
    """
    return {
        "input_mean": images.mean().item(),
        "input_std": images.std().item(),
        "input_min": images.min().item(),
        "input_max": images.max().item(),
        "recon_mean": recon.mean().item(),
        "recon_std": recon.std().item(),
        "recon_min": recon.min().item(),
        "recon_max": recon.max().item(),
        "recon_relative_error": (
            (recon - images).abs().mean().item() / images.abs().mean().clamp(min=1e-6).item()
        ),
    }


# ---------------------------------------------------------------------------
# Mixup statistics
# ---------------------------------------------------------------------------

def mixup_stats(
    n_mixed: int,
    n_total: int,
) -> Dict[str, float]:
    """Track how often mixup actually fired.

    Args:
        n_mixed: number of batches where mixup was applied.
        n_total: total batches.

    Returns:
        {"mixup_count": ..., "mixup_frac": ...}
    """
    return {
        "mixup_count": float(n_mixed),
        "mixup_frac": (n_mixed / max(1, n_total)),
    }
