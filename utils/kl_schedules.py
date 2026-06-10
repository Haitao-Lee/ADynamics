"""
KL weight schedules and latent-space mixup for ADynamics Stage 1.

Why these exist
---------------
The default linear warmup (KL goes 0→target once) permanently commits the
encoder to N(0,1) once warmup ends, killing class-discriminative structure
in the latent (silhouette_score ≈ -0.01 with linear).

Cyclical KL (Fu et al. 2019, "Cyclical Annealing Schedule") lets the encoder
use latent capacity for class information during KL-off phases, then "bake in"
that structure when KL climbs back up. Empirically this turns the latent from
class-blind to class-aware in the same number of epochs.

Latent-space mixup (Zhang et al. 2018, "mixup: Beyond Empirical Risk
Minimization") trains the decoder and classifier to handle convex
combinations of latents, which acts as a strong regularizer against the
18% train/val gap we saw with 1247 train samples and batch=2.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def linear_kl_weight(epoch: int, target: float, warmup_epochs: int) -> float:
    """Original linear warmup: 0 -> target over warmup_epochs, then constant.

    Kept for ablation. Use cyclical_kl_weight in production.
    """
    if epoch < warmup_epochs:
        return target * (epoch + 1) / max(1, warmup_epochs)
    return target


def cyclical_kl_weight(
    epoch: int,
    target: float,
    cycle_len: int,
    low_frac: float = 0.1,
) -> float:
    """Cyclical KL annealing (Fu et al. 2019).

    Within each cycle of length `cycle_len`:
        - First half: KL climbs from low_frac*target -> target (linearly)
        - Second half: KL drops from target -> low_frac*target (linearly)

    This lets the encoder "rest" between KL-pressure cycles, during which it
    fills the latent with class-discriminative info. When KL comes back on,
    that structure gets regularized (not destroyed) into a tighter N(0,1)
    fit. Net effect: latent becomes class-aware.

    Args:
        epoch: 0-indexed current epoch.
        target: Peak KL weight (e.g. 0.3 from config).
        cycle_len: Epochs per cycle. 15 is a sane default.
        low_frac: Minimum fraction of target. 0.0 = full KL-off, 0.1 = never
            below 10% (recommended to prevent total posterior collapse
            between cycles).

    Returns:
        KL weight for this epoch, in [low_frac * target, target].
    """
    cycle_len = max(2, int(cycle_len))  # need at least 2 epochs per half-cycle
    pos_in_cycle = epoch % cycle_len
    half = cycle_len / 2.0
    if pos_in_cycle < half:
        # Climb low -> target
        frac = pos_in_cycle / max(1.0, half)
    else:
        # Drop target -> low
        frac = 1.0 - (pos_in_cycle - half) / max(1.0, half)
    frac = max(0.0, min(1.0, frac))
    return target * (low_frac + (1.0 - low_frac) * frac)


def get_kl_weight(epoch: int, config: dict) -> Tuple[float, str]:
    """Dispatch helper: returns (kl_weight, strategy_name) for this epoch.

    Reads config["kl_strategy"] to pick the schedule.
    """
    target = config.get("kl_weight", 0.3)
    strategy = config.get("kl_strategy", "linear")
    if strategy == "cyclical":
        cycle_len = config.get("kl_cycle_len", 15)
        low_frac = config.get("kl_cycle_low_frac", 0.1)
        w = cyclical_kl_weight(epoch, target, cycle_len=cycle_len, low_frac=low_frac)
    else:
        warmup = config.get("kl_warmup_epochs", 30)
        w = linear_kl_weight(epoch, target, warmup)
    return w, strategy


# ---------------------------------------------------------------------------
# Latent-space mixup (Zhang et al. 2018)
# ---------------------------------------------------------------------------

def should_apply_mixup(epoch: int, config: dict) -> bool:
    """Decide whether to mixup this batch.

    Reads config["mixup_alpha"] and config["mixup_prob"].
    If mixup_alpha <= 0, always returns False (disabled).
    """
    alpha = config.get("mixup_alpha", 0.0)
    prob = config.get("mixup_prob", 0.5)
    if alpha <= 0.0:
        return False
    # Deterministic schedule: more mixup early, taper off late
    # so the final epochs can sharpen the class boundaries.
    return torch.rand(1).item() < prob


def sample_mixup_lambda(alpha: float, device: torch.device) -> float:
    """Sample lambda ~ Beta(alpha, alpha). Always in [0, 1].

    For alpha=0.4, lambda is concentrated near 0 and 1 (strong mixup).
    For alpha=1.0, lambda is uniform (mild mixup).
    For alpha=2.0, lambda is concentrated near 0.5 (gentle mixup).
    """
    if alpha <= 0:
        return 1.0
    # Use scipy-free Beta sampling via Gamma trick.
    # Beta(a, a) = X / (X + Y) where X, Y ~ Gamma(a, 1) iid.
    # We use torch.distributions for portability.
    beta_dist = torch.distributions.Beta(torch.tensor([alpha], device=device),
                                          torch.tensor([alpha], device=device))
    lam = beta_dist.sample().item()
    # Clamp away from exactly 0/1 to keep both originals contributing.
    return max(min(lam, 0.999), 0.001)


def mixup_latents(
    mu: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mix latents and labels for two random samples.

    Args:
        mu: [B, C, D, H, W] latent mean tensor (B must be >= 2).
        labels: [B] integer class labels.
        alpha: Beta distribution parameter.

    Returns:
        mu_mixed: [B, C, D, H, W] mixed latents.
        labels_a: [B] original labels (first half).
        labels_b: [B] mixed-in labels (second half).
        lam: the mixing coefficient actually used.
    """
    if mu.size(0) < 2:
        # Can't mixup a single sample. Return originals.
        return mu, labels, labels, 1.0

    lam = sample_mixup_lambda(alpha, mu.device)
    # Shuffle within the batch
    perm = torch.randperm(mu.size(0), device=mu.device)
    mu_mixed = lam * mu + (1.0 - lam) * mu[perm]
    labels_a = labels
    labels_b = labels[perm]
    return mu_mixed, labels_a, labels_b, lam


def mixup_classification_loss(
    cls_logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Classification loss with mixup.

    Loss = lam * CE(logits, a) + (1-lam) * CE(logits, b)
    """
    if labels_a is None or labels_b is None:
        return torch.tensor(0.0, device=cls_logits.device)
    loss_a = torch.nn.functional.cross_entropy(cls_logits, labels_a)
    loss_b = torch.nn.functional.cross_entropy(cls_logits, labels_b)
    return lam * loss_a + (1.0 - lam) * loss_b


def mixup_regression_loss(
    value: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Generic regression loss with mixup (used for recon and ordinal_reg)."""
    if target_a is None or target_b is None:
        return torch.tensor(0.0, device=value.device)
    loss_a = (value - target_a).abs().mean()
    loss_b = (value - target_b).abs().mean()
    return lam * loss_a + (1.0 - lam) * loss_b
