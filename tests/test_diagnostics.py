"""
Smoke tests for the deep diagnostic utilities in utils/diagnostics.py.

Verifies:
  A) silhouette_score is +1 for well-separated clusters, ~0 for random
  B) per_class_centroid_distance matches manual computation
  C) per_dim_latent_stats correctly flags collapsed vs active dims
  D) per_class_pred_frequency sums to ~1.0
  E) grad_norm_by_module groups params correctly
  F) recon_intensity_stats returns finite values
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import math
import torch
import torch.nn as nn

from utils.diagnostics import (
    silhouette_score,
    per_class_centroid_distance,
    per_class_pred_frequency,
    per_dim_latent_stats,
    grad_norm_by_module,
    recon_intensity_stats,
)


def test_A_silhouette() -> None:
    """A) Silhouette: +1 for well-separated, ~0 for random overlap."""
    print("\n[Test A] silhouette_score")
    # Case 1: two well-separated clusters (4 samples each)
    cluster_a = torch.randn(4, 8) + 10.0  # far from origin
    cluster_b = torch.randn(4, 8) - 10.0
    mu = torch.cat([cluster_a, cluster_b], dim=0)
    labels = torch.tensor([0] * 4 + [1] * 4)
    sil_sep = silhouette_score(mu, labels)
    assert sil_sep > 0.8, f"well-separated should be > 0.8, got {sil_sep}"
    print(f"  [OK] well-separated clusters: sil={sil_sep:.3f} (expect ~1)")

    # Case 2: heavily overlapping
    overlap_a = torch.randn(10, 8)
    overlap_b = torch.randn(10, 8) + 0.1
    mu2 = torch.cat([overlap_a, overlap_b], dim=0)
    labels2 = torch.tensor([0] * 10 + [1] * 10)
    sil_overlap = silhouette_score(mu2, labels2)
    assert -0.3 < sil_overlap < 0.5, f"overlapping should be near 0, got {sil_overlap}"
    print(f"  [OK] overlapping clusters: sil={sil_overlap:.3f} (expect ~0)")

    # Edge case: single sample
    sil_single = silhouette_score(torch.randn(1, 4), torch.tensor([0]))
    assert sil_single == 0.0
    print(f"  [OK] single-sample edge case: sil={sil_single:.3f}")


def test_B_centroid_distance() -> None:
    """B) Centroid distances: known distances match manual computation."""
    print("\n[Test B] per_class_centroid_distance")
    # Class centroids on the unit circle: 0=(1,0), 1=(0,1), 2=(-1,0), 3=(0,-1)
    # Adjacent pairs (c01, c12, c23, c30) are at distance sqrt(2)
    # Opposite pairs (c02, c13) are at distance 2
    mu = torch.tensor([
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],   # class 0 centroid (1, 0)
        [0.0, 1.0], [0.0, 1.0],                 # class 1 centroid (0, 1)
        [-1.0, 0.0], [-1.0, 0.0],               # class 2 centroid (-1, 0)
        [0.0, -1.0],                            # class 3 centroid (0, -1)
    ])
    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3])
    cd = per_class_centroid_distance(mu, labels, num_classes=4)
    expected = {
        "c01_dist": math.sqrt(2.0),
        "c02_dist": 2.0,
        "c03_dist": math.sqrt(2.0),
        "c12_dist": math.sqrt(2.0),
        "c13_dist": 2.0,
        "c23_dist": math.sqrt(2.0),
    }
    for k, exp_v in expected.items():
        got = cd.get(k, 0.0)
        assert abs(got - exp_v) < 1e-4, f"{k}: expected {exp_v:.4f}, got {got:.4f}"
    print(f"  [OK] 6 pair distances verified (adjacent sqrt(2), opposite 2.0)")


def test_C_per_dim_stats() -> None:
    """C) per_dim_latent_stats: collapse detection works."""
    print("\n[Test C] per_dim_latent_stats")
    N, C = 50, 8
    mu = torch.zeros(N, C)
    # First 4 dims: active (std > 0.1)
    mu[:, :4] = torch.randn(N, 4)
    # Last 4 dims: collapsed (std ≈ 0)
    # Threshold: sqrt(2*free_bits + 1e-6) with free_bits=0.0 ~ 0.001
    stats = per_dim_latent_stats(mu, free_bits=0.0)
    assert stats["n_active_dims"] == 4, f"expected 4 active dims, got {stats['n_active_dims']}"
    assert stats["n_collapsed_dims"] == 4, f"expected 4 collapsed, got {stats['n_collapsed_dims']}"
    assert abs(stats["latent_std_mean"] - 0.5) < 0.1, f"expected ~0.5 mean, got {stats['latent_std_mean']}"
    print(f"  [OK] 4 active + 4 collapsed correctly identified")
    print(f"       active={stats['n_active_dims']}, mean_std={stats['latent_std_mean']:.3f}")


def test_D_pred_frequency() -> None:
    """D) per_class_pred_frequency: normalized histogram."""
    print("\n[Test D] per_class_pred_frequency")
    # 7 preds: 3 zeros, 2 ones, 1 two, 1 three -> 3/7, 2/7, 1/7, 1/7
    preds = torch.tensor([0, 0, 0, 1, 1, 2, 3])
    freq = per_class_pred_frequency(preds, num_classes=4)
    assert abs(freq[0] - 3 / 7) < 1e-5
    assert abs(freq[1] - 2 / 7) < 1e-5
    assert abs(freq[2] - 1 / 7) < 1e-5
    assert abs(freq[3] - 1 / 7) < 1e-5
    assert abs(sum(freq) - 1.0) < 1e-5
    print(f"  [OK] freq={[round(f, 3) for f in freq]}, sum={sum(freq):.3f}")


def test_E_grad_norm() -> None:
    """E) grad_norm_by_module groups params by name substring."""
    print("\n[Test E] grad_norm_by_module")
    # Tiny model with named params matching our default groups
    encoder = nn.Linear(4, 4)
    decoder = nn.Linear(4, 4)
    classifier = nn.Linear(4, 2)
    encoder.weight.grad = torch.ones_like(encoder.weight) * 0.5  # norm = sqrt(16)*0.5 = 2
    encoder.bias.grad = torch.ones_like(encoder.bias) * 0.5       # norm = sqrt(4)*0.5 = 1
    decoder.weight.grad = torch.ones_like(decoder.weight) * 0.1  # norm = 0.4
    classifier.weight.grad = torch.ones_like(classifier.weight)  # norm = sqrt(8)
    # Build a parent module
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_t1 = encoder
            self.decoder = decoder
            self.classifier = classifier
    m = M()
    gn = grad_norm_by_module(m, module_names=["encoder_t1", "decoder", "classifier"])
    # encoder_t1 = sqrt(16)*0.5 + sqrt(4)*0.5 = 2 + 1 = 3
    assert abs(gn["encoder_t1_grad"] - 3.0) < 1e-4, f"encoder_t1: {gn['encoder_t1_grad']}"
    # decoder = sqrt(16)*0.1 = 0.4
    assert abs(gn["decoder_grad"] - 0.4) < 1e-4, f"decoder: {gn['decoder_grad']}"
    # classifier = sqrt(8) ~ 2.828
    assert abs(gn["classifier_grad"] - math.sqrt(8)) < 1e-3, f"classifier: {gn['classifier_grad']}"
    print(f"  [OK] encoder_t1={gn['encoder_t1_grad']:.3f}, "
          f"decoder={gn['decoder_grad']:.3f}, classifier={gn['classifier_grad']:.3f}")


def test_F_recon_stats() -> None:
    """F) recon_intensity_stats: returns finite values."""
    print("\n[Test F] recon_intensity_stats")
    images = torch.rand(2, 1, 16, 16, 12) * 0.5 + 0.25  # values in [0.25, 0.75], mean ~0.50
    recon = images + torch.randn_like(images) * 0.01     # near-perfect recon
    s = recon_intensity_stats(images, recon)
    assert 0.45 < s["input_mean"] < 0.55, f"input_mean out of [0.45, 0.55]: {s['input_mean']}"
    assert math.isfinite(s["recon_relative_error"])
    assert s["recon_relative_error"] < 0.1, f"good recon should have small rel error, got {s['recon_relative_error']}"
    print(f"  [OK] input_mean={s['input_mean']:.3f}, recon_mean={s['recon_mean']:.3f}, "
          f"rel_err={s['recon_relative_error']:.4f}")


def main() -> None:
    print("=" * 60)
    print("Deep Diagnostic Tests (silhouette, centroids, grad-norms, ...)")
    print("=" * 60)
    tests = [
        ("A: silhouette_score", test_A_silhouette),
        ("B: centroid_distance", test_B_centroid_distance),
        ("C: per_dim_stats", test_C_per_dim_stats),
        ("D: pred_frequency", test_D_pred_frequency),
        ("E: grad_norm_by_module", test_E_grad_norm),
        ("F: recon_stats", test_F_recon_stats),
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            raise
    print("\n" + "=" * 60)
    print(f"All {len(tests)} tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
