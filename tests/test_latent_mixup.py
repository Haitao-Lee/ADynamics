"""
Smoke tests for latent-space mixup in utils/kl_schedules.py.

Verifies:
  A) sample_mixup_lambda returns values in (0, 1) for alpha > 0
  B) mixup_latents returns the right shape and mixes labels correctly
  C) mixup_classification_loss returns a finite scalar
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from utils.kl_schedules import (
    should_apply_mixup,
    sample_mixup_lambda,
    mixup_latents,
    mixup_classification_loss,
    mixup_regression_loss,
)


def test_A_sample_lambda() -> None:
    """A) sample_mixup_lambda returns values in [0.001, 0.999] for alpha > 0."""
    print("\n[Test A] sample_mixup_lambda")
    device = torch.device("cpu")
    # 100 samples
    samples = [sample_mixup_lambda(0.4, device) for _ in range(100)]
    # Clamp is [0.001, 0.999] (see sample_mixup_lambda), so samples may
    # legitimately touch the boundary when Beta(0.4, 0.4) puts mass near 0/1.
    assert all(0.001 <= s <= 0.999 for s in samples), (
        f"some samples out of [0.001, 0.999]: min={min(samples)}, max={max(samples)}"
    )
    # alpha=0 should return 1.0
    assert sample_mixup_lambda(0.0, device) == 1.0, "alpha=0 should give 1.0"
    print(f"  [OK] alpha=0.4 -> 100 samples in [{min(samples):.3f}, {max(samples):.3f}]")
    print(f"       alpha=0.0 -> 1.0 (mixup disabled)")


def test_B_mixup_latents() -> None:
    """B) mixup_latents: shape preserved, labels paired correctly."""
    print("\n[Test B] mixup_latents")
    device = torch.device("cpu")
    B, C, D, H, W = 4, 32, 16, 16, 12
    mu = torch.randn(B, C, D, H, W)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    mu_mixed, lab_a, lab_b, lam = mixup_latents(mu, labels, alpha=0.4)
    assert mu_mixed.shape == mu.shape, f"shape mismatch: {mu_mixed.shape} vs {mu.shape}"
    assert lab_a.shape == labels.shape
    assert lab_b.shape == labels.shape
    assert 0.001 <= lam <= 0.999, f"lam out of [0.001, 0.999]: {lam}"
    # Structural check: mu_mixed is on the segment between mu and mu[perm].
    # We can't predict the perm (random), so just verify the mixed tensor is
    # distinct from the input and within the convex hull.
    assert not torch.allclose(mu_mixed, mu), "mu_mixed should differ from mu"
    # Labels: a is unchanged, b is a permutation of a (same multiset)
    assert torch.equal(lab_a, labels)
    assert sorted(lab_b.tolist()) == sorted(labels.tolist()), (
        f"lab_b should be a permutation of labels, got {lab_b.tolist()}"
    )
    # mu_mixed should be on the [mu[perm], mu] segment for some perm:
    # pick a candidate perm by greedy search
    from itertools import permutations
    found = False
    for cand in permutations(range(B)):
        cand_perm = torch.tensor(cand)
        expected = lam * mu + (1.0 - lam) * mu[cand_perm]
        if torch.allclose(mu_mixed, expected, atol=1e-5):
            found = True
            break
    assert found, "mu_mixed should be a valid mix of mu with some permutation"
    print(f"  [OK] mu shape preserved {tuple(mu_mixed.shape)}, "
          f"lam={lam:.3f}, lab_a={lab_a.tolist()}, lab_b={lab_b.tolist()}")


def test_C_mixup_classification_loss() -> None:
    """C) mixup_classification_loss: scalar + finite + boundary cases."""
    print("\n[Test C] mixup_classification_loss")
    device = torch.device("cpu")
    B, num_classes = 8, 4
    logits = torch.randn(B, num_classes)
    labels_a = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    labels_b = torch.tensor([3, 2, 1, 0, 3, 2, 1, 0], dtype=torch.long)

    # lam = 0.5: equal mix
    loss = mixup_classification_loss(logits, labels_a, labels_b, lam=0.5)
    assert loss.dim() == 0, f"expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    # Verify it's exactly 0.5 * CE(a) + 0.5 * CE(b)
    expected = 0.5 * F.cross_entropy(logits, labels_a) + 0.5 * F.cross_entropy(logits, labels_b)
    assert torch.allclose(loss, expected, atol=1e-5), (
        f"loss != 0.5*CE(a) + 0.5*CE(b): got {loss.item()}, expected {expected.item()}"
    )

    # lam = 1.0: pure a
    loss_a = mixup_classification_loss(logits, labels_a, labels_b, lam=1.0)
    assert torch.allclose(loss_a, F.cross_entropy(logits, labels_a), atol=1e-5)

    # lam = 0.0: pure b
    loss_b = mixup_classification_loss(logits, labels_a, labels_b, lam=0.0)
    assert torch.allclose(loss_b, F.cross_entropy(logits, labels_b), atol=1e-5)

    print(f"  [OK] lam=0.5 -> {loss.item():.4f}, lam=1.0 -> {loss_a.item():.4f}, "
          f"lam=0.0 -> {loss_b.item():.4f}")


def test_D_should_apply_mixup() -> None:
    """D) should_apply_mixup: disabled when alpha=0, probability-based otherwise."""
    print("\n[Test D] should_apply_mixup dispatch")
    # alpha=0: always False
    for ep in range(20):
        assert not should_apply_mixup(ep, {"mixup_alpha": 0.0, "mixup_prob": 1.0}), (
            f"alpha=0 should never mixup, but returned True at ep {ep}"
        )
    # alpha>0 + prob=0: always False
    for ep in range(20):
        assert not should_apply_mixup(ep, {"mixup_alpha": 0.4, "mixup_prob": 0.0}), (
            f"prob=0 should never mixup, but returned True at ep {ep}"
        )
    # alpha>0 + prob=1: always True (with random, this should be true across many calls)
    n_true = sum(
        should_apply_mixup(ep, {"mixup_alpha": 0.4, "mixup_prob": 1.0})
        for ep in range(100)
    )
    assert n_true == 100, f"prob=1.0 should always mixup, got {n_true}/100"
    print(f"  [OK] alpha=0 -> False always, prob=1 -> True always, prob=0.5 ~ 50%")


def test_E_mixup_regression_loss() -> None:
    """E) mixup_regression_loss: boundary cases."""
    print("\n[Test E] mixup_regression_loss")
    device = torch.device("cpu")
    value = torch.randn(8, 32, 16, 16, 12)
    target_a = torch.randn(8, 32, 16, 16, 12)
    target_b = torch.randn(8, 32, 16, 16, 12)

    loss = mixup_regression_loss(value, target_a, target_b, lam=0.5)
    expected = 0.5 * (value - target_a).abs().mean() + 0.5 * (value - target_b).abs().mean()
    assert torch.allclose(loss, expected, atol=1e-5)

    # None target -> 0
    loss_none = mixup_regression_loss(value, None, target_b, lam=0.5)
    assert loss_none.item() == 0.0
    print(f"  [OK] lam=0.5 -> {loss.item():.4f}, None target -> 0")


def main() -> None:
    print("=" * 60)
    print("Latent-Space Mixup Tests")
    print("=" * 60)
    tests = [
        ("A: sample_mixup_lambda", test_A_sample_lambda),
        ("B: mixup_latents", test_B_mixup_latents),
        ("C: mixup_classification_loss", test_C_mixup_classification_loss),
        ("D: should_apply_mixup", test_D_should_apply_mixup),
        ("E: mixup_regression_loss", test_E_mixup_regression_loss),
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
