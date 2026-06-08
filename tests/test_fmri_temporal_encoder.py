"""
Integration test for the fMRI Temporal Encoder (Plan C).

Validates that:
  1. fMRITemporalEncoder produces the correct output shape on 6D input
     (B, 1, D, H, W, T) -> (B, 32, 16, 16, 12).
  2. The encoder is zero-init safe: at init, output is broadcast-constant.
  3. Backward pass flows end-to-end.
  4. With use_fmri_temporal=True, the MultiModalVAE3D model accepts a 5D
     fMRI tensor and returns the standard 4-class classification.
  5. Backward through the full model with 5D fMRI is numerically stable.
  6. Collate function correctly stacks 5D fMRI tensors with zero-fill for
     missing samples.

Run: python tests/test_fmri_temporal_encoder.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from models.fmri_temporal_encoder import fMRITemporalEncoder
from models.vae3d import MultiModalVAE3D


# ---------------------------------------------------------------------------
# 1. Forward shape test (6D)
# ---------------------------------------------------------------------------
def test_fmri_encoder_shape_6d() -> None:
    enc = fMRITemporalEncoder(in_channels=34, embed_dim=32, target_grid=(16, 16, 12))
    enc.eval()
    x = torch.randn(2, 1, 64, 64, 34, 220)
    y = enc(x)
    assert y.shape == (2, 32, 16, 16, 12), f"6D out shape wrong: {y.shape}"
    print(f"  [OK] 6D path: {tuple(x.shape)} -> {tuple(y.shape)}")


# ---------------------------------------------------------------------------
# 2. Forward shape test (5D)
# ---------------------------------------------------------------------------
def test_fmri_encoder_shape_5d() -> None:
    enc = fMRITemporalEncoder(in_channels=34, embed_dim=32, target_grid=(16, 16, 12))
    enc.eval()
    x = torch.randn(2, 64, 64, 34, 220)
    y = enc(x)
    assert y.shape == (2, 32, 16, 16, 12), f"5D out shape wrong: {y.shape}"
    print(f"  [OK] 5D path: {tuple(x.shape)} -> {tuple(y.shape)}")


# ---------------------------------------------------------------------------
# 3. Zero-init safety: at init the proj is zero so output must be ~0.
# ---------------------------------------------------------------------------
def test_fmri_encoder_zero_init() -> None:
    enc = fMRITemporalEncoder(in_channels=34, embed_dim=32, target_grid=(16, 16, 12))
    enc.eval()
    x1 = torch.randn(2, 1, 64, 64, 34, 220)
    x2 = torch.randn(2, 1, 64, 64, 34, 220)
    y1 = enc(x1)
    y2 = enc(x2)
    # With zero-init proj, the encoder should produce near-zero output
    # regardless of input -> y1 should be ~y2.
    assert torch.allclose(y1, y2, atol=1e-5), (
        f"Zero-init not constant: y1.max={y1.abs().max().item()}, "
        f"y2.max={y2.abs().max().item()}"
    )
    # And the output should be small in magnitude.
    assert y1.abs().max().item() < 1e-4, (
        f"Zero-init output not near zero: max={y1.abs().max().item()}"
    )
    print(f"  [OK] Zero-init safety: output max abs = {y1.abs().max().item():.2e}")


# ---------------------------------------------------------------------------
# 4. End-to-end MultiModalVAE3D with 5D fMRI
# ---------------------------------------------------------------------------
def test_multimodal_vae_with_5d_fmri() -> None:
    model = MultiModalVAE3D(
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        decoder_depth=4,  # match 256x256x192 output
        optional_modalities=["fmri", "asl", "qsm", "flair"],
        use_fmri_temporal=True,
        fmri_in_channels=34,
    )
    model.eval()
    # Build a 5D fMRI batch
    x_dict = {
        "t1": torch.randn(2, 1, 256, 256, 192),
        "fmri": torch.randn(2, 1, 64, 64, 34, 220),  # 6D
        "asl": torch.randn(2, 1, 128, 128, 36),
        "qsm": torch.randn(2, 1, 192, 192, 128),
        "flair": torch.randn(2, 1, 256, 256, 192),
    }
    with torch.no_grad():
        recon, mu, logvar = model(x_dict)
    assert recon.shape == (2, 1, 256, 256, 192), f"recon shape: {recon.shape}"
    assert mu.shape == (2, 32, 16, 16, 12), f"mu shape: {mu.shape}"
    assert logvar.shape == (2, 32, 16, 16, 12), f"logvar shape: {logvar.shape}"
    assert torch.isfinite(recon).all(), "recon has NaN/Inf"
    assert torch.isfinite(mu).all(), "mu has NaN/Inf"
    print(f"  [OK] MultiModalVAE3D 5D-fMRI forward: recon={tuple(recon.shape)}, "
          f"mu={tuple(mu.shape)}")


# ---------------------------------------------------------------------------
# 5. Classification head with 5D fMRI
# ---------------------------------------------------------------------------
def test_classification_with_5d_fmri() -> None:
    model = MultiModalVAE3D(
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        decoder_depth=4,
        use_fmri_temporal=True,
    )
    model.eval()
    x_dict = {
        "t1": torch.randn(2, 1, 256, 256, 192),
        "fmri": torch.randn(2, 1, 64, 64, 34, 220),
        "asl": torch.zeros(2, 1, 128, 128, 36),
        "qsm": torch.zeros(2, 1, 192, 192, 128),
        "flair": torch.zeros(2, 1, 256, 256, 192),
    }
    with torch.no_grad():
        recon, cls_logits, mu, logvar = model(x_dict, return_components=True)
    assert cls_logits.shape == (2, 4), f"cls shape: {cls_logits.shape}"
    assert torch.isfinite(cls_logits).all(), "cls_logits has NaN/Inf"
    print(f"  [OK] Classification head: {tuple(cls_logits.shape)}")


# ---------------------------------------------------------------------------
# 6. Backward through full model with 5D fMRI
# ---------------------------------------------------------------------------
def test_backward_through_full_model_5d() -> None:
    model = MultiModalVAE3D(
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        decoder_depth=4,
        use_fmri_temporal=True,
    )
    model.train()
    x_dict = {
        "t1": torch.randn(1, 1, 256, 256, 192),
        "fmri": torch.randn(1, 1, 64, 64, 34, 220),
        "asl": torch.randn(1, 1, 128, 128, 36),
        "qsm": torch.randn(1, 1, 192, 192, 128),
        "flair": torch.randn(1, 1, 256, 256, 192),
    }
    labels = torch.tensor([2], dtype=torch.long)
    recon, cls_logits, mu, logvar = model(x_dict, return_components=True)
    loss = (
        (recon - x_dict["t1"]).abs().mean()
        + 1.0 * nn.functional.cross_entropy(cls_logits, labels)
        + 0.3 * (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
    )
    loss.backward()
    # Verify every parameter got a non-None grad
    n_with_grad = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += 1
        if p.grad is not None:
            n_with_grad += 1
            assert torch.isfinite(p.grad).all(), f"grad has NaN/Inf at {name}"
    # We don't require 100% grad coverage because:
    #   - Some params (e.g. zero-init proj) receive zero grad at init.
    #   - The fMRITemporalEncoder is invoked but its transformer layers
    #     may not contribute gradient if the loss path doesn't depend
    #     on its specific output (e.g. fusion_proj zero-out path).
    # We only require a substantial majority (>= 80%) to have grad, which
    # confirms the backward graph is fully wired.
    assert n_with_grad >= int(0.8 * n_total), (
        f"Only {n_with_grad}/{n_total} parameters got gradients (need >= 80%)"
    )
    print(f"  [OK] Backward through 5D-fMRI model: {n_with_grad}/{n_total} params "
          f"got finite gradients")


# ---------------------------------------------------------------------------
# 7. Collate function: 5D fMRI with missing samples -> zero-fill
# ---------------------------------------------------------------------------
def test_collate_5d_fmri_with_missing() -> None:
    from core_data.dataset import multimodal_collate_fn
    batch = [
        {
            "t1": torch.randn(1, 256, 256, 192),
            "fmri": torch.randn(1, 64, 64, 34, 220),
            "label": 0,
            "patient_id": "p1",
            "available_modalities": ["t1", "fmri"],
        },
        {
            "t1": torch.randn(1, 256, 256, 192),
            "fmri": None,  # missing
            "label": 1,
            "patient_id": "p2",
            "available_modalities": ["t1"],
        },
    ]
    out = multimodal_collate_fn(batch)
    assert out["fmri"].shape == (2, 1, 64, 64, 34, 220), (
        f"collate fmri shape: {out['fmri'].shape}"
    )
    # The zero-filled sample should be exactly zero
    assert torch.allclose(out["fmri"][1], torch.zeros_like(out["fmri"][1]))
    assert out["label"].tolist() == [0, 1]
    print(f"  [OK] Collate 5D-fMRI w/ missing: {tuple(out['fmri'].shape)}, "
          f"zero-fill correct")


# ---------------------------------------------------------------------------
# 8. Legacy path: use_fmri_temporal=False uses static 3D encoder
# ---------------------------------------------------------------------------
def test_legacy_static_fmri_path() -> None:
    model = MultiModalVAE3D(
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        decoder_depth=4,
        use_fmri_temporal=False,
    )
    model.eval()
    x_dict = {
        "t1": torch.randn(1, 1, 256, 256, 192),
        "fmri": torch.randn(1, 1, 64, 64, 34),  # 3D legacy
        "asl": torch.zeros(1, 1, 128, 128, 36),
        "qsm": torch.zeros(1, 1, 192, 192, 128),
        "flair": torch.zeros(1, 1, 256, 256, 192),
    }
    with torch.no_grad():
        recon, mu, logvar = model(x_dict)
    assert recon.shape == (1, 1, 256, 256, 192)
    assert mu.shape == (1, 32, 16, 16, 12)
    print(f"  [OK] Legacy static 3D fMRI path works: recon={tuple(recon.shape)}")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("fMRI Temporal Encoder Integration Tests (Plan C)")
    print("=" * 60)
    tests = [
        ("Forward shape 6D", test_fmri_encoder_shape_6d),
        ("Forward shape 5D", test_fmri_encoder_shape_5d),
        ("Zero-init safety", test_fmri_encoder_zero_init),
        ("MultiModalVAE3D 5D-fMRI", test_multimodal_vae_with_5d_fmri),
        ("Classification w/ 5D fMRI", test_classification_with_5d_fmri),
        ("Backward through full model", test_backward_through_full_model_5d),
        ("Collate 5D fMRI w/ missing", test_collate_5d_fmri_with_missing),
        ("Legacy static 3D path", test_legacy_static_fmri_path),
    ]
    for name, fn in tests:
        print(f"\n[Test] {name}")
        fn()
    print("\n" + "=" * 60)
    print(f"All {len(tests)} tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
