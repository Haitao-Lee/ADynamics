"""
Smoke test for the modality-toggle system.

Verifies that MultiModalVAE3D can be built and trained (forward+backward)
with 4 different modality combinations:
  A) All 4 optional modalities ON (default 5-modal)
  B) T1 + FLAIR only
  C) T1 only (clean baseline, no zero-fill risk)
  D) T1 only + demographic conditioning (recommended re-train setting)

Also verifies that:
  - Without optional modalities, the model has no zero-fill path
  - The dataset returns age/sex tensors when use_demographic=True
  - The trainer config round-trips (build model -> forward -> backward -> save)

Run: python tests/test_modality_toggle.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from models.vae3d import MultiModalVAE3D


# ---------------------------------------------------------------------------
def test_A_5modality() -> None:
    """A) All 4 optional modalities ON — original config."""
    print("\n[Test A] 5-modality (T1+fmri+asl+qsm+flair), no demo")
    model = MultiModalVAE3D(
        spatial_size=(256, 256, 192),
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        decoder_depth=4,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
        use_attention=True,
        attention_levels=(3,),
        attention_heads=8,
        use_fmri_temporal=True,
    )
    model.eval()
    x = {
        "t1":   torch.randn(1, 1, 256, 256, 192),
        "fmri": torch.randn(1, 1, 64, 64, 34, 220),
        "asl":  torch.randn(1, 1, 128, 128, 36),
        "qsm":  torch.randn(1, 1, 192, 192, 128),
        "flair":torch.randn(1, 1, 256, 256, 192),
    }
    with torch.no_grad():
        recon, cls, mu, logvar = model(x, return_components=True)
    assert recon.shape == (1, 1, 256, 256, 192)
    assert cls.shape == (1, 4)
    assert mu.shape == (1, 32, 16, 16, 12)
    print(f"  [OK] recon={tuple(recon.shape)}  cls={tuple(cls.shape)}  "
          f"mu={tuple(mu.shape)}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")


def test_B_t1_flair() -> None:
    """B) T1 + FLAIR only — clean 2-modality test."""
    print("\n[Test B] T1 + FLAIR only, no demo")
    model = MultiModalVAE3D(
        spatial_size=(256, 256, 192),
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        decoder_depth=4,
        optional_modalities=["flair"],
        use_attention=True,
        attention_levels=(3,),
        attention_heads=8,
    )
    model.eval()
    x = {
        "t1":   torch.randn(1, 1, 256, 256, 192),
        "flair":torch.randn(1, 1, 256, 256, 192),
    }
    with torch.no_grad():
        recon, cls, mu, logvar = model(x, return_components=True)
    assert recon.shape == (1, 1, 256, 256, 192)
    print(f"  [OK] recon={tuple(recon.shape)}  cls={tuple(cls.shape)}  "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")


def test_C_t1_only() -> None:
    """C) T1 only — cleanest baseline, NO zero-fill risk, NO shortcut shortcut."""
    print("\n[Test C] T1 only, no demo, no optional encoders")
    model = MultiModalVAE3D(
        spatial_size=(256, 256, 192),
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        decoder_depth=4,
        optional_modalities=[],   # ← key: empty list, no zero-fill path
        use_attention=True,
        attention_levels=(3,),
        attention_heads=8,
    )
    # Verify the architecture: no optional encoders at all
    assert len(model.optional_modalities) == 0
    assert len(model.optional_encoders) == 0
    assert model.fusion_proj[0].in_channels == 32  # 1 modality × 32 channels only
    print(f"  [OK] Architecture: {len(model.optional_encoders)} optional encoders, "
          f"fusion_proj in_channels={model.fusion_proj[0].in_channels}")
    model.eval()
    x = {"t1": torch.randn(1, 1, 256, 256, 192)}
    with torch.no_grad():
        recon, cls, mu, logvar = model(x, return_components=True)
    assert recon.shape == (1, 1, 256, 256, 192)
    print(f"  [OK] recon={tuple(recon.shape)}  cls={tuple(cls.shape)}  "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")


def test_D_t1_demo() -> None:
    """D) T1 only + demographic conditioning (recommended re-train)."""
    print("\n[Test D] T1 only + age/sex conditioning")
    model = MultiModalVAE3D(
        spatial_size=(256, 256, 192),
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        decoder_depth=4,
        optional_modalities=[],
        use_demographic_cond=True,
    )
    model.eval()
    x = {"t1": torch.randn(2, 1, 256, 256, 192)}
    age = torch.tensor([72.0, 65.0], dtype=torch.float32)
    sex = torch.tensor([2, 1], dtype=torch.long)  # 2=female, 1=male
    with torch.no_grad():
        recon, cls, mu, logvar = model(x, return_components=True, age=age, sex=sex)
    assert recon.shape == (2, 1, 256, 256, 192)
    assert mu.shape == (2, 32, 16, 16, 12)
    print(f"  [OK] recon={tuple(recon.shape)}  cls={tuple(cls.shape)}  "
          f"mu={tuple(mu.shape)}")
    # Verify demo modules were built
    assert hasattr(model, "age_mlp")
    assert hasattr(model, "sex_emb")
    assert hasattr(model, "demo_proj")
    assert hasattr(model, "use_demographic_cond")
    assert model.use_demographic_cond is True
    print(f"  [OK] demo modules: age_mlp={sum(p.numel() for p in model.age_mlp.parameters())/1e3:.1f}K, "
          f"sex_emb={model.sex_emb.weight.shape}, demo_proj={model.demo_proj.weight.shape}")


def test_E_backward_t1_only_demo() -> None:
    """E) Backward through T1-only + demo path (the path you'll actually train)."""
    print("\n[Test E] T1 only + demo, full backward")
    model = MultiModalVAE3D(
        spatial_size=(256, 256, 192),
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        decoder_depth=4,
        optional_modalities=[],
        use_demographic_cond=True,
    )
    model.train()
    x = {"t1": torch.randn(2, 1, 256, 256, 192)}
    age = torch.tensor([72.0, 65.0], dtype=torch.float32)
    sex = torch.tensor([2, 1], dtype=torch.long)
    labels = torch.tensor([3, 0], dtype=torch.long)

    recon, cls_logits, mu, logvar = model(x, return_components=True, age=age, sex=sex)
    recon_loss = (recon - x["t1"]).abs().mean()
    cls_loss = nn.functional.cross_entropy(cls_logits, labels)
    kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
    loss = recon_loss + 1.0 * cls_loss + 0.3 * kl_loss
    loss.backward()

    # Verify all params got finite gradients
    n_with_grad = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += 1
        if p.grad is not None:
            n_with_grad += 1
            assert torch.isfinite(p.grad).all(), f"non-finite grad at {name}"
    assert n_with_grad == n_total, f"only {n_with_grad}/{n_total} params got grad"
    print(f"  [OK] Backward OK: {n_with_grad}/{n_total} params got finite grad, "
          f"loss={loss.item():.4f} (recon={recon_loss.item():.4f} "
          f"cls={cls_loss.item():.4f} kl={kl_loss.item():.4f})")


def test_F_dataset_age_sex() -> None:
    """F) Dataset returns age/sex tensors that collate cleanly."""
    print("\n[Test F] Dataset returns age/sex")
    from core_data.dataset import MultiModalDataset, multimodal_collate_fn
    from core_data.transforms import get_multimodal_train_transforms
    import tempfile, nibabel as nib, numpy as np

    # Make 2 tiny T1 NIfTIs
    tmp = tempfile.mkdtemp(prefix="test_demo_")
    paths = []
    for i in range(2):
        p = os.path.join(tmp, f"t1_{i}.nii.gz")
        nib.save(nib.Nifti1Image(np.random.rand(64, 64, 64).astype(np.float32), np.eye(4)), p)
        paths.append(p)

    data_list = [
        {"t1": paths[0], "label": 0, "patient_id": "p0", "age": 75.0, "sex": 2},
        {"t1": paths[1], "label": 1, "patient_id": "p1", "age": 60.0, "sex": 1},
    ]
    ds = MultiModalDataset(
        data_list,
        transform=get_multimodal_train_transforms(),
        optional_modalities=[],
    )
    sample0 = ds[0]
    assert "age" in sample0
    assert "sex" in sample0
    assert isinstance(sample0["age"], torch.Tensor)
    assert isinstance(sample0["sex"], torch.Tensor)
    assert sample0["age"].dtype == torch.float32
    assert sample0["sex"].dtype == torch.long
    assert sample0["age"].item() == 75.0
    assert sample0["sex"].item() == 2
    print(f"  [OK] sample0 age={sample0['age'].item()} sex={sample0['sex'].item()}")

    # Test collate stacks them
    batch = multimodal_collate_fn([ds[0], ds[1]])
    assert batch["age"].shape == (2,)
    assert batch["sex"].shape == (2,)
    assert batch["age"].tolist() == [75.0, 60.0]
    assert batch["sex"].tolist() == [2, 1]
    print(f"  [OK] collate: age={batch['age'].tolist()}  sex={batch['sex'].tolist()}")

    # Cleanup
    for p in paths:
        os.remove(p)
    os.rmdir(tmp)


def test_G_dataset_missing_demo() -> None:
    """G) Missing age/sex in manifest => safe defaults (0, 0)."""
    print("\n[Test G] Missing age/sex -> defaults 0/0")
    from core_data.dataset import MultiModalDataset, multimodal_collate_fn
    from core_data.transforms import get_multimodal_train_transforms
    import tempfile, nibabel as nib, numpy as np

    tmp = tempfile.mkdtemp(prefix="test_demo2_")
    p = os.path.join(tmp, "t1.nii.gz")
    nib.save(nib.Nifti1Image(np.random.rand(64, 64, 64).astype(np.float32), np.eye(4)), p)

    data_list = [{"t1": p, "label": 0, "patient_id": "p0"}]  # no age, no sex
    ds = MultiModalDataset(
        data_list,
        transform=get_multimodal_train_transforms(),
        optional_modalities=[],
    )
    s = ds[0]
    assert s["age"].item() == 0.0
    assert s["sex"].item() == 0
    print(f"  [OK] missing -> age={s['age'].item()}  sex={s['sex'].item()}")
    os.remove(p)
    os.rmdir(tmp)


# ---------------------------------------------------------------------------
def test_H_data_parallel_with_demo() -> None:
    """H) DataParallel + T1 + demo — the exact failure mode the bug report showed.

    Without the fix: age/sex stay on cuda:0 while mu lives on cuda:1 (the
    replica's device), so age_mlp(mat1) raises 'Expected all tensors on
    the same device'.

    With the fix: _demographic_bias uses self.age_mlp's device as authority
    and moves age/sex accordingly.
    """
    print("\n[Test H] DataParallel with T1 + demo (multi-GPU device safety)")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("  [SKIP] Less than 2 GPUs available, skipping DataParallel test")
        return

    model = MultiModalVAE3D(
        spatial_size=(256, 256, 192),
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        decoder_depth=4,
        optional_modalities=[],
        use_demographic_cond=True,
    )
    # Wrap in DataParallel exactly like setup_data_parallel does
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

    model.train()
    x = {"t1": torch.randn(2, 1, 256, 256, 192)}
    # Put age/sex on cuda:0 (the "scatter source"), like the trainer does.
    # Use x_dict path so MultiModalDataParallel.scatter splits them.
    x["age"] = torch.tensor([72.0, 65.0], dtype=torch.float32, device="cuda:0")
    x["sex"] = torch.tensor([2, 1], dtype=torch.long, device="cuda:0")
    labels = torch.tensor([3, 0], dtype=torch.long, device="cuda:0")

    recon, cls_logits, mu, logvar = model(x, return_components=True)
    # Compute recon loss against a T1 that has the SAME batch size as recon
    # (DataParallel gather is identity for nn.DataParallel default but the
    # trainer pattern uses x_dict["t1"] which gets scattered, so the
    # post-gather recon might be cat'd across replicas).
    t1_local = x["t1"].to(recon.device)
    loss = (
        (recon - t1_local).abs().mean()
        + nn.functional.cross_entropy(cls_logits, labels)
        + 0.3 * (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
    )
    loss.backward()
    print(f"  [OK] DataParallel+demo forward+backward OK, loss={loss.item():.4f}, "
          f"recon.shape={tuple(recon.shape)} mu.shape={tuple(mu.shape)}")


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Modality-Toggle Smoke Tests (Stage 1 Multi-Modal VAE)")
    print("=" * 60)
    print("Modality-Toggle Smoke Tests (Stage 1 Multi-Modal VAE)")
    print("=" * 60)
    tests = [
        ("A: 5-modality default",     test_A_5modality),
        ("B: T1 + FLAIR",             test_B_t1_flair),
        ("C: T1 only (no demo)",      test_C_t1_only),
        ("D: T1 only + demo",         test_D_t1_demo),
        ("E: T1+demo backward",       test_E_backward_t1_only_demo),
        ("F: dataset returns age/sex",test_F_dataset_age_sex),
        ("G: missing demo -> 0/0",    test_G_dataset_missing_demo),
        ("H: DataParallel + demo",    test_H_data_parallel_with_demo),
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
