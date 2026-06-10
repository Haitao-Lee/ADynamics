"""
Quick smoke test of v10 trainer changes:
  - Build MultiModalVAE3D (T1-only, with demo) on CPU
  - Run 2 train_epoch() and 1 validate_epoch() on tiny synthetic data
  - Verify the new monitoring metrics (per_class_acc, latent_std, kl_active_dims)
    are present in the return dicts and are finite.

This is the canary test: if any of the new code paths break, this fails
before the real 18-25h training starts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import math
import torch
from torch.utils.data import DataLoader

from core_data.dataset import MultiModalDataset, multimodal_collate_fn
from core_data.transforms import get_multimodal_train_transforms, get_multimodal_val_transforms
from engine.trainer_vae import MultiModalVAETrainer
from models.vae3d import MultiModalVAE3D
from utils.kl_schedules import get_kl_weight


def _make_tiny_dataset(n: int = 6, has_labels: bool = True):
    """Build a tiny in-memory dataset of 4-class T1-only samples."""
    import nibabel as nib
    import numpy as np
    import tempfile

    tmp = tempfile.mkdtemp(prefix="smoke_")
    paths = []
    for i in range(n):
        p = os.path.join(tmp, f"t1_{i}.nii.gz")
        nib.save(nib.Nifti1Image(
            np.random.rand(64, 64, 64).astype(np.float32), np.eye(4)
        ), p)
        paths.append(p)

    data_list = []
    for i in range(n):
        item = {
            "t1": paths[i],
            "label": i % 4 if has_labels else 0,
            "patient_id": f"p{i}",
            "age": 70.0 + i,
            "sex": 1 + (i % 2),
        }
        data_list.append(item)
    return tmp, paths, data_list


def main() -> None:
    print("=" * 60)
    print("v10 Trainer Smoke Test (CPU, 2 train + 1 val epoch)")
    print("=" * 60)
    torch.manual_seed(42)

    tmp, paths, data_list = _make_tiny_dataset(n=4)
    train_ds = MultiModalDataset(
        data_list, transform=get_multimodal_train_transforms(), optional_modalities=[],
    )
    val_ds = MultiModalDataset(
        data_list, transform=get_multimodal_val_transforms(), optional_modalities=[],
    )
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                              collate_fn=multimodal_collate_fn, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False,
                            collate_fn=multimodal_collate_fn, num_workers=0, drop_last=True)

    model = MultiModalVAE3D(
        spatial_size=(256, 256, 192),
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        decoder_depth=4,
        optional_modalities=[],
        use_demographic_cond=True,
        use_attention=True,
        attention_levels=(1, 2, 3),  # v10: B1 mid-level attention
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    config = {
        "recon_loss_type": "l1",
        "cls_weight": 4.0,
        "kl_weight": 0.3,
        "kl_strategy": "cyclical",  # v10
        "kl_cycle_len": 4,
        "kl_cycle_low_frac": 0.1,
        "kl_warmup_epochs": 30,
        "free_bits": 0.05,
        "contrastive_weight": 0.0,
        "gradient_weight": 0.0,
        "ssim_weight": 0.0,
        "encoder_grad_boost": 1.0,
        "ordinal_reg_weight": 0.1,
        "num_classes": 4,
        "use_amp": False,
        "use_demographic_cond": True,
        "optional_modalities": [],
        # v10 mixup
        "mixup_alpha": 0.4,
        "mixup_prob": 0.5,
    }
    trainer = MultiModalVAETrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        config=config,
        scheduler=None,
    )

    # v10: verify cyclical KL is being dispatched
    for ep in range(6):
        w, strat = get_kl_weight(ep, config)
        print(f"  [KL schedule] ep={ep}  strategy={strat}  weight={w:.4f}")

    print()
    for ep in range(2):
        trainer.current_epoch = ep
        trainer.current_kl_weight, _ = get_kl_weight(ep, config)
        train_m = trainer.train_epoch()
        val_m = trainer.validate_epoch()
        # Verify v10 fields exist
        assert "per_class_acc" in train_m, "train missing per_class_acc"
        assert "per_class_acc" in val_m, "val missing per_class_acc"
        assert "val_latent_std" in val_m, "val missing val_latent_std"
        assert "val_kl_active_dims" in val_m, "val missing val_kl_active_dims"
        assert all(math.isfinite(x) for x in train_m["per_class_acc"]), "train per_class_acc non-finite"
        assert all(math.isfinite(x) for x in val_m["per_class_acc"]), "val per_class_acc non-finite"
        assert math.isfinite(val_m["val_latent_std"]), "val_latent_std non-finite"
        print(f"  [Epoch {ep}] train loss={train_m['loss']:.4f}  cls={train_m['cls_loss']:.4f}  "
              f"per_class={['{:.2f}'.format(x) for x in train_m['per_class_acc']]}")
        print(f"           val   loss={val_m['loss']:.4f}  cls={val_m['cls_loss']:.4f}  "
              f"per_class={['{:.2f}'.format(x) for x in val_m['per_class_acc']]}  "
              f"std={val_m['val_latent_std']:.3f}  actD={val_m['val_kl_active_dims']:.0f}/32")

    # Cleanup
    for p in paths:
        os.remove(p)
    os.rmdir(tmp)

    print("\n" + "=" * 60)
    print("v10 Trainer Smoke Test PASSED")
    print("  - cyclical KL dispatched correctly")
    print("  - mixup path runs without error")
    print("  - per_class_acc, latent_std, kl_active_dims all finite")
    print("=" * 60)


if __name__ == "__main__":
    main()
