"""
GPU memory profiler for the OOM site at:
    models/vae3d.py:1260 (decoder_conv_out) inside the mixup path
    engine/trainer_vae.py:1017 (model_ref.decode(mu_mixed))

This script DOES NOT modify the trainer or model. It runs ONE train_epoch()
and prints the live / reserved / peak memory after each significant line, so
we can see exactly where the 22 GB goes.

Run:
    python tests/profile_oom.py

Saves a copy of the printed table to:
    inference_results/oom_profile.txt
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Project import path (matches other tests) -----------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Disable some training-side overhead before any big import ------------
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from core_data.dataset import MultiModalDataset, multimodal_collate_fn
from core_data.transforms import (
    MULTI_MODAL_SPATIAL_SIZES,
    get_multimodal_train_transforms,
    get_multimodal_val_transforms,
)
from engine.trainer_vae import MultiModalVAETrainer
from models.vae3d import MultiModalVAE3D
from utils.kl_schedules import get_kl_weight, should_apply_mixup, mixup_latents
from engine.losses import ordinal_cross_entropy_loss, ordinal_regression_loss

# -----------------------------------------------------------------------------
# Memory reporting
# -----------------------------------------------------------------------------
# All numbers in GiB (1e9 bytes).  This matches the format the user already
# sees from `nvidia-smi` and from the OOM traceback ("22.83 GiB is allocated").

def _mem() -> Dict[str, float]:
    return {
        "alloc_GB":   torch.cuda.memory_allocated() / 1e9,
        "reserved_GB": torch.cuda.memory_reserved() / 1e9,
        "peak_GB":    torch.cuda.max_memory_allocated() / 1e9,
    }


_LOG: List[str] = []
_LABELS: List[str] = []


def snapshot(label: str) -> None:
    """Record (and immediately print) a memory snapshot."""
    if not torch.cuda.is_available():
        line = f"{label:<55} | NO CUDA"
    else:
        m = _mem()
        line = (
            f"{label:<55} | "
            f"alloc={m['alloc_GB']:6.3f} GB | "
            f"reserved={m['reserved_GB']:6.3f} GB | "
            f"peak={m['peak_GB']:6.3f} GB"
        )
    _LOG.append(line)
    _LABELS.append(label)
    print(line, flush=True)


def per_tensor_bytes(name: str, tensor: Optional[torch.Tensor]) -> str:
    """Compute the per-tensor memory cost (in GiB) of an optional tensor."""
    if tensor is None:
        return f"{name:<20} = None"
    n = tensor.numel()
    elem = tensor.element_size()
    gbytes = (n * elem) / 1e9
    return f"{name:<20} = {tuple(tensor.shape)!s:<30} dtype={str(tensor.dtype):<20} nbytes={gbytes:6.3f} GB"


# -----------------------------------------------------------------------------
# Small in-memory T1 dataset (8 samples) so we don't fight disk I/O
# during a memory profile run.  Files live in a temp dir and are cleaned up.
# -----------------------------------------------------------------------------
def _make_t1_only_dataset(n: int = 8) -> tuple[str, List[Dict[str, Any]]]:
    import nibabel as nib
    import numpy as np
    import tempfile

    tmp = tempfile.mkdtemp(prefix="oom_profile_")
    paths = []
    for i in range(n):
        p = os.path.join(tmp, f"t1_{i}.nii.gz")
        # 256^3-ish volume so MONAI's transforms can pad/crop to 256x256x192.
        # Smaller here is fine: the transforms resize to 256^3 anyway.
        arr = np.random.rand(128, 128, 96).astype(np.float32)
        nib.save(nib.Nifti1Image(arr, np.eye(4)), p)
        paths.append(p)

    data_list = []
    for i in range(n):
        data_list.append({
            "t1": paths[i],
            "label": i % 4,                 # 4-class for stress
            "patient_id": f"profile_p{i}",
            "age": 70.0 + (i % 10),
            "sex": 1 + (i % 2),
        })
    return tmp, data_list


# -----------------------------------------------------------------------------
# Hand-rolled ONE-batch driver that mirrors the trainer's forward/backward
# step for batch 0 only, with memory snapshots at every requested point.
# Everything here is local to this file -- we do NOT modify the trainer.
# -----------------------------------------------------------------------------
def run_one_batch(trainer: MultiModalVAETrainer, batch: Dict[str, torch.Tensor]) -> None:
    device = trainer.device
    model = trainer.model
    config = trainer.config

    mixup_alpha = config.get("mixup_alpha", 0.4)
    mixup_prob  = config.get("mixup_prob", 0.5)
    cls_weight  = config.get("cls_weight", 1.0)
    kl_weight   = getattr(trainer, "current_kl_weight", config.get("kl_weight", 0.01))
    num_classes = config.get("num_classes", 4)
    use_amp     = config.get("use_amp", False)
    use_demo    = config.get("use_demographic_cond", False)
    free_bits   = config.get("free_bits", 0.0)
    ord_reg_w   = config.get("ordinal_reg_weight", 0.1)

    snapshot("[batch 0] start of batch")

    # --- move batch to device (mirrors trainer lines 936-968) -----------------
    t1 = batch["t1"].to(device)
    snapshot("[batch 0] after t1.to(device)")

    x_dict: Dict[str, torch.Tensor] = {"t1": t1}
    if use_demo and "age" in batch and "sex" in batch:
        x_dict["age"] = batch["age"].to(device)
        x_dict["sex"] = batch["sex"].to(device)
        snapshot("[batch 0] after demo moved to device")
    else:
        snapshot("[batch 0] no demo in batch (t1_only / no_demo)")

    labels = batch.get("label")
    if labels is not None:
        labels = labels.to(device)
        if labels.dim() > 1:
            labels = labels.squeeze()
    snapshot(f"[batch 0] after labels moved  (B={int(t1.size(0))})")

    # --- zero grad -------------------------------------------------------------
    trainer.optimizer.zero_grad(set_to_none=True)
    snapshot("[batch 0] after zero_grad(set_to_none=True)")

    # --- forward (autocast) ----------------------------------------------------
    import torch.nn.functional as F

    autocast_active = False
    if use_amp:
        autocast_ctx = autocast("cuda", enabled=True)
        autocast_ctx.__enter__()
        autocast_active = True
        snapshot("[batch 0] autocast entered")
    else:
        snapshot("[batch 0] autocast NOT used (use_amp=False)")

    # The actual model call.
    recon, cls_logits, mu, logvar = model(x_dict, return_components=True)
    snapshot("[batch 0] after model.forward (recon,cls_logits,mu,logvar)")

    # Per-tensor cost breakdown (the user explicitly asked for this).
    print("\n--- per-tensor memory cost (first forward) ---")
    for nm, t in [
        ("recon", recon), ("cls_logits", cls_logits),
        ("mu", mu), ("logvar", logvar),
    ]:
        print(per_tensor_bytes(nm, t))
    print("--- end per-tensor ---\n", flush=True)

    # --- mixup decision --------------------------------------------------------
    do_mixup = (mixup_alpha > 0) and (labels is not None) and should_apply_mixup(
        trainer.current_epoch, config,
    )
    print(f"[batch 0] mixup decision: do_mixup={do_mixup}  "
          f"alpha={mixup_alpha} prob={mixup_prob} mu.size(0)={int(mu.size(0))}",
          flush=True)

    if do_mixup and mu.size(0) >= 2:
        # Allocations BEFORE the no-grad decode
        snapshot("[mixup] before mixup_latents(...)")
        mu_mixed, lab_a, lab_b, lam = mixup_latents(mu, labels, mixup_alpha)
        snapshot("[mixup] after mixup_latents (mu_mixed built)")
        print(per_tensor_bytes("mu_mixed", mu_mixed))

        # ---- THE CRITICAL LINE in question ------------------------------------
        snapshot("[mixup] BEFORE model_ref.decode(mu_mixed)  <-- OOM site @ vae3d.py:1260")
        model_ref = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            recon_mixed = model_ref.decode(mu_mixed)
        snapshot("[mixup] AFTER  model_ref.decode(mu_mixed)")
        print(per_tensor_bytes("recon_mixed", recon_mixed))

        # verify the no-grad claim
        print(f"[mixup] recon_mixed.requires_grad={recon_mixed.requires_grad}  "
              f"grad_fn={recon_mixed.grad_fn}",
              flush=True)

        snapshot("[mixup] BEFORE model_ref.classify(mu_mixed)")
        with torch.no_grad():
            cls_logits_mixed = model_ref.classify(mu_mixed)
        snapshot("[mixup] AFTER  model_ref.classify(mu_mixed)")
        print(per_tensor_bytes("cls_logits_mixed", cls_logits_mixed))

        # mixup recon / cls losses
        perm = torch.randperm(t1.size(0), device=t1.device)
        recon_loss = F.l1_loss(
            recon_mixed, lam * t1 + (1.0 - lam) * t1[perm],
        )
        snapshot("[mixup] after recon_loss = l1(recon_mixed, ...)")
        cls_loss = __import__("utils.kl_schedules", fromlist=["mixup_classification_loss"]).mixup_classification_loss(
            cls_logits_mixed, lab_a, lab_b, lam,
        )
        snapshot("[mixup] after mixup_classification_loss")
        ordinal_reg_loss = ordinal_regression_loss(mu_mixed, lab_a, num_classes=num_classes)
        snapshot("[mixup] after ordinal_reg_loss(mu_mixed,...)")

        del recon_mixed, cls_logits_mixed, mu_mixed, lab_a, lab_b
        snapshot("[mixup] after del of mixup intermediates")
    else:
        snapshot("[non-mixup] branch entered")
        recon_loss = F.l1_loss(recon, x_dict["t1"])
        snapshot("[non-mixup] after recon_loss = l1(recon, t1)")
        if labels is not None:
            cls_loss = ordinal_cross_entropy_loss(cls_logits, labels, num_classes=num_classes)
            snapshot("[non-mixup] after cls_loss = ordinal_ce(cls_logits,labels)")
            ordinal_reg_loss = ordinal_regression_loss(mu, labels, num_classes=num_classes)
            snapshot("[non-mixup] after ordinal_reg_loss(mu,labels)")
        else:
            cls_loss = torch.tensor(0.0, device=device)
            ordinal_reg_loss = torch.tensor(0.0, device=device)
            snapshot("[non-mixup] no-labels path")

    # KL loss with free bits
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss = kl_per_dim.mean()
    snapshot("[batch 0] after KL (kl_loss)")

    loss = (recon_loss
            + cls_weight * cls_loss
            + kl_weight * kl_loss
            + ord_reg_w * ordinal_reg_loss)
    snapshot("[batch 0] after total loss assembled")

    if autocast_active:
        # exit the autocast context manager
        from torch.amp import autocast as _ac
        _ac("cuda", enabled=True).__exit__(None, None, None)
        snapshot("[batch 0] after autocast EXIT")
    else:
        snapshot("[batch 0] (autocast was off)")

    # backward
    if use_amp:
        trainer.scaler.scale(loss).backward()
        snapshot("[batch 0] AFTER scaler.scale(loss).backward()")
        trainer.scaler.unscale_(trainer.optimizer)
        snapshot("[batch 0] AFTER scaler.unscale_")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        snapshot("[batch 0] AFTER grad clip")
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        snapshot("[batch 0] AFTER scaler.step + update")
    else:
        loss.backward()
        snapshot("[batch 0] AFTER loss.backward()")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        snapshot("[batch 0] AFTER grad clip")
        trainer.optimizer.step()
        snapshot("[batch 0] AFTER optimizer.step()")

    # Trainer-style cleanup
    del loss, recon, cls_logits, mu, logvar
    snapshot("[batch 0] after del loss,recon,cls_logits,mu,logvar (BEFORE empty_cache)")
    torch.cuda.empty_cache()
    snapshot("[batch 0] after torch.cuda.empty_cache()  (this is what the trainer does)")


# -----------------------------------------------------------------------------
# Optional: also profile a NON-mixup batch so we can diff them.
# -----------------------------------------------------------------------------
def run_non_mixup_batch(trainer: MultiModalVAETrainer,
                        batch: Dict[str, torch.Tensor]) -> None:
    """Same driver but with mixup disabled in config (non-invasive: copy the dict)."""
    saved_alpha = trainer.config.get("mixup_alpha")
    trainer.config["mixup_alpha"] = 0.0
    try:
        run_one_batch(trainer, batch)
    finally:
        trainer.config["mixup_alpha"] = saved_alpha


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("OOM profiler -- ADynamics Stage 1 train_epoch batch 0")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("ERROR: no CUDA available; this profiler requires a GPU.")
        sys.exit(1)

    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"torch  : {torch.__version__}")
    print(flush=True)

    # Reset peak counter so all measurements are relative to model creation.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    snapshot("[init] after reset_peak_memory_stats")

    # --- Build model on GPU ----------------------------------------------------
    model = MultiModalVAE3D(
        spatial_size=(256, 256, 192),
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        decoder_depth=4,
        optional_modalities=[],          # t1_only
        use_demographic_cond=True,
        use_attention=True,
        attention_levels=(1, 2, 3),
        use_checkpointing=True,           # OOM fix
    )
    snapshot("[init] after model construction (CPU)")
    model = model.cuda()
    snapshot("[init] after model.cuda()")

    # Build a fake optimizer with NO momentum buffers (zero_grad) for fairness
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # --- Build dataset ---------------------------------------------------------
    tmp, data_list = _make_t1_only_dataset(n=8)
    print(f"\nCreated 8-sample T1-only dataset in {tmp}", flush=True)
    train_ds = MultiModalDataset(
        data_list, transform=get_multimodal_train_transforms(),
        optional_modalities=[],
    )
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=False,
        collate_fn=multimodal_collate_fn, num_workers=0, drop_last=False,
    )
    snapshot("[init] after DataLoader built (no items loaded yet)")

    # Materialize one batch to GPU so the snapshots show the real cost.
    batch = next(iter(train_loader))
    snapshot("[init] after first batch loaded by DataLoader (still on CPU)")
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.cuda()
    snapshot("[init] after first batch moved to GPU")

    # --- Build config (mirror stage1_vae.yaml's t1_only branch) ---------------
    config = {
        "recon_loss_type": "l1",
        "cls_weight": 4.0,
        "kl_weight": 0.3,
        "kl_strategy": "cyclical",
        "kl_cycle_len": 4,
        "kl_cycle_low_frac": 0.1,
        "kl_warmup_epochs": 30,
        "free_bits": 0.05,
        "num_classes": 4,
        "use_amp": False,                # stage1 yaml has use_amp: false
        "use_demographic_cond": True,
        "optional_modalities": [],
        "mixup_alpha": 0.4,              # <<< mixup ON
        "mixup_prob": 1.0,               # <<< force mixup on this one batch
        "encoder_grad_boost": 1.0,
        "ordinal_reg_weight": 0.1,
    }

    trainer = MultiModalVAETrainer(
        model=model, optimizer=optimizer,
        train_loader=train_loader, val_loader=train_loader,
        device=torch.device("cuda"),
        config=config, scheduler=None,
    )
    trainer.current_epoch = 0
    trainer.current_kl_weight, _ = get_kl_weight(0, config)
    snapshot("[init] after MultiModalVAETrainer(...) constructed")

    print()
    print("=" * 80)
    print("A) MIXUP BRANCH  (mixup_alpha=0.4, mixup_prob=1.0 -> forces mixup)")
    print("=" * 80, flush=True)
    run_one_batch(trainer, batch)

    # Re-load a fresh batch and force a non-mixup run for the diff.
    print()
    print("=" * 80)
    print("B) NON-MIXUP BRANCH  (mixup_alpha=0.0, mixup_prob=0)")
    print("=" * 80, flush=True)
    batch2 = next(iter(train_loader))
    for k, v in list(batch2.items()):
        if isinstance(v, torch.Tensor):
            batch2[k] = v.cuda()
    run_non_mixup_batch(trainer, batch2)

    # --- Summary ---------------------------------------------------------------
    print()
    print("=" * 80)
    print("SUMMARY (per-tensor cost estimates, computed in-script):")
    print("=" * 80)
    # The shapes the user mentioned:
    #   recon   = [2, 1, 256, 256, 192]  fp32
    #   mu      = [2, 32, 16, 16, 12]    fp32
    recon_theoretical = 2 * 1 * 256 * 256 * 192 * 4 / 1e9
    mu_theoretical    = 2 * 32 * 16 * 16 * 12 * 4 / 1e9
    print(f"  recon [2,1,256,256,192] fp32  theoretical = {recon_theoretical:.3f} GB")
    print(f"  mu    [2,32,16,16,12]   fp32  theoretical = {mu_theoretical:.3f} GB")
    print(f"  2x recon (first + mixup) = {2*recon_theoretical:.3f} GB")
    print(f"  recon + mu               = {recon_theoretical + mu_theoretical:.3f} GB")

    # --- Save the log ----------------------------------------------------------
    out_path = PROJECT_ROOT / "inference_results" / "oom_profile.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG))
        f.write("\n")
    print(f"\nFull log saved to: {out_path}")


if __name__ == "__main__":
    main()
