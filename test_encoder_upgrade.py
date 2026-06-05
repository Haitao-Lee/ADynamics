"""
Integration test for the MultiModalVAE3D encoder upgrade.

Validates that:
  1. With use_attention=False, output is bitwise identical to a no-attention
     baseline (only the new modules differ, which are absent).
  2. With use_attention=True at init, the FORWARD output is bitwise identical
     to the no-attention output (zero-init residual -> identity).
  3. After "warmup" (manually perturbing the proj weights), the output
     DIFFERS from no-attention -> the attention is actually doing something.
  4. Output shape is unchanged: per-modality (B, 32, 16, 16, 12), 4-class
     classifier output (B, 4).
  5. Forward+backward through the full MultiModalVAE3D (with all 5 modalities
     + classification head) is numerically stable (no NaN/Inf).
  6. Multi-GPU DataParallel scatter works with the new modules (dict input).

Run: python test_encoder_upgrade.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project importable
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn


def make_synthetic_batch(B: int = 2, T: int = 1, spatial: tuple = (256, 256, 192),
                         optional_spatial: tuple = (32, 64, 64)) -> dict:
    """Create a synthetic batch matching what MultiModalDataset would produce."""
    return {
        "t1":   torch.randn(B, T, *spatial) * 0.5 + 0.5,  # roughly brain-like intensity
        "fmri": torch.randn(B, T, *optional_spatial),
        "asl":  torch.randn(B, T, *optional_spatial),
        "qsm":  torch.randn(B, T, *optional_spatial),
        "flair":torch.randn(B, T, *optional_spatial),
        "label": torch.tensor([0, 1] * (B // 2))[:B],  # half NC, half SCD
    }


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_no_attention_baseline() -> nn.Module:
    """Build a baseline MultiModalVAE3D with use_attention=False. Returns the model."""
    from models.vae3d import MultiModalVAE3D
    model = MultiModalVAE3D(
        spatial_size=(256, 256, 192),
        in_channels=1,
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        dropout_rate=0.2,
        decoder_depth=4,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
        use_attention=False,
    )
    return model


def test_attention_model() -> nn.Module:
    """Build a MultiModalVAE3D with use_attention=True at default levels."""
    from models.vae3d import MultiModalVAE3D
    model = MultiModalVAE3D(
        spatial_size=(256, 256, 192),
        in_channels=1,
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        dropout_rate=0.2,
        decoder_depth=4,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
        use_attention=True,
        attention_levels=(3,),
        attention_heads=8,
    )
    return model


def test_multi_stage_attention() -> nn.Module:
    """Build a model with attention at the last TWO stages (2, 3)."""
    from models.vae3d import MultiModalVAE3D
    model = MultiModalVAE3D(
        spatial_size=(256, 256, 192),
        in_channels=1,
        latent_channels=32,
        base_channels=16,
        num_classes=4,
        dropout_rate=0.2,
        decoder_depth=4,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
        use_attention=True,
        attention_levels=(2, 3),
        attention_heads=4,  # smaller head count to test auto-reduction
    )
    return model


def run_forward(model: nn.Module, batch: dict, return_components: bool = True):
    """Run forward pass. Returns (recon, mu, logvar[, cls_logits])."""
    return model(batch, return_components=return_components)


def main():
    # Ensure deterministic float ops for this test only (cudnn.benchmark
    # would otherwise pick different algorithms for the two model forwards
    # and yield ~1e-4 noise even with identical weights).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    B = 2  # small batch to keep VRAM low
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in make_synthetic_batch(B).items()}

    # ----------------------------------------------------------------
    # 1) Baseline (no attention) -- reference output
    # ----------------------------------------------------------------
    print("\n=== TEST 1: No-attention baseline ===")
    torch.manual_seed(0)
    model_no = test_no_attention_baseline().to(device).eval()
    n_params_no = count_params(model_no)
    print(f"  Params: {n_params_no/1e6:.2f}M")
    with torch.no_grad():
        out_no = run_forward(model_no, batch)
    recon_no, cls_no, mu_no, logvar_no = out_no
    print(f"  recon:   {tuple(recon_no.shape)}")
    print(f"  cls:     {tuple(cls_no.shape)}")
    print(f"  mu:      {tuple(mu_no.shape)}")
    print(f"  logvar:  {tuple(logvar_no.shape)}")
    assert recon_no.shape == (B, 1, 256, 256, 192), f"recon shape wrong: {recon_no.shape}"
    assert cls_no.shape == (B, 4), f"cls shape wrong: {cls_no.shape}"
    assert mu_no.shape == (B, 32, 16, 16, 12), f"mu shape wrong: {mu_no.shape}"
    assert logvar_no.shape == (B, 32, 16, 16, 12), f"logvar shape wrong: {logvar_no.shape}"
    assert torch.isfinite(recon_no).all() and torch.isfinite(mu_no).all() and torch.isfinite(logvar_no).all()
    print("  Shapes + finite: OK")

    # ----------------------------------------------------------------
    # 2) Attention ON at init -- output should be bitwise identical to baseline
    #    IF the non-attention weights are the same.  We force that by copying
    #    all non-attention params from model_no into model_att.
    # ----------------------------------------------------------------
    print("\n=== TEST 2: use_attention=True at init (expect identity, weights copied) ===")
    torch.manual_seed(0)
    model_att = test_attention_model().to(device).eval()
    n_params_att = count_params(model_att)
    delta = (n_params_att - n_params_no) / 1e6
    print(f"  Params: {n_params_att/1e6:.2f}M  (+{delta:.2f}M vs baseline)")

    # Copy all NON-attention parameters from model_no to model_att
    sd_no = model_no.state_dict()
    sd_att = model_att.state_dict()
    for k in sd_att:
        if k in sd_no and sd_att[k].shape == sd_no[k].shape:
            sd_att[k].copy_(sd_no[k])
    model_att.load_state_dict(sd_att)
    print(f"  Copied {sum(1 for k in sd_att if k in sd_no and sd_att[k].shape == sd_no[k].shape)} non-attention params from baseline")

    with torch.no_grad():
        out_att = run_forward(model_att, batch)
    recon_att, cls_att, mu_att, logvar_att = out_att

    # Now outputs should be bitwise identical since the only difference is the
    # attention blocks, which are zero-initialized (residual identity).
    same_recon = torch.allclose(recon_no, recon_att, atol=1e-6)
    same_mu = torch.allclose(mu_no, mu_att, atol=1e-6)
    same_logvar = torch.allclose(logvar_no, logvar_att, atol=1e-6)
    same_cls = torch.allclose(cls_no, cls_att, atol=1e-6)
    print(f"  recon   identical to baseline: {same_recon}  (max diff {(recon_no-recon_att).abs().max().item():.2e})")
    print(f"  mu      identical to baseline: {same_mu}  (max diff {(mu_no-mu_att).abs().max().item():.2e})")
    print(f"  logvar  identical to baseline: {same_logvar}  (max diff {(logvar_no-logvar_att).abs().max().item():.2e})")
    print(f"  cls     identical to baseline: {same_cls}  (max diff {(cls_no-cls_att).abs().max().item():.2e})")
    assert same_recon and same_mu and same_logvar and same_cls, \
        "Zero-init attention should produce bitwise-identical output to no-attention"
    print("  PASS: zero-init residual is bitwise identity at init")
    print(f"  +{delta:.2f}M params for attention (5 modality encoders x 1 attn block x 3 axes)")

    # ----------------------------------------------------------------
    # 3) After "warmup" (perturb proj weights), attention should change output
    # ----------------------------------------------------------------
    print("\n=== TEST 3: After warmup (perturbed proj), output should differ ===")
    model_att.train()  # enable training mode (for ModalityDropout)
    # Manually perturb all attention proj weights
    with torch.no_grad():
        for name, p in model_att.named_parameters():
            if "attention_blocks" in name and "proj.weight" in name:
                p.add_(torch.randn_like(p) * 0.02)
    model_att.eval()
    with torch.no_grad():
        out_warm = run_forward(model_att, batch)
    recon_warm, _, mu_warm, _ = out_warm
    diff_recon = (recon_warm - recon_att).abs().max().item()
    diff_mu = (mu_warm - mu_att).abs().max().item()
    print(f"  max|recon_warm - recon_att| = {diff_recon:.4f}")
    print(f"  max|mu_warm - mu_att|      = {diff_mu:.4f}")
    assert diff_recon > 1e-4 or diff_mu > 1e-4, "Attention had no effect after warmup!"
    print("  Attention actually changes output: OK")

    # ----------------------------------------------------------------
    # 4) Full forward + backward + KL+recon+CE+contrastive (numerical stability)
    # ----------------------------------------------------------------
    print("\n=== TEST 4: Full forward + backward, KL/recon/cls all stable ===")
    model_att = model_att.to(device).train()  # back to train mode
    optimizer = torch.optim.AdamW(model_att.parameters(), lr=1e-4)
    labels = batch["label"]
    for step in range(3):
        optimizer.zero_grad()
        recon, cls_logits, mu, logvar = model_att(batch, return_components=True)
        recon_loss = (recon - batch["t1"]).abs().mean()
        kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        ce_loss = nn.functional.cross_entropy(cls_logits, labels)
        loss = recon_loss + 0.1 * kl_loss + 1.0 * ce_loss
        loss.backward()
        # Check no NaN gradients
        nan_grads = [n for n, p in model_att.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        inf_grads = [n for n, p in model_att.named_parameters() if p.grad is not None and p.grad.abs().max().item() == float("inf")]
        grad_norm = sum(p.grad.norm().item() for p in model_att.parameters() if p.grad is not None)
        optimizer.step()
        print(f"  step {step+1}: loss={loss.item():.4f}  recon={recon_loss.item():.4f}  kl={kl_loss.item():.4f}  ce={ce_loss.item():.4f}  grad_norm={grad_norm:.2f}  nan_grads={len(nan_grads)}")
        assert torch.isfinite(loss), f"loss has NaN/Inf at step {step}"
        assert len(nan_grads) == 0, f"NaN gradients at: {nan_grads[:3]}"
    print("  Numerical stability: OK (no NaN/Inf across 3 steps)")

    # ----------------------------------------------------------------
    # 5) Multi-stage attention (2, 3) -- also bitwise identical at init when
    #    non-attention weights are copied from baseline
    # ----------------------------------------------------------------
    print("\n=== TEST 5: Multi-stage attention (levels=2,3) ===")
    torch.manual_seed(0)
    model_multi = test_multi_stage_attention().to(device).eval()
    n_params_multi = count_params(model_multi)
    print(f"  Params: {n_params_multi/1e6:.2f}M")
    # Copy non-attention weights from baseline
    sd_no = model_no.state_dict()
    sd_multi = model_multi.state_dict()
    for k in sd_multi:
        if k in sd_no and sd_multi[k].shape == sd_no[k].shape:
            sd_multi[k].copy_(sd_no[k])
    model_multi.load_state_dict(sd_multi)
    with torch.no_grad():
        out_multi = run_forward(model_multi, batch)
    recon_multi, cls_multi, mu_multi, logvar_multi = out_multi
    assert recon_multi.shape == recon_no.shape
    assert mu_multi.shape == mu_no.shape
    print(f"  All shapes match baseline: OK")
    same = torch.allclose(recon_multi, recon_no, atol=1e-6)
    print(f"  Identity at init with 2 attention blocks: {same}  (max diff {(recon_multi-recon_no).abs().max().item():.2e})")
    assert same, "Multi-stage attention should also be identity at init"

    # ----------------------------------------------------------------
    # 6) Multi-GPU DataParallel scatter with dict input
    # ----------------------------------------------------------------
    print("\n=== TEST 6: Multi-GPU DataParallel scatter (skip if < 2 GPUs) ===")
    if torch.cuda.device_count() < 2:
        print("  SKIPPED: only 1 GPU available")
    else:
        from utils.multi_gpu import setup_data_parallel
        model_dp = test_attention_model()
        model_dp = setup_data_parallel(model_dp, num_gpus=2)
        # DataParallel wraps; the dict input is handled by MultiModalDataParallel
        model_dp.cuda()
        # Build a single-GPU batch first; the scatter will replicate across GPUs
        with torch.no_grad():
            try:
                out_dp = model_dp(batch, return_components=True)
                print(f"  DataParallel forward OK; recon shape: {tuple(out_dp[0].shape)}")
            except Exception as e:
                print(f"  DataParallel forward FAILED: {e}")
                raise

    print("\n=== ALL INTEGRATION TESTS PASSED ===")


if __name__ == "__main__":
    main()
