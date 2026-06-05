"""
3D Multi-Axis Attention modules for ADynamics.

Provides efficient 3D self-attention by decomposing the full O((DHW)^2) attention
into three axial O(DHW*(D+H+W)) attention passes along the depth, height, and
width axes. This makes attention tractable on 3D medical volumes (e.g. 16x16x12
bottleneck features) while still capturing long-range dependencies.

Pattern:  NeuroQuant (CVPR Findings 2026, Li et al.) "Modality-Aware and
          Anatomical Vector-Quantized Autoencoding for Multimodal Brain MRI"
          -> blocks.py::MultiAxisAttention

Design notes for ADynamics:
- Pre-norm (GroupNorm) for training stability with fp32 + small batch
- Zero-initialized output projection so the block starts as identity
  -> safe insertion into existing checkpoints / enables --no_attention
- Channels-first 3D tensors everywhere (B, C, D, H, W)
- GroupNorm group count is auto-computed to divide channels
- No dropout by default; can be enabled via constructor arg
- AMP-safe: pure float ops, no special dtype handling needed

Integration with MultiModalVAE3D:
- All 5 modality encoders (T1 + 4 optional) can share the same attention block
- The attention is inserted AFTER the last N strided-conv stages
- With use_attention=False, the model output is bitwise identical to the
  pre-attention design (backward compatible).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(num_channels: int, default_groups: int = 8) -> int:
    """Return a group count that divides num_channels, <= default_groups."""
    if num_channels % default_groups == 0:
        return default_groups
    for g in range(min(default_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class AxialAttention3D(nn.Module):
    """Self-attention along a single spatial axis of a 3D feature map.

    For axis=0 (depth): each (B, H, W) column attends across the D dimension.
    Implementation:
        1. Project to (q, k, v) via 1x1 conv
        2. Reshape to (B', seq, head_dim) where seq = attention axis length
        3. Standard multi-head self-attention over seq
        4. Reshape back to (B, C, D, H, W)

    Args:
        channels: Number of input/output channels.
        axis: Which axis to attend along (0=D, 1=H, 2=W).
        num_heads: Number of attention heads. Must divide channels.
                   Auto-reduced if not divisible.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        channels: int,
        axis: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0/1/2, got {axis}")
        # Auto-reduce num_heads if it doesn't divide channels
        if channels % num_heads != 0:
            for nh in range(num_heads, 0, -1):
                if channels % nh == 0:
                    num_heads = nh
                    break
        self.channels = channels
        self.axis = axis
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # qkv projection (in_proj)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=True)
        # out projection - zero-initialized for stable identity at start
        self.proj = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply axial attention.  Args:
            x: (B, C, D, H, W) feature map.
        Returns:
            (B, C, D, H, W) feature map with attention applied.
            At init (zero proj), output is exactly zero.
        """
        B, C, D, H, W = x.shape
        nH = self.num_heads
        head_dim = self.head_dim

        # 1) qkv projection: (B, 3C, D, H, W)
        qkv = self.qkv(x).chunk(3, dim=1)  # three (B, C, D, H, W)
        q, k, v = qkv

        # 2) Split channels into heads: (B, nH, head_dim, D, H, W)
        q = q.view(B, nH, head_dim, D, H, W)
        k = k.view(B, nH, head_dim, D, H, W)
        v = v.view(B, nH, head_dim, D, H, W)

        # 3) Permute so the attention axis is in position 2 (sequence),
        #    merge all other spatial dims into the "batch_others" slot.
        if self.axis == 0:
            # Move D to seq, merge H, W into batch_others
            # (B, nH, head_dim, D, H, W) -> (B, nH, D, H, W, head_dim)
            q = q.permute(0, 1, 3, 4, 5, 2).contiguous()
            k = k.permute(0, 1, 3, 4, 5, 2).contiguous()
            v = v.permute(0, 1, 3, 4, 5, 2).contiguous()
            # Merge (B, nH, H, W) into batch
            q = q.view(B * nH * H * W, D, head_dim)
            k = k.view(B * nH * H * W, D, head_dim)
            v = v.view(B * nH * H * W, D, head_dim)
        elif self.axis == 1:
            # (B, nH, head_dim, D, H, W) -> (B, nH, H, D, W, head_dim)
            q = q.permute(0, 1, 4, 3, 5, 2).contiguous()
            k = k.permute(0, 1, 4, 3, 5, 2).contiguous()
            v = v.permute(0, 1, 4, 3, 5, 2).contiguous()
            q = q.view(B * nH * D * W, H, head_dim)
            k = k.view(B * nH * D * W, H, head_dim)
            v = v.view(B * nH * D * W, H, head_dim)
        else:  # axis == 2
            # (B, nH, head_dim, D, H, W) -> (B, nH, W, D, H, head_dim)
            q = q.permute(0, 1, 5, 3, 4, 2).contiguous()
            k = k.permute(0, 1, 5, 3, 4, 2).contiguous()
            v = v.permute(0, 1, 5, 3, 4, 2).contiguous()
            q = q.view(B * nH * D * H, W, head_dim)
            k = k.view(B * nH * D * H, W, head_dim)
            v = v.view(B * nH * D * H, W, head_dim)

        # 4) Standard scaled dot-product attention over the seq axis
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B', seq, seq)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.bmm(attn, v)  # (B', seq, head_dim)

        # 5) Reshape back to (B, C, D, H, W)
        if self.axis == 0:
            # (B', D, head_dim) = (B*nH*H*W, D, head_dim) -> (B, nH, H, W, D, head_dim)
            out = out.view(B, nH, H, W, D, head_dim)
            # -> (B, nH, head_dim, D, H, W) via permute
            out = out.permute(0, 1, 5, 4, 2, 3).contiguous()
            out = out.view(B, C, D, H, W)
        elif self.axis == 1:
            # (B', H, head_dim) = (B*nH*D*W, H, head_dim) -> (B, nH, D, W, H, head_dim)
            out = out.view(B, nH, D, W, H, head_dim)
            out = out.permute(0, 1, 5, 2, 4, 3).contiguous()
            out = out.view(B, C, D, H, W)
        else:  # axis == 2
            # (B', W, head_dim) = (B*nH*D*H, W, head_dim) -> (B, nH, D, H, W, head_dim)
            out = out.view(B, nH, D, H, W, head_dim)
            out = out.permute(0, 1, 5, 2, 3, 4).contiguous()
            out = out.view(B, C, D, H, W)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MultiAxisAttention3D(nn.Module):
    """3D self-attention decomposed into three axial passes (D, H, W).

    Pattern from NeuroQuant (CVPR Findings 2026):
        attn_out = AxialAttn(D) + AxialAttn(H) + AxialAttn(W)

    Residual: out = x + sum(axial_attn(norm(x)))
    Pre-norm:  GroupNorm before attention, residual on the input.
    Zero-init: each axial sub-block's output projection is zero-initialized
               so the entire block starts as identity. This makes insertion
               into a pre-trained model safe (no initial loss spike).

    Args:
        channels: Number of input/output channels.
        num_heads: Number of attention heads per axial block.
        dropout: Dropout probability.
        use_d_h_w: Tuple of booleans for which axes to enable (default: all 3).
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_d_h_w: tuple = (True, True, True),
    ) -> None:
        super().__init__()
        self.channels = channels

        # Pre-norm (groupnorm) for stability
        gn_groups = _gn_groups(channels, default_groups=8)
        self.norm = nn.GroupNorm(gn_groups, channels)

        # Three axial attentions, each starts as identity (zero-init proj)
        self.attn_d = AxialAttention3D(channels, axis=0, num_heads=num_heads, dropout=dropout) if use_d_h_w[0] else None
        self.attn_h = AxialAttention3D(channels, axis=1, num_heads=num_heads, dropout=dropout) if use_d_h_w[1] else None
        self.attn_w = AxialAttention3D(channels, axis=2, num_heads=num_heads, dropout=dropout) if use_d_h_w[2] else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-axis attention with residual.

        Args:
            x: (B, C, D, H, W) feature map.

        Returns:
            (B, C, D, H, W) feature map.  When all attn submodules have
            zero-initialized output projections, output == input exactly
            (bitwise) for the first forward pass.
        """
        h = self.norm(x)
        out = 0
        if self.attn_d is not None:
            out = out + self.attn_d(h)
        if self.attn_h is not None:
            out = out + self.attn_h(h)
        if self.attn_w is not None:
            out = out + self.attn_w(h)
        return x + out


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

def _selftest() -> None:
    """Quick self-test: shapes, no NaN, residual identity at init."""
    print("=== AxialAttention3D selftest ===")
    for axis, B, C, D, H, W in [
        (0, 2, 32, 8, 8, 8),
        (1, 2, 32, 8, 8, 8),
        (2, 2, 32, 8, 8, 8),
        (0, 1, 64, 16, 16, 12),  # typical ADynamics bottleneck
        (1, 1, 128, 8, 8, 8),    # mid stage
    ]:
        block = AxialAttention3D(C, axis=axis)
        x = torch.randn(B, C, D, H, W)
        y = block(x)
        # With zero-init output proj, output should be exactly zero
        assert y.shape == x.shape, f"shape mismatch axis={axis}: {y.shape} vs {x.shape}"
        assert torch.isfinite(y).all(), f"NaN/Inf at axis={axis}"
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-6), \
            f"expected zero init at axis={axis}, got max diff {(y).abs().max().item()}"
        print(f"  axis={axis}  ({B},{C},{D},{H},{W})  OK (zero-init verified)")

    print("=== MultiAxisAttention3D selftest ===")
    for B, C, D, H, W in [
        (2, 32, 16, 16, 12),  # ADynamics bottleneck
        (1, 64, 8, 8, 8),
        (1, 128, 8, 8, 8),
    ]:
        # --- Part A: zero-init at construction -> identity, but grads still flow ---
        block = MultiAxisAttention3D(C)
        x = torch.randn(B, C, D, H, W)
        y = block(x)
        assert y.shape == x.shape, f"shape mismatch: {y.shape}"
        # Zero-init on all 3 axial proj -> residual identity
        assert torch.allclose(y, x, atol=1e-6), \
            f"expected identity at init, got max diff {(y - x).abs().max().item()}"
        assert torch.isfinite(y).all(), "NaN/Inf"
        # LayerScale behavior: at init the FORWARD is identity (weight=0),
        # but the GRADIENT on proj.weight is non-zero (so it can learn).
        y.sum().backward()
        proj_grad = block.attn_d.proj.weight.grad.norm().item()
        assert proj_grad > 0, f"proj grad should be non-zero (LayerScale), got {proj_grad}"
        for p in block.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # --- Part B: after un-zeroing proj, output should differ from input ---
        with torch.no_grad():
            for ax in (block.attn_d, block.attn_h, block.attn_w):
                ax.proj.weight.normal_(std=0.02)
                ax.proj.bias.zero_()
        y2 = block(x)
        assert not torch.allclose(y2, x, atol=1e-4), \
            "After un-zeroing proj, output should differ from input"
        y2.sum().backward()
        qkv_grad = block.attn_d.qkv.weight.grad.norm().item()
        assert qkv_grad > 0, f"qkv weight got no gradient after un-zero! norm={qkv_grad}"
        for p in block.parameters():
            if p.grad is not None:
                p.grad.zero_()
        print(f"  ({B},{C},{D},{H},{W})  OK  (identity@init, gradient flows both before and after)")

    # Test with attention partially enabled
    print("=== MultiAxisAttention3D with D-only ===")
    block = MultiAxisAttention3D(32, use_d_h_w=(True, False, False))
    x = torch.randn(1, 32, 8, 8, 8)
    y = block(x)
    assert torch.allclose(y, x, atol=1e-6), "expected identity with D-only (zero-init)"
    assert block.attn_d is not None
    assert block.attn_h is None
    assert block.attn_w is None
    print("  OK")

    # Test num_heads auto-reduction
    print("=== num_heads auto-reduction ===")
    block = AxialAttention3D(60, axis=0, num_heads=8)  # 60 not divisible by 8
    assert 60 % block.num_heads == 0
    print(f"  channels=60, requested 8 heads -> got {block.num_heads} heads")

    # Test that after training (non-zero proj), attention actually changes output
    print("=== Training signal verification ===")
    torch.manual_seed(0)
    block = MultiAxisAttention3D(32).train()
    # Manually un-zero the proj layers to simulate post-init state
    with torch.no_grad():
        for ax in (block.attn_d, block.attn_h, block.attn_w):
            ax.proj.weight.normal_(std=0.02)
            ax.proj.bias.zero_()
    x = torch.randn(1, 32, 8, 8, 8)
    y = block(x)
    assert not torch.allclose(y, x, atol=1e-4), \
        "After un-zeroing proj, output should differ from input"
    print(f"  max |y - x| = {(y - x).abs().max().item():.4f}  OK")

    print("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    _selftest()
