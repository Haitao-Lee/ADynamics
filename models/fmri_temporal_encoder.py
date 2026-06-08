"""
fMRI Temporal Encoder for ADynamics.

Handles the 4D BOLD signal (T, D, H, W) by treating the time axis as the
primary axis and reducing spatial dimensions via average pooling. This
preserves the full BOLD time series (functionally relevant for AD) instead
of collapsing it to a static mean image like the original 4D→mean path.

Pipeline:
    fMRI [B, 1, D, H, W, T]
      -> Spatial average pool over (D, H) -> time series per voxel slice [B, T, W]
      -> 1D conv stack to compress time
      -> Adaptive pool to fixed 16 timepoints
      -> Lightweight TransformerEncoder (2 layers)
      -> Reshape to 3D feature volume [B, C, 4, 4, 1]
      -> Interpolate to target latent grid [B, C, 16, 16, 12]

Why not use a 3D temporal conv on the full 4D tensor?
    - A 3D conv on (B, 1, 64, 64, 34, 220) explodes memory:
        float32 * 1 * 64 * 64 * 34 * 220 = ~30 MB per sample, x batch and
        channel multiplier -> OOM on RTX 3090 with batch 2 and the rest of
        the model. By collapsing (D, H) first we reduce the working tensor
        by ~64*64 = 4096x.
    - The 3D structure of fMRI is shared with T1; the unique information
        BOLD adds is the *temporal dynamics* (BOLD fluctuations and their
        cross-region correlations), not the spatial structure. Hence
        spatial-pool-then-model-time is the right inductive bias.

Why spatial avg over (D, H), not (D, H, W)?
    - For resting-state fMRI in this dataset, the depth axis (W=34) is the
        partition dimension (anterior-posterior); averaging over the full
        3D volume throws away all spatial structure entirely. Keeping the
        W axis as the "channel" axis of the 1D conv gives a lightweight
        spatial summary that the time conv can mix.

This module is intentionally:
    - Lightweight (~0.5M params, vs ~10M for a 3D conv on full 4D)
    - Zero-initialized to start as a near-identity transform
    - Backward-compatible: if input is already 3D (legacy path), it just
        adaptive-pools to the target grid
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _Conv1DBlock(nn.Module):
    """Conv1D + GroupNorm + GELU. Halves temporal length."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class fMRITemporalEncoder(nn.Module):
    """
    1D temporal encoder for fMRI BOLD time series.

    Args:
        in_channels: Number of input "spatial channel" slices.
            After spatial avg pool over (D, H), the remaining spatial axis
            (W=34 by default) is treated as channels. Default 34.
        hidden_dim: Hidden dim of the 1D conv stack. Default 128.
        embed_dim:   Output embedding dim (== latent_channels consumed by
                     the fusion projection). Default 32.
        num_pool:    Number of 1D conv blocks (each halves the time length).
                     Default 3 -> 220 -> 110 -> 55 -> 28.
        target_t:    Adaptive-pool to this many timepoints before the
                     transformer. Default 16.
        num_transformer_layers: Default 2.
        num_heads:   Multi-head attention heads. Default 4.
        target_grid: Final 3D spatial grid to broadcast to. Default
                     (16, 16, 12) to match the T1 latent after 4 downsamples.
        zero_init:   If True, the final projection is zero-initialized so
                     the encoder is a near-identity at the start. Default
                     True (safe to add to an existing model).
        legacy_3d:   If True and the input is 3D (no time axis), skip the
                     1D path and just adaptive-pool the spatial dims to
                     target_grid. This lets old call-sites that already
                     averaged over time keep working.
    """

    def __init__(
        self,
        in_channels: int = 34,
        hidden_dim: int = 128,
        embed_dim: int = 32,
        num_pool: int = 3,
        target_t: int = 16,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        target_grid: Tuple[int, int, int] = (16, 16, 12),
        zero_init: bool = True,
        legacy_3d: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_pool = num_pool
        self.target_t = target_t
        self.target_grid = target_grid
        self.legacy_3d = legacy_3d

        # 1D conv stack
        ch_in = in_channels
        ch = hidden_dim
        self.conv_blocks = nn.ModuleList()
        for _ in range(num_pool):
            self.conv_blocks.append(_Conv1DBlock(ch_in, ch))
            ch_in = ch
            ch = max(ch // 2, embed_dim)
        self.conv_out_dim = ch_in  # final conv block output dim

        # Adaptive pool to fixed T (safety net if input T < target_t or odd)
        self.t_pool = nn.AdaptiveAvgPool1d(target_t)

        # Lightweight transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.conv_out_dim,
            nhead=num_heads,
            dim_feedforward=self.conv_out_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Project transformer output -> embed_dim
        self.proj = nn.Linear(self.conv_out_dim, embed_dim)

        # Zero-init the projection: at start the encoder returns ~0,
        # so the contribution to fusion is zero (safe to add to an
        # already-trained multi-modal VAE without disrupting the latent).
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

        # 3D reshape helpers (fixed): target_t tokens, map to target_grid.
        # We treat the T tokens as the spatial grid (16 -> 4*4*1).
        assert target_t == 16, (
            f"target_t must be 16 to match the (4,4,1) grid; got {target_t}"
        )

    def _spatial_avg_pool(self, x: Tensor) -> Tensor:
        """
        Reduce 5D fMRI (B, 1, D, H, W, T) to 3D time series (B, T, W)
        by averaging over (D, H).
        """
        # x shape: [B, 1, D, H, W, T]
        assert x.dim() == 6, f"fMRI must be 6D (B,1,D,H,W,T); got shape {tuple(x.shape)}"
        # Move T to dim=1 for clarity
        x = x.squeeze(1)             # [B, D, H, W, T]
        x = x.permute(0, 4, 1, 2, 3) # [B, T, D, H, W]
        x = x.mean(dim=(2, 3))       # [B, T, W] - average over D, H
        return x

    def _reshape_to_3d(self, z: Tensor) -> Tensor:
        """
        Reshape (B, target_t, embed_dim) -> (B, embed_dim, 4, 4, 1) and
        trilinearly interpolate to target_grid.
        """
        B, T, C = z.shape
        # 16 = 4 * 4 * 1
        z = z.view(B, 4, 4, 1, C)
        z = z.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, 4, 4, 1]
        z = F.interpolate(z, size=self.target_grid, mode="trilinear", align_corners=False)
        return z

    def _handle_3d_legacy(self, x: Tensor) -> Tensor:
        """
        Fallback: input is already 3D (1, D, H, W) after time-mean.
        Just adaptive-pool to target_grid with embed_dim channels.
        """
        # x shape: [B, 1, D, H, W]
        if x.dim() == 4:
            x = x.unsqueeze(1)
        B = x.shape[0]
        # Repeat to embed_dim then pool spatially
        x = x.expand(B, self.embed_dim, *x.shape[2:])
        x = F.adaptive_avg_pool3d(x, output_size=self.target_grid)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode fMRI to a 3D latent volume matching the T1 latent grid.

        Args:
            x: One of:
                - 6D tensor [B, 1, D, H, W, T] (full BOLD time series)
                - 5D tensor [B, D, H, W, T]     (no leading singleton)
                - 4D tensor [B, 1, D, H, W]     (legacy: already time-averaged)

        Returns:
            5D latent tensor [B, embed_dim, 16, 16, 12] matching the T1
            encoder's output spatial grid.
        """
        # Detect legacy 3D path: a 4D [B,1,D,H,W] or 5D [B,1,D,H,W] tensor
        # is the time-averaged case (no time axis). We treat anything without
        # a trailing time dim as 3D legacy. 5D shape (B, D, H, W, T) is the
        # 4D-with-leading-channel stripped form of a true 4D time series.
        is_3d_legacy = (
            x.dim() == 4
            or (x.dim() == 5 and x.shape[1] == 1 and self.legacy_3d)
        )
        if is_3d_legacy:
            return self._handle_3d_legacy(x)

        # 6D path: [B, 1, D, H, W, T] or 5D: [B, D, H, W, T]
        if x.dim() == 6:
            ts = self._spatial_avg_pool(x)   # [B, T, W]
        else:
            # 5D: [B, D, H, W, T]
            assert x.dim() == 5, f"fMRI must be 5D or 6D; got {x.dim()}D"
            x = x.unsqueeze(1)               # [B, 1, D, H, W, T]
            ts = self._spatial_avg_pool(x)   # [B, T, W]

        # ts: [B, T, W]. The 1D conv expects [B, channels, length] -> permute
        ts = ts.permute(0, 2, 1).contiguous()  # [B, W, T]

        for blk in self.conv_blocks:
            ts = blk(ts)

        # Pool to fixed T
        ts = self.t_pool(ts)  # [B, C, target_t]

        # Transformer expects [B, T, C]
        ts = ts.permute(0, 2, 1).contiguous()
        ts = self.transformer(ts)  # [B, target_t, C]

        # Project to embed_dim
        ts = self.proj(ts)  # [B, target_t, embed_dim]

        # Reshape to 3D grid
        return self._reshape_to_3d(ts)


def smoke_test() -> None:
    """Quick self-test: forward pass with realistic shapes."""
    enc = fMRITemporalEncoder(
        in_channels=34,
        hidden_dim=128,
        embed_dim=32,
        target_grid=(16, 16, 12),
    )
    enc.eval()

    # 6D input
    x6 = torch.randn(2, 1, 64, 64, 34, 220)
    y6 = enc(x6)
    assert y6.shape == (2, 32, 16, 16, 12), f"6D out shape wrong: {y6.shape}"
    print(f"6D path OK: {tuple(x6.shape)} -> {tuple(y6.shape)}")

    # 5D input
    x5 = torch.randn(2, 64, 64, 34, 220)
    y5 = enc(x5)
    assert y5.shape == (2, 32, 16, 16, 12), f"5D out shape wrong: {y5.shape}"
    print(f"5D path OK: {tuple(x5.shape)} -> {tuple(y5.shape)}")

    # Legacy 3D input (already time-averaged) — 4D [B,1,D,H,W]
    x4 = torch.randn(2, 1, 64, 64, 34)
    y4 = enc(x4)
    assert y4.shape == (2, 32, 16, 16, 12), f"4D out shape wrong: {y4.shape}"
    print(f"4D legacy path OK: {tuple(x4.shape)} -> {tuple(y4.shape)}")

    # Zero-init sanity: with zero init, output should be ~constant (broadcast)
    assert torch.allclose(y6, y6[0:1].expand_as(y6), atol=1e-6), "Zero-init not constant"
    print("Zero-init sanity OK (output is broadcast-constant)")

    # Backward sanity — temporarily reinit the proj to break the
    # zero-init chain so we can verify gradient flows end-to-end.
    enc.train()
    nn.init.xavier_uniform_(enc.proj.weight)
    nn.init.zeros_(enc.proj.bias)
    x = torch.randn(1, 1, 64, 64, 34, 220, requires_grad=True)
    y = enc(x)
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "No grad on input!"
    print("Backward pass OK")


if __name__ == "__main__":
    smoke_test()
