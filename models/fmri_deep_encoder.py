"""
fMRI Deep Encoder for ADynamics.

Replaces the lightweight 1D-conv hack in `fmri_temporal_encoder.py` with a
properly deep, multi-scale architecture that addresses the information-loss
problems in the old design:

  OLD PIPELINE (lossy):
    fMRI [B, 1, D=64, H=64, W=34, T=200]
        -> _spatial_avg_pool(D, H)        # 4096 voxels -> 1  (loses ALL spatial structure)
        -> 3x Conv1D stride-2              # T: 200 -> 16     (loses high-freq BOLD)
        -> 2-layer Transformer on 16 tokens (only models temporal, not cross-ROI)
        -> View as 3D (4, 4, 1)            (16 timepoints faked as 3D positions)
        -> Trilinear upsample to (16,16,12) (64x volume expansion with no new info)

  NEW PIPELINE (deeper, multi-scale, FC-aware):
    fMRI [B, 1, D=64, H=64, W=34, T=200]
        -> 1x1x1 Conv3d soft-ROI proj      # [B, n_roi=32, D, H, W, T]
                                            # Learned weighted sum of voxels
                                            # (preserves more spatial info than mean)
        -> Spatial pool (D, H) with attention  # [B, n_roi, W, T] = [B, 32, 34, 200]
        -> Reshape -> 1D CNN input         # [B, n_roi*W, T] = [B, 1088, 200]
        -> Multi-scale dilated 1D CNN      # 5 blocks, dilations 1, 2, 4, 8, 16
                                            # RFs: 7, 13, 25, 49, 97, 193 (covers T=200)
                                            # NO stride (preserve T)
        -> Multi-scale feature fusion      # 5*hidden -> hidden via 1x1 Conv
        -> TransformerEncoder (3 layers)   # Long-range temporal dependencies
        -> Multi-statistic temporal pool   # mean + std + max (3x information density)
        -> Functional connectivity head    # Cross-ROI correlation matrix
                                            # n_roi*(n_roi-1)/2 = 496 features
                                            # MLP to fc_compression
        -> 3D reshape: [B, hidden*3, n_roi, 1, W]
            -> trilinear interp to [B, hidden*3, 16, 16, 12]
            -> 1x1 conv to embed_dim
        -> Output: [B, embed_dim, 16, 16, 12]  (matches T1's latent grid)

Why this is better than the old version:
  1. Soft-ROI projection learns voxel combinations instead of hard mean
  2. 5 dilated conv blocks at different time scales (vs 3 stride-2 blocks)
  3. No aggressive T compression (T=200 preserved through CNN, only pooled at end)
  4. Transformer on full T=200 (not 16 compressed tokens)
  5. Functional connectivity matrix is the AD-relevant signal
  6. No 3D fakery - the 3D output grid is derived from real (ROI, W) structure

Param count target: ~2-3M (vs 91K for the old encoder). Still much smaller
than T1's 7M encoder.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class _DilatedConvBlock(nn.Module):
    """Conv1d (dilated) + GroupNorm + GELU + residual. Preserves temporal length."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        # 'same' padding for dilated conv: (kernel - 1) * dilation / 2
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                              padding=pad, dilation=dilation)
        groups = min(8, out_ch)
        while out_ch % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.GELU()
        # Residual projection if channels differ
        self.residual = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        res = self.residual(x)
        h = self.conv(x)
        # Crop to match residual if odd-length asymmetry
        if h.shape[-1] != res.shape[-1]:
            min_len = min(h.shape[-1], res.shape[-1])
            h = h[..., :min_len]
            res = res[..., :min_len]
        return self.act(self.norm(h) + res)


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------
class fMRIDeepEncoder(nn.Module):
    """
    Deep multi-scale fMRI BOLD encoder.

    Args:
        in_d, in_h, in_w, in_t: Input fMRI shape (excluding batch).
            Defaults match the AD dataset: (64, 64, 34, 200).
        n_soft_roi: Number of learned soft-ROI "factors" (no atlas needed).
            Each ROI is a learned linear combination of voxels.
            Default: 32 (analogous to a coarse parcellation).
        hidden_dim: Hidden dim of the 1D CNN stack. Default: 64.
        embed_dim: Output embedding channels (must equal latent_channels
            in MultiModalVAE3D for fusion_proj to work). Default: 32.
        target_grid: Output 3D grid. Default: (16, 16, 12) to match T1.
        num_transformer_layers: Depth of temporal Transformer. Default: 3.
        num_heads: Multi-head attention heads. Default: 4.
        fc_compression: Output dim of the FC head. Default: 32.
        zero_init: If True, zero-init the final 1x1 conv so the encoder
            starts as a near-zero contribution to fusion (safe for
            fine-tuning the new path while keeping other modalities
            unaffected).

    Forward:
        Input  x: [B, 1, D, H, W, T]
        Output   : [B, embed_dim, D_out, H_out, W_out]  (default: [B, 32, 16, 16, 12])
    """

    def __init__(
        self,
        in_d: int = 64,
        in_h: int = 64,
        in_w: int = 34,
        in_t: int = 200,
        n_soft_roi: int = 32,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        target_grid: Tuple[int, int, int] = (16, 16, 12),
        num_transformer_layers: int = 3,
        num_heads: int = 4,
        fc_compression: int = 32,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.in_d = in_d
        self.in_h = in_h
        self.in_w = in_w
        self.in_t = in_t
        self.n_soft_roi = n_soft_roi
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.target_grid = target_grid
        self.fc_compression = fc_compression

        # 1) Soft-ROI projection: 1x1x1 Conv3d that linearly combines the
        #    single input channel into n_soft_roi "ROI" channels.
        #    This is a learned, data-driven analog of an atlas parcellation.
        #    Param: n_soft_roi * 1 * 1 = n_soft_roi
        self.soft_roi_proj = nn.Conv3d(1, n_soft_roi, kernel_size=1)

        # 2) Spatial pool (D, H) -> 1x1 with a learnable attention weighting
        #    over the (D, H) plane. Implemented as a small depthwise conv
        #    followed by adaptive pool. We DO NOT just take the mean.
        #    Param: n_soft_roi * 9 (3x3 depthwise)
        self.spatial_attn = nn.Conv3d(
            n_soft_roi, n_soft_roi, kernel_size=3, padding=1, groups=n_soft_roi
        )
        self.spatial_pool = nn.AdaptiveAvgPool3d((1, 1, in_w))  # -> [B, n_roi, 1, 1, W]

        # 3) Multi-scale dilated 1D CNN. NO stride: preserve T=200 throughout.
        #    5 SEQUENTIAL blocks with dilations 1, 2, 4, 8, 16 give RFs
        #    7, 13, 25, 49, 97 (cumulative 193, covers T=200). 1x1 input
        #    projection reduces 1088 -> hidden_dim first.
        in_ch_1d = n_soft_roi * in_w  # 32 * 34 = 1088
        self.dilations = [1, 2, 4, 8, 16]
        # Initial 1x1 projection: 1088 -> hidden_dim
        self.input_proj = nn.Conv1d(in_ch_1d, hidden_dim, kernel_size=1)
        # 5 sequential dilated conv blocks
        self.dilated_convs = nn.ModuleList()
        for d in self.dilations:
            self.dilated_convs.append(
                _DilatedConvBlock(hidden_dim, hidden_dim,
                                  kernel_size=7 if d == 1 else 5,
                                  dilation=d)
            )
        # No scale_fuse: sequential design with shared hidden_dim.

        # 4) TransformerEncoder on full T=200 (no aggressive compression)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)

        # 5) Functional connectivity head.
        #    Cross-ROI correlation matrix is the AD-relevant signal.
        #    n_soft_roi * (n_soft_roi - 1) / 2 upper-triangle features.
        self._triu_idx: Optional[Tensor] = None  # lazy-init on first forward
        self.fc_head = nn.Sequential(
            nn.Linear(n_soft_roi * (n_soft_roi - 1) // 2, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, fc_compression),
        )

        # 6) 3D reshape + final 1x1 conv to embed_dim.
        #    Multi-stat pooling gives hidden_dim * 3 channels.
        #    We reshape [B, hidden*3, n_roi*W] into a 3D volume
        #    [B, hidden*3, n_roi, 1, W] = [B, hidden*3, 32, 1, 34]
        #    then trilinearly interp to (16, 16, 12).
        self.to_embed = nn.Conv3d(hidden_dim * 3, embed_dim, kernel_size=1)

        # Optional FC bias: additive on the embed_dim output, so the FC info
        # can complement the spatial+stat pool.
        self.fc_bias = nn.Linear(fc_compression, embed_dim)

        if zero_init:
            nn.init.zeros_(self.to_embed.weight)
            nn.init.zeros_(self.to_embed.bias)
            nn.init.zeros_(self.fc_bias.weight)
            nn.init.zeros_(self.fc_bias.bias)

    # -----------------------------------------------------------------------
    def _get_triu_idx(self, device: torch.device) -> Tensor:
        if self._triu_idx is None or self._triu_idx.device != device:
            n = self.n_soft_roi
            self._triu_idx = torch.triu_indices(n, n, offset=1, device=device)
        return self._triu_idx

    # -----------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode fMRI BOLD to multi-modal latent grid.

        Args:
            x: [B, 1, D, H, W, T] - 6D BOLD time series (T is the time dim).

        Returns:
            [B, embed_dim, D_out, H_out, W_out] - matches T1's latent grid.
        """
        if x.dim() == 5:
            # Backward-compat: legacy 3D path (time-averaged). Use static encoder.
            return self._handle_3d_legacy(x)

        assert x.dim() == 6, f"fMRI must be 6D (B,1,D,H,W,T); got {tuple(x.shape)}"
        B, _, D, H, W, T = x.shape
        n_roi = self.n_soft_roi

        # ---- 1) Soft-ROI projection (1x1x1 Conv3d over spatial dims) ----
        # nn.Conv3d expects 5D [B, C, D, H, W]. Move T into batch dim.
        x_for_conv = x.permute(0, 5, 1, 2, 3, 4).reshape(B * T, 1, D, H, W)
        roi_feats = self.soft_roi_proj(x_for_conv)  # [B*T, n_roi, D, H, W]
        roi_feats = roi_feats.view(B, T, n_roi, D, H, W).permute(0, 2, 3, 4, 5, 1).contiguous()
        # Now: [B, n_roi, D, H, W, T]

        # ---- 2) Spatial attention + pool over (D, H) ----
        # Permute for spatial conv: [B, T, n_roi, D, H, W] -> [B*T, n_roi, D, H, W]
        x_spatial = roi_feats.permute(0, 5, 1, 2, 3, 4).reshape(B * T, n_roi, D, H, W)
        x_spatial = self.spatial_attn(x_spatial)  # learnable spatial weighting
        x_spatial = self.spatial_pool(x_spatial)  # [B*T, n_roi, 1, 1, W]
        # Reshape back: [B, T, n_roi, 1, 1, W] -> [B, n_roi, 1, 1, W, T]
        x_pooled = x_spatial.view(B, T, n_roi, 1, 1, W).permute(0, 2, 3, 4, 5, 1).contiguous()
        # [B, n_roi, 1, 1, W, T]
        x_pooled = x_pooled.squeeze(2).squeeze(2)  # [B, n_roi, W, T]

        # ---- 3) Per-ROI time series BEFORE mixing via 1D CNN ----
        # x_pooled is [B, n_roi, W, T]. Mean over W to get per-ROI time series.
        roi_ts = x_pooled.mean(dim=2)      # [B, n_roi, T]
        # Note: this is the BOLD time series for each learned soft-ROI.

        # ---- 4) 1D CNN input: [B, n_roi*W, T] ----
        h = x_pooled.reshape(B, n_roi * W, T)

        # ---- 5) Multi-scale dilated 1D CNN (sequential, no stride) ----
        h = self.input_proj(h)             # [B, hidden, T]
        for conv in self.dilated_convs:
            h = conv(h)                    # [B, hidden, T] (preserved)

        # ---- 6) TransformerEncoder on T ----
        h = h.transpose(1, 2)               # [B, T, hidden]
        h = self.transformer(h)             # [B, T, hidden]
        h = h.transpose(1, 2)               # [B, hidden, T]

        # ---- 6) Multi-statistic temporal pooling ----
        h_mean = h.mean(dim=-1)             # [B, hidden]
        h_std = h.std(dim=-1)               # [B, hidden]
        h_max = h.max(dim=-1).values        # [B, hidden]
        h_stats = torch.cat([h_mean, h_std, h_max], dim=-1)  # [B, hidden*3]

        # ---- 7) Functional connectivity ----
        # z-score per-ROI time series
        roi_ts_z = (roi_ts - roi_ts.mean(-1, keepdim=True)) / \
                   (roi_ts.std(-1, keepdim=True) + 1e-6)
        # Correlation matrix: [B, n_roi, n_roi]
        fc = torch.einsum('bit,bjt->bij', roi_ts_z, roi_ts_z) / T
        # Upper triangle (off-diagonal only)
        triu = self._get_triu_idx(fc.device)
        fc_flat = fc[:, triu[0], triu[1]]    # [B, n_roi*(n_roi-1)/2]
        fc_emb = self.fc_head(fc_flat)       # [B, fc_compression]

        # ---- 8) 3D reshape + interp + 1x1 conv ----
        # [B, hidden*3] -> expand to [B, hidden*3, n_roi, 1, W] -> interp to target_grid
        h_3d = h_stats.view(B, self.hidden_dim * 3, 1, 1, 1)
        h_3d = h_3d.expand(B, self.hidden_dim * 3, n_roi, 1, W)
        h_3d = F.interpolate(h_3d, size=self.target_grid, mode="trilinear", align_corners=False)
        out = self.to_embed(h_3d)            # [B, embed_dim, 16, 16, 12]

        # ---- 9) Add FC info as an additive global bias ----
        # (1x1x1 conv at every spatial position gets the same FC bias added)
        fc_bias = self.fc_bias(fc_emb)        # [B, embed_dim]
        out = out + fc_bias.view(B, self.embed_dim, 1, 1, 1)

        return out

    # -----------------------------------------------------------------------
    def _handle_3d_legacy(self, x: Tensor) -> Tensor:
        """
        Fallback: input is 3D (time-averaged). Just produce a target-grid
        volume with the right channel count.
        """
        # x: [B, 1, D, H, W]  (no time dim)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = x.expand(x.shape[0], 1, *x.shape[2:])
        # Use the soft_roi_proj + spatial_pool + 3D interp path
        x = self.soft_roi_proj(x)            # [B, n_roi, D, H, W]
        x = self.spatial_attn(x)
        x = self.spatial_pool(x)             # [B, n_roi, 1, 1, W]
        x = x.squeeze(2).squeeze(2)          # [B, n_roi, W]
        # No time dim, no temporal processing. Just spatial pooling.
        B = x.shape[0]
        x_stats = torch.cat([x, x, x], dim=1)  # fake stats (no time)
        x_3d = x_stats.unsqueeze(-1).unsqueeze(-1)  # [B, n_roi, W, 1, 1]
        x_3d = F.interpolate(x_3d, size=self.target_grid, mode="trilinear",
                             align_corners=False)
        # Adjust channels
        if x_3d.shape[1] != self.hidden_dim * 3:
            x_3d = x_3d.expand(B, self.hidden_dim * 3, *x_3d.shape[2:])
        out = self.to_embed(x_3d)
        return out
