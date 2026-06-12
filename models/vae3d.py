"""
3D Variational Autoencoder (VAE) for ADynamics.

Implements:
- Single-modal 3D VAE: ADynamicsVAE3D
- Multi-modal 3D VAE: MultiModalVAE3D (T1 + optional fMRI/ASL/QSM/FLAIR)

Architecture:
    - Encoder: 3D CNN with residual blocks, outputting mu and logvar
    - Reparameterization: z = mu + std * epsilon (or z = mu in eval mode)
    - Decoder: 3D transposed CNN with residual blocks, outputting sigmoid activation

HD Support:
    - Input: [B, 1, 256, 256, 192]
    - After 4 downsampling blocks: [B, 512, 16, 16, 12]
    - Decoder mirrors encoder structure
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Local import: the new multi-axis 3D attention module.
# Cyclic-import safe: attention_3d.py only depends on torch.
from models.attention_3d import MultiAxisAttention3D

# Optional fMRI temporal encoder (4D BOLD -> 3D latent). When
# use_fmri_temporal=True, the static ModalityEncoder3D for fMRI is
# replaced by this lightweight 1D conv + transformer that preserves the
# full BOLD time series instead of collapsing it to a static mean.
try:
    from models.fmri_temporal_encoder import fMRITemporalEncoder
except ImportError:  # pragma: no cover
    fMRITemporalEncoder = None  # type: ignore[assignment]


class ResidualBlock3D(nn.Module):
    """
    3D Residual Block with GroupNorm and LeakyReLU.

    Implements a standard residual block with two 3D convolution layers,
    group normalization, and residual connection.
    """

    @staticmethod
    def _compute_groups(num_channels: int, default_groups: int = 8) -> int:
        """Compute appropriate number of groups for GroupNorm."""
        if num_channels % default_groups == 0:
            return default_groups
        # Find the largest divisor <= default_groups
        for g in range(min(default_groups, num_channels), 0, -1):
            if num_channels % g == 0:
                return g
        return 1  # Fall back to LayerNorm equivalent

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
        leakyrelu_slope: float = 0.2,
    ) -> None:
        """
        Initialize a 3D residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_groups: Number of groups for GroupNorm. Must divide in_channels and out_channels
            leakyrelu_slope: Negative slope for LeakyReLU
        """
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=leakyrelu_slope, inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=leakyrelu_slope, inplace=True)

        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through residual block.

        Args:
            x: Input tensor of shape [B, C, D, H, W]

        Returns:
            Output tensor of shape [B, C_out, D, H, W]
        """
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        return out + residual


class DownBlock3D(nn.Module):
    """
    3D Downsampling block with strided convolution.

    Reduces spatial dimensions by factor of 2 while doubling channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
        leakyrelu_slope: float = 0.2,
    ) -> None:
        """
        Initialize a 3D downsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (typically 2x in_channels)
            num_groups: Number of groups for GroupNorm
            leakyrelu_slope: Negative slope for LeakyReLU
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(negative_slope=leakyrelu_slope, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with downsampling.

        Args:
            x: Input tensor of shape [B, C, D, H, W]

        Returns:
            Output tensor of shape [B, C_out, D/2, H/2, W/2]
        """
        return self.block(x)


class SelfAttentionBlock3D(nn.Module):
    """
    3D Spatial + Channel Attention Block (Memory-Efficient).

    Applies channel attention (SE-style) + light spatial attention.
    Uses spatial pooling to reduce memory footprint before attention computation.
    """

    @staticmethod
    def _compute_groups(num_channels: int, default_groups: int = 8) -> int:
        """Compute appropriate number of groups for GroupNorm."""
        if num_channels % default_groups == 0:
            return default_groups
        for g in range(min(default_groups, num_channels), 0, -1):
            if num_channels % g == 0:
                return g
        return 1

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
    ) -> None:
        """
        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio for SE module. Default: 4
        """
        super().__init__()

        mid_ch = max(channels // reduction, 8)

        # Channel attention (SE-style)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, mid_ch, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(mid_ch, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Light spatial attention via 1x1 conv to keep memory low
        spatial_groups = self._compute_groups(channels // 4)
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(channels, channels // 4, kernel_size=1),
            nn.GroupNorm(spatial_groups, channels // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [B, C, D, H, W]

        Returns:
            Output tensor [B, C, D, H, W] with attention applied
        """
        # Channel attention
        ch_attn = self.se(x) * x

        # Spatial attention
        sp_attn = self.spatial_conv(ch_attn)
        out = ch_attn * sp_attn

        return self.gamma * out + x


class UpBlock3D(nn.Module):
    """
    3D Upsampling block with transposed convolution.

    Increases spatial dimensions by factor of 2 while halving channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
        leakyrelu_slope: float = 0.2,
        use_attention: bool = False,
    ) -> None:
        """
        Initialize a 3D upsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (typically in_channels/2)
            num_groups: Number of groups for GroupNorm
            leakyrelu_slope: Negative slope for LeakyReLU
            use_attention: If True, apply self-attention after upsampling. Default: False
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(negative_slope=leakyrelu_slope, inplace=True),
        )

        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttentionBlock3D(out_channels, reduction=2)

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        if self.use_attention:
            out = self.attention(out)
        return out


class ADynamicsVAE3D(nn.Module):
    """
    3D Variational Autoencoder for Alzheimer's Disease MRI Analysis.

    Learns a compressed latent representation of T1-weighted MRI scans
    using a VAE architecture with residual blocks and GroupNorm.

    HD Input: [B, 1, 256, 256, 192] (high-definition)
    After 4 downsampling blocks: [B, 512, 16, 16, 12]
    The decoder mirrors this structure to reconstruct [B, 1, 256, 256, 192].

    Memory optimization: Set base_channels=16 if OOM occurs with HD inputs.
    Gradient checkpointing can be enabled via use_checkpointing=True for
    memory-constrained environments (decoder long chains benefit most).

    Attributes:
        spatial_size: Original spatial dimensions
        in_channels: Number of input channels (1 for T1 MRI)
        latent_channels: Number of channels in latent space
        base_channels: Base channel count (32 default, 16 for memory saving)
        use_checkpointing: Enable gradient checkpointing for memory efficiency
    """

    def __init__(
        self,
        spatial_size: Tuple[int, int, int] = (256, 256, 192),
        in_channels: int = 1,
        latent_channels: int = 64,
        base_channels: int = 32,
        use_checkpointing: bool = False,
        use_attention: bool = False,
        use_demographic_cond: bool = False,
        decoder_depth: int = 4,
        age_range: Tuple[float, float] = (0.0, 100.0),
        num_age_bins: int = 100,
        num_sex_values: int = 3,
    ) -> None:
        """
        Initialize the 3D VAE.

        Args:
            spatial_size: Spatial dimensions of input MRI (D, H, W). Default: (256, 256, 192)
            in_channels: Number of input channels (default: 1 for T1)
            latent_channels: Number of channels in latent representation
            base_channels: Base channel count for conv layers. Default: 32.
                           Reduce to 16 if OOM occurs with HD inputs.
            use_checkpointing: If True, use torch.utils.checkpoint to save memory.
                               Recommended for HD (256,256,192) inputs. Default: False
            use_attention: If True, add self-attention after each UpBlock in decoder.
                           Helps capture long-range dependencies for better texture. Default: False
            use_demographic_cond: If True, condition on age and sex embeddings in latent space.
                                  Uses FiLM modulation to adjust mu/logvar based on demographics.
                                  Default: False
            decoder_depth: Number of decoder upsampling blocks. Default: 4.
                          For Stage 1 (encoder-focused training), use 3 to reduce decoder memory.
                          With 3 blocks: 8x upsampling to [128,128,96], then final transposed conv to full resolution.
                          NOTE: decoder_depth >= 3 and <= 4 only. Values outside this range will not produce correct output size.
            age_range: Min and max age for normalization. Default: (0.0, 100.0)
            num_age_bins: Number of bins for age embedding (continuous age -> discretized bin).
                          Higher = finer granularity. Default: 100
            num_sex_values: Number of sex categories in dataset encoding.
                            0 = Unknown/missing, 1 = Male, 2 = Female. Default: 3
        """
        super().__init__()

        self.spatial_size = spatial_size
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.use_checkpointing = use_checkpointing
        self.use_attention = use_attention
        self.use_demographic_cond = use_demographic_cond
        self.decoder_depth = decoder_depth

        # With 4 downsampling blocks: 256 -> 128 -> 64 -> 32 -> 16
        # Latent spatial size for HD (256, 256, 192): [16, 16, 12]
        num_downsamples = 4

        # Encoder
        self.encoder_conv_in = nn.Conv3d(
            in_channels, base_channels, kernel_size=3, padding=1
        )
        # Compute num_groups for encoder_norm_in such that it's divisible by base_channels
        # Use min(8, base_channels) groups, falling back to base_channels itself if needed
        encoder_norm_groups = min(8, base_channels)
        if base_channels % encoder_norm_groups != 0:
            encoder_norm_groups = 1  # Fall back to LayerNorm equivalent
        self.encoder_norm_in = nn.GroupNorm(encoder_norm_groups, base_channels)
        self.encoder_act_in = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.encoder_layers = nn.ModuleList()
        ch = base_channels
        for _ in range(num_downsamples):
            self.encoder_layers.append(
                DownBlock3D(ch, ch * 2, num_groups=8)
            )
            self.encoder_layers.append(
                ResidualBlock3D(ch * 2, ch * 2, num_groups=8)
            )
            ch *= 2

        self.latent_conv = nn.Conv3d(
            ch, latent_channels * 2, kernel_size=3, padding=1
        )

        # Demographic conditioning: age + sex embeddings in latent space
        if use_demographic_cond:
            self.age_embedding = nn.Embedding(num_age_bins, latent_channels)
            self.sex_embedding = nn.Embedding(num_sex_values, latent_channels)
            self.age_range = age_range
            self.num_age_bins = num_age_bins
            # FiLM-like modulation: scale and shift for each spatial location
            self.demographic_fc = nn.Sequential(
                nn.Linear(latent_channels * 2, latent_channels * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(latent_channels * 4, latent_channels * 2),
            )

        # Decoder: use decoder_depth instead of num_downsamples
        # When decoder_depth < 4, add a final transposed conv to complete the upsampling
        self.decoder_latent_conv = nn.Conv3d(
            latent_channels, ch, kernel_size=3, padding=1
        )

        self.decoder_layers = nn.ModuleList()
        for _ in range(decoder_depth):
            self.decoder_layers.append(
                ResidualBlock3D(ch, ch, num_groups=4)
            )
            self.decoder_layers.append(
                UpBlock3D(ch, ch // 2, num_groups=4, use_attention=use_attention)
            )
            ch //= 2

        # Final transposed conv if decoder is shallower (less than 4 blocks)
        # 4 blocks -> 16x upsampling (16->256, 16->256, 12->192)
        # 3 blocks -> 8x upsampling + transposed conv for final 2x
        self.final_upsample = None
        if decoder_depth < 4:
            # After decoder_depth blocks: spatial = [16*2^decoder_depth, 16*2^decoder_depth, 12*2^decoder_depth]
            # e.g., decoder_depth=3: [128, 128, 96], need final 2x to [256, 256, 192]
            self.final_upsample = nn.ConvTranspose3d(
                ch, ch, kernel_size=2, stride=2
            )
            up_ch = ch
        else:
            up_ch = ch

        self.decoder_conv_out = nn.Sequential(
            nn.Conv3d(up_ch, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, base_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Multi-scale output heads: conv from each UpBlock output to prediction
        # Scale outputs at intermediate decoder depths
        self.scale_heads = nn.ModuleDict()
        # With decoder_depth, the upsample positions are [1, 3, 5, ...] (0-indexed: odd positions)
        self._decoder_upblock_positions = [1 + i * 2 for i in range(decoder_depth)]
        # Channel counts after each UpBlock: starts at ch=512, halves each block
        # For decoder_depth=4: [256, 128, 64, 32] spatial: [32,32,24], [64,64,48], [128,128,96], [256,256,192]
        # For decoder_depth=3: [256, 128, 64] spatial: [32,32,24], [64,64,48], [128,128,96]
        self._scale_channels = {}
        ch_for_scale = base_channels * 16  # 512 for base_channels=32, starts at encoder output
        for i in range(decoder_depth):
            # Each UpBlock halves channels: input ch, output ch//2
            # So after UpBlock i (0-indexed), h has ch_for_scale//2 channels
            self._scale_channels[i + 1] = ch_for_scale // 2
            ch_for_scale //= 2
        for scale_idx in range(1, decoder_depth + 1):
            c = self._scale_channels[scale_idx]
            c_half = c // 2
            gn_groups = ResidualBlock3D._compute_groups(c_half)
            self.scale_heads[str(scale_idx)] = nn.Sequential(
                nn.Conv3d(c, c_half, kernel_size=1),
                nn.GroupNorm(gn_groups, c_half),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(c_half, 1, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode MRI to latent space parameters.

        Args:
            x: Input MRI tensor of shape [B, 1, 256, 256, 192]

        Returns:
            Tuple of (mu, logvar) both of shape [B, latent_channels, 16, 16, 12]
        """
        h = self.encoder_conv_in(x)
        h = self.encoder_norm_in(h)
        h = self.encoder_act_in(h)
        # shape: [B, 32, 256, 256, 192]

        for layer in self.encoder_layers:
            h = layer(h)
        # shape after 4 downsamples: [B, 512, 16, 16, 12]

        latent = self.latent_conv(h)
        # shape: [B, latent_channels*2, 16, 16, 12]

        mu, logvar = latent.chunk(2, dim=1)
        # shape: [B, latent_channels, 16, 16, 12] each

        return mu, logvar

    def condition_on_demographic(self, mu: Tensor, logvar: Tensor, age: Tensor, sex: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply demographic conditioning via FiLM (Feature-wise Linear Modulation).

        Modulates latent distribution based on age and sex embeddings:
            mu' = mu + gamma
            logvar' = logvar + beta
        where gamma, beta are derived from age + sex embeddings.

        Args:
            mu: Mean of latent distribution [B, latent_channels, D, H, W]
            logvar: Log variance of latent distribution [B, latent_channels, D, H, W]
            age: Age values [B], raw float values
            sex: Sex labels [B], integer values (0, 1, 2)

        Returns:
            Tuple of (modulated_mu, modulated_logvar)
        """
        if not self.use_demographic_cond:
            return mu, logvar

        # Normalize age to [0, 1] then to bin index
        age_min, age_max = self.age_range
        age_norm = (age.float() - age_min) / (age_max - age_min + 1e-8)
        age_bin = (age_norm * (self.num_age_bins - 1)).long().clamp(0, self.num_age_bins - 1)

        # Embeddings: [B] -> [B, latent_channels]
        age_emb = self.age_embedding(age_bin)      # [B, C]
        sex_emb = self.sex_embedding(sex.long())   # [B, C]

        # Concatenate and project to gamma, beta: [B, 2C] -> [B, C, D, H, W]
        demo_emb = torch.cat([age_emb, sex_emb], dim=1)  # [B, 2C]
        film = self.demographic_fc(demo_emb)  # [B, 2C]
        gamma, beta = film.chunk(2, dim=1)    # each [B, C]

        # Broadcast to spatial dimensions: [B, C] -> [B, C, D, H, W]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # FiLM modulation: additive residual on mu, additive on logvar
        mu_mod = mu + gamma
        logvar_mod = logvar + beta

        return mu_mod, logvar_mod

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick for VAE sampling.

        In training mode: z = mu + std * epsilon (with stochastic noise)
        In eval mode: z = mu (deterministic, no noise for stable inference)

        Args:
            mu: Mean of latent distribution, shape [B, C, D, H, W]
            logvar: Log variance of latent distribution, shape [B, C, D, H, W]

        Returns:
            Sampled latent tensor
        """
        if not self.training:
            # Eval mode: deterministic, return mean directly
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: Tensor, return_multi_scale: bool = False):
        """
        Decode latent representation back to MRI space.

        Args:
            z: Latent tensor of shape [B, latent_channels, 16, 16, 12]
            return_multi_scale: If True, return list of [recon_s1, recon_s2, recon_s3, recon_full].
                                If False, return only full-scale reconstruction (backward compatible).

        Returns:
            When return_multi_scale=False: Reconstructed MRI of shape [B, 1, 256, 256, 192]
            When return_multi_scale=True: List of 4 reconstructions at different scales.

        Note:
            Multi-scale outputs help preserve fine texture details by computing loss at each scale.
            Scales: 1=smallest (256,256,192 / 8), 2=medium (256,256,192 / 4), 3=large (256,256,192 / 2), full=256,256,192
        """
        h = self.decoder_latent_conv(z)
        # shape: [B, 512, 16, 16, 12]

        scale_idx = 0
        scale_outputs = []

        # Dynamic upblock positions based on decoder_depth
        # Each UpBlock is at odd index (1, 3, 5, ...)
        upblock_positions = [1 + i * 2 for i in range(self.decoder_depth)]

        if self.use_checkpointing:
            for i, layer in enumerate(self.decoder_layers):
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
                # Only produce up to decoder_depth scale outputs (1 per UpBlock)
                if i in upblock_positions and len(scale_outputs) < self.decoder_depth:
                    scale_recon = self.scale_heads[str(len(scale_outputs) + 1)](h)
                    scale_outputs.append(scale_recon)
        else:
            for i, layer in enumerate(self.decoder_layers):
                h = layer(h)
                # Only produce up to decoder_depth scale outputs (1 per UpBlock)
                if i in upblock_positions and len(scale_outputs) < self.decoder_depth:
                    scale_recon = self.scale_heads[str(len(scale_outputs) + 1)](h)
                    scale_outputs.append(scale_recon)

        # Final upsampling if decoder is shallower (needs transposed conv)
        if self.final_upsample is not None:
            h = self.final_upsample(h)

        # Final full-scale reconstruction
        recon_full = self.decoder_conv_out(h)

        if return_multi_scale:
            # Scale 1 (smallest): spatial = 32,32,24
            # Scale 2: spatial = 64,64,48
            # Scale 3: spatial = 128,128,96
            # Full: spatial = 256,256,192
            return scale_outputs + [recon_full]
        else:
            return recon_full

    def forward(
        self,
        x: Tensor,
        age: Tensor | None = None,
        sex: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full VAE forward pass.

        Args:
            x: Input MRI tensor of shape [B, 1, 256, 256, 192]
            age: Optional age values of shape [B], float. If provided with sex, uses demographic conditioning.
            sex: Optional sex labels of shape [B], int (0/1/2). If provided with age, uses demographic conditioning.

        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        if age is not None and sex is not None and self.use_demographic_cond:
            return self.forward_with_demographic(x, age, sex)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def forward_with_demographic(
        self,
        x: Tensor,
        age: Tensor,
        sex: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full VAE forward pass with demographic conditioning.

        Args:
            x: Input MRI tensor of shape [B, 1, 256, 256, 192]
            age: Age values of shape [B], float
            sex: Sex labels of shape [B], int (0/1/2)

        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        if self.use_demographic_cond:
            mu, logvar = self.condition_on_demographic(mu, logvar, age, sex)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_latent(self, x: Tensor) -> Tensor:
        """
        Encode and return latent representation without reparameterization.

        Useful for extracting fixed latent features for downstream tasks
        like disease classification. Always returns mu (no stochasticity).

        Args:
            x: Input MRI tensor of shape [B, 1, 256, 256, 192]

        Returns:
            Latent representation of shape [B, latent_channels, 16, 16, 12]
        """
        mu, _ = self.encode(x)
        return mu


def vae_kl_loss(
    mu: Tensor,
    logvar: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute KL divergence loss for VAE latent regularization.

    KL divergence between N(mu, sigma) and N(0, 1):
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2) over spatial dims
        Then mean over batch for balanced scale with reconstruction loss.

    Args:
        mu: Mean of latent distribution, shape [B, C, D, H, W]
        logvar: Log variance of latent distribution, shape [B, C, D, H, W]
        reduction: Reduction method for loss. Options: "mean", "sum", "none".
                   Default: "mean" (recommended for balancing with reconstruction loss)

    Returns:
        Scalar KL divergence loss (balanced scale for HD inputs)
    """
    # Sum over channel and spatial dimensions, then mean over batch
    # This keeps spatial structure but normalizes across batch size
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2, 3, 4))

    if reduction == "mean":
        return torch.mean(kl_per_sample)
    elif reduction == "sum":
        return torch.sum(kl_per_sample)
    elif reduction == "none":
        return kl_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean', 'sum', or 'none'.")


def vae_reconstruction_loss(
    recon: Tensor,
    target: Tensor,
    loss_type: str = "l1",
) -> Tensor:
    """
    Compute reconstruction loss between predicted and target MRI.

    Args:
        recon: Reconstructed MRI of shape [B, 1, D, H, W]
        target: Target MRI of shape [B, 1, D, H, W]
        loss_type: Type of loss ("l1" or "l2"). Default: "l1"

    Returns:
        Scalar reconstruction loss
    """
    if loss_type == "l1":
        return F.l1_loss(recon, target)
    elif loss_type == "l2":
        return F.mse_loss(recon, target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'l1' or 'l2'.")


class ModalityEncoder3D(nn.Module):
    """
    3D CNN Encoder for a single modality.

    Used as building block for MultiModalVAE3D. Each modality gets its own encoder
    to capture modality-specific features, followed by fusion to a shared latent space.

    Optional multi-axis 3D attention: when use_attention=True, a MultiAxisAttention3D
    block is inserted AFTER the last (DownBlock + ResidualBlock) pair at each index
    in attention_levels.  The block is zero-initialized so it starts as an identity
    function -> safe to add without changing the existing training dynamics on the
    first iteration.

    Stage-wise channel map (default base_channels=16, num_downsamples=4):
        after conv_in   :  16
        after stage 0   :  32   (spatial /2)
        after stage 1   :  64   (spatial /4)
        after stage 2   : 128   (spatial /8)
        after stage 3   : 256   (spatial /16)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        num_downsamples: int = 4,
        latent_channels: int = 64,
        use_attention: bool = True,
        attention_levels: tuple = (3,),
        attention_heads: int = 8,
        use_checkpointing: bool = False,
    ) -> None:
        """
        Initialize the modality encoder.

        Args:
            in_channels: Number of input channels for this modality
            base_channels: Base channel count for conv layers
            num_downsamples: Number of downsampling blocks (default: 4)
            latent_channels: Number of channels in output latent
            use_attention: If True, insert MultiAxisAttention3D at the levels
                           specified by attention_levels. Default: True.
            attention_levels: Tuple of 0-indexed stage numbers where attention
                              is inserted (after the corresponding DownBlock +
                              ResidualBlock). Default: (3,) — only the last
                              (deepest) stage. Use (2, 3) for last two stages.
                              Must be a subset of range(num_downsamples).
            attention_heads: Number of heads per axial attention block. Default 8
                             (auto-reduced if it doesn't divide the channel count).
        """
        super().__init__()

        # Validate attention_levels
        for lvl in attention_levels:
            if not (0 <= lvl < num_downsamples):
                raise ValueError(
                    f"attention_levels entry {lvl} out of range "
                    f"[0, {num_downsamples})"
                )

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_downsamples = num_downsamples
        self.latent_channels = latent_channels
        self.use_attention = use_attention
        # Keep attention_levels sorted so order is deterministic
        self.attention_levels = tuple(sorted(attention_levels))
        self.attention_heads = attention_heads
        # OOM fix for multi-modal: wrap the encoder layer stack in
        # checkpoint_sequential during training to reduce peak memory.
        # See forward() below for the trade-off discussion.
        self.use_checkpointing = use_checkpointing

        # Pre-block (no change from before)
        self.encoder_conv_in = nn.Conv3d(
            in_channels, base_channels, kernel_size=3, padding=1
        )
        encoder_norm_groups = min(8, base_channels)
        if base_channels % encoder_norm_groups != 0:
            encoder_norm_groups = 1
        self.encoder_norm_in = nn.GroupNorm(encoder_norm_groups, base_channels)
        self.encoder_act_in = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Stages: each stage = DownBlock + ResidualBlock.  We keep the same
        # structure (encoder_layers ModuleList) for backward-compat with existing
        # checkpoints; attention is added as a separate nn.ModuleList so checkpoint
        # state_dict keys for encoder_layers.* remain unchanged.
        self.encoder_layers = nn.ModuleList()
        ch = base_channels
        for _ in range(num_downsamples):
            self.encoder_layers.append(
                DownBlock3D(ch, ch * 2, num_groups=8)
            )
            self.encoder_layers.append(
                ResidualBlock3D(ch * 2, ch * 2, num_groups=8)
            )
            ch *= 2

        # Optional multi-axis attention: one block per requested stage.
        # Each block is zero-initialized at construction so the layer is an
        # exact identity for the first forward pass — see models/attention_3d.py.
        self.attention_blocks = nn.ModuleList()
        if use_attention:
            ch_at_stage = [base_channels * (2 ** (i + 1)) for i in range(num_downsamples)]
            for lvl in self.attention_levels:
                self.attention_blocks.append(
                    MultiAxisAttention3D(
                        channels=ch_at_stage[lvl],
                        num_heads=attention_heads,
                    )
                )

        self.latent_conv = nn.Conv3d(
            ch, latent_channels, kernel_size=3, padding=1
        )

        # Adaptive pooling to ensure fixed output size regardless of input
        # Original latent size for 256x256x192 input is (16, 16, 12) after 4 downsamples
        self.pool = nn.AdaptiveAvgPool3d((16, 16, 12))

    def forward(self, x: Tensor) -> Tensor:
        """
        Encode input to latent space.

        Args:
            x: Input tensor of shape [B, in_channels, D, H, W]

        Returns:
            Latent tensor of shape [B, latent_channels, 16, 16, 12]
            (after adaptive pool)
        """
        h = self.encoder_conv_in(x)
        h = self.encoder_norm_in(h)
        h = self.encoder_act_in(h)

        # Build an iterator over attention blocks, applied in attention_levels order.
        # Each stage = 2 layers in encoder_layers; after the 2nd layer of a stage we
        # are at the end of that stage. If that stage is in attention_levels, we
        # pop the next attention block and apply it.
        if self.use_attention and len(self.attention_blocks) > 0:
            attn_iter = iter(self.attention_blocks)
        else:
            attn_iter = None
        # Map: stage index -> attention block (or None if not requested at that stage)
        stage_to_block = {lvl: blk for lvl, blk in zip(self.attention_levels, self.attention_blocks)}

        # OOM fix for multi-modal: when use_checkpointing is on (and we are
        # training), wrap the encoder layer stack in checkpoint_sequential.
        # Without this, with 5 modality encoders (T1 + fMRI + ASL + QSM +
        # FLAIR) the autograd graph holds intermediate activations at full
        # 256^3 resolution for every encoder, blowing past 24GB on RTX 3090
        # at batch=1. Recomputing on backward trades ~15% of wall time for
        # ~40% peak memory reduction. Single-modality T1 doesn't trigger
        # this path because the decoder checkpoint already keeps peak under
        # 14GB, but multi-modal needs both encoder AND decoder wrapped.
        if self.training and getattr(self, 'use_checkpointing', False) and len(self.encoder_layers) > 0:
            from torch.utils.checkpoint import checkpoint_sequential
            h = checkpoint_sequential(self.encoder_layers, 1, h, use_reentrant=False)
        else:
            for li, layer in enumerate(self.encoder_layers):
                h = layer(h)
                # After even-indexed entry (1, 3, 5, ...) we just finished a ResBlock,
                # which is the end of a stage. stage_idx = (li + 1) // 2 - 1
                if (li + 1) % 2 == 0:
                    stage_idx = (li + 1) // 2 - 1
                    if attn_iter is not None and stage_idx in stage_to_block:
                        h = stage_to_block[stage_idx](h)

        latent = self.latent_conv(h)
        latent = self.pool(latent)
        return latent


class ModalityDropout(nn.Module):
    """
    Modality Dropout for handling missing modalities during training.

    During training, randomly drops optional modalities with probability p.
    This makes the model robust to missing modalities at inference time.
    T1 (required modality) is never dropped.
    """

    def __init__(self, p: float = 0.2):
        """
        Initialize modality dropout.

        Args:
            p: Probability of dropping an optional modality during training.
               Only applied when model.training = True.
        """
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply dropout to modality tensor.

        Args:
            x: Input tensor

        Returns:
            Tensor with random zeroing applied to optional modalities
        """
        if not self.training:
            return x
        if torch.rand(1).item() < self.p:
            return torch.zeros_like(x)
        return x


class MultiModalVAE3D(nn.Module):
    """
    Multi-Modal 3D VAE for Alzheimer's Disease MRI Analysis.

    Supports T1 (required) + optional modalities (fMRI, ASL, QSM, FLAIR).
    Each modality has its own encoder, then features are fused via concat + linear
    projection to a shared latent space.

    Fusion strategy:
        - T1 encoder (required, never dropped)
        - Optional modality encoders with dropout during training
        - Concat all modality latents + Linear projection to unified latent

    Training:
        - Stage 1: Multi-modal encoder + decoder + classifier
        - Loss: recon_loss + cls_loss + kl_loss

    Inference:
        - Any subset of optional modalities can be present
        - T1 must always be present
        - Latent is projected to unified space regardless of which modalities are available

    Attributes:
        modalities: List of available modality names
        spatial_size: Original spatial dimensions
        latent_channels: Base latent channels per modality
        total_latent_channels: Total channels after concat fusion
    """

    MODALITY_SIZES = {
        "t1": (256, 256, 192),
        "fmri": (34, 64, 64),
        "asl": (128, 128, 36),
        "qsm": (192, 192, 128),
        "flair": (256, 256, 192),
    }

    def __init__(
        self,
        spatial_size: Tuple[int, int, int] = (256, 256, 192),
        in_channels: int = 1,
        latent_channels: int = 32,
        base_channels: int = 16,
        num_classes: int = 4,
        dropout_rate: float = 0.2,
        decoder_depth: int = 3,
        optional_modalities: Optional[List[str]] = None,
        use_attention: bool = True,
        attention_levels: Tuple[int, ...] = (3,),
        attention_heads: int = 8,
        use_fmri_temporal: bool = True,
        fmri_in_channels: int = 34,
        fmri_hidden_dim: int = 128,
        fmri_num_pool: int = 3,
        fmri_num_transformer_layers: int = 2,
        fmri_num_heads: int = 4,
        use_demographic_cond: bool = False,
        age_emb_dim: int = 16,
        use_checkpointing: bool = False,
        sex_emb_dim: int = 8,
    ) -> None:
        """
        Initialize the Multi-Modal VAE.

        Args:
            spatial_size: Target spatial dimensions for T1 (D, H, W). Default: (256, 256, 192)
            in_channels: Number of input channels (1 for T1 MRI)
            latent_channels: Base latent channels per modality encoder. Default: 32
            base_channels: Base channel count for encoder. Default: 16
            num_classes: Number of disease classes (4: NC, SCD, MCI, AD). Default: 4
            dropout_rate: Probability of dropping optional modalities during training. Default: 0.2
            decoder_depth: Number of decoder upsampling blocks. Default: 3
            optional_modalities: List of optional modality names to use. Default: ['fmri', 'asl', 'qsm', 'flair']
            use_attention: Insert multi-axis 3D attention into each modality
                           encoder. Default: True.
            attention_levels: Stages at which to insert attention (passed through
                              to each ModalityEncoder3D). Default: (3,) — the
                              deepest (bottleneck) stage only.
            attention_heads: Number of attention heads per axial block. Default 8.
            use_fmri_temporal: If True and 'fmri' is in optional_modalities, use
                               the lightweight 1D conv + Transformer fMRI encoder
                               (fMRITemporalEncoder) instead of the static 3D CNN.
                               This preserves the full BOLD time series.
                               Default: True.
            fmri_in_channels: Number of "spatial channel" slices for the fMRI
                              temporal encoder (typically 34 = W axis of (D,H,W)
                              fMRI after spatial avg pool). Default 34.
            fmri_hidden_dim: Hidden dim of fMRI 1D conv stack. Default 128.
            fmri_num_pool: Number of 1D conv blocks (each halves T). Default 3.
            fmri_num_transformer_layers: TransformerEncoder depth. Default 2.
            fmri_num_heads: Multi-head attention heads in the transformer. Default 4.
            use_demographic_cond: If True, condition the latent on age and sex
                embeddings (additive, opt-in). Works for any combination of
                optional modalities (T1-only, T1+FLAIR, etc.). Default: False.
            age_emb_dim: Embedding dim for age (float, normalized 0-1). Default 16.
            sex_emb_dim: Embedding dim for sex (categorical: 0=unknown, 1=male, 2=female). Default 8.
            use_checkpointing: If True, wrap the decoder layer stack in
                `torch.utils.checkpoint.checkpoint_sequential`. Trades a small
                amount of extra compute (~15% slower per epoch) for a large
                reduction in autograd-graph memory (peak activation memory
                drops by ~40% on 256^3 inputs). Defaults to False to keep
                behavior identical to the original implementation; enable
                via the YAML `model.use_checkpointing: true` or the
                `--use_checkpointing` CLI flag when OOM is hit on the
                forward pass.
        """
        super().__init__()

        self.spatial_size = spatial_size
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.decoder_depth = decoder_depth
        self.use_fmri_temporal = use_fmri_temporal
        self.use_demographic_cond = use_demographic_cond
        self.use_checkpointing = use_checkpointing

        # Default optional modalities (T1 is always required, never in this list)
        if optional_modalities is None:
            optional_modalities = ["fmri", "asl", "qsm", "flair"]
        # Defensive dedup + filter to known modalities
        known_optionals = {"fmri", "asl", "qsm", "flair"}
        self.optional_modalities = [m for m in optional_modalities if m in known_optionals]

        # Demographic conditioning modules (only built if use_demographic_cond=True)
        if use_demographic_cond:
            # age: scalar float per sample, mapped to age_emb_dim via small MLP
            self.age_mlp = nn.Sequential(
                nn.Linear(1, age_emb_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(age_emb_dim, age_emb_dim),
            )
            # sex: int (0=unknown, 1=male, 2=female) — embed to sex_emb_dim
            self.sex_emb = nn.Embedding(num_embeddings=4, embedding_dim=sex_emb_dim)
            # Combine: project to a per-channel additive bias of size latent_channels
            self.demo_proj = nn.Linear(age_emb_dim + sex_emb_dim, latent_channels)

        # T1 encoder (required, always present)
        self.encoder_t1 = ModalityEncoder3D(
            in_channels=1,
            base_channels=base_channels,
            num_downsamples=4,
            latent_channels=latent_channels,
            use_attention=use_attention,
            attention_levels=attention_levels,
            attention_heads=attention_heads,
            use_checkpointing=use_checkpointing,
        )

        # Optional modality encoders
        # 'fmri' can use either the static 3D encoder (legacy, time-averaged)
        # or the new fMRITemporalEncoder (preserves BOLD time series).
        self.optional_encoders = nn.ModuleDict()
        self.optional_dropouts = nn.ModuleDict()

        for mod in optional_modalities:
            if mod == "fmri" and use_fmri_temporal and fMRITemporalEncoder is not None:
                # New: lightweight 1D temporal encoder (preserves BOLD time)
                self.optional_encoders[mod] = fMRITemporalEncoder(
                    in_channels=fmri_in_channels,
                    hidden_dim=fmri_hidden_dim,
                    embed_dim=latent_channels,  # match T1 latent_channels for fusion
                    num_pool=fmri_num_pool,
                    target_t=16,
                    num_transformer_layers=fmri_num_transformer_layers,
                    num_heads=fmri_num_heads,
                    target_grid=(16, 16, 12),  # match T1 latent grid
                )
            else:
                # Static 3D encoder (also used for ASL/QSM/FLAIR)
                self.optional_encoders[mod] = ModalityEncoder3D(
                    in_channels=1,
                    base_channels=base_channels,
                    num_downsamples=4,
                    latent_channels=latent_channels,
                    use_attention=use_attention,
                    attention_levels=attention_levels,
                    attention_heads=attention_heads,
                    use_checkpointing=use_checkpointing,
                )
            self.optional_dropouts[mod] = ModalityDropout(p=dropout_rate)

        # Number of modalities that contribute to fusion
        # 1 (T1) + len(optional_modalities)
        self.num_modalities = 1 + len(optional_modalities)
        total_latent_channels = latent_channels * self.num_modalities

        # Fusion: Concat + Linear projection to unified latent space
        # Latent spatial size after 4 downsamples: [16, 16, 12] for 256x256x192 input
        # For different modality sizes, we resize to this common size
        self.fusion_proj = nn.Sequential(
            nn.Conv3d(total_latent_channels, latent_channels * 4, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(latent_channels * 4, latent_channels, kernel_size=1),
        )

        # Log-var projection for true VAE (enables KL loss)
        self.logvar_proj = nn.Sequential(
            nn.Conv3d(total_latent_channels, latent_channels * 4, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(latent_channels * 4, latent_channels, kernel_size=1),
        )

        # Decoder (mirrors encoder structure)
        # Input: [B, latent_channels, 16, 16, 12]
        # Output: [B, 1, 256, 256, 192]
        self._setup_decoder(latent_channels, decoder_depth)

        # Disease classifier head
        # Takes pooled latent features and predicts disease stage
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_channels, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def _setup_decoder(self, latent_channels: int, decoder_depth: int) -> None:
        """Setup decoder layers for multi-modal VAE.

        Latent input is [B, latent_channels, 16, 16, 12] after adaptive pooling.
        Decoder upsamples to T1 size [256, 256, 192].

        With decoder_depth=4: 4 upsampling blocks = 2^4 = 16x upsampling
        16*16 = 256, 12*16 = 192. Perfect match!
        """
        ch = latent_channels  # 32

        # Project latent channels to decoder starting channels
        self.decoder_latent_conv = nn.Conv3d(
            latent_channels, ch * 4, kernel_size=3, padding=1  # 32 -> 128
        )
        ch = ch * 4  # 128

        self.decoder_layers = nn.ModuleList()
        for _ in range(decoder_depth):
            self.decoder_layers.append(
                ResidualBlock3D(ch, ch, num_groups=4)
            )
            self.decoder_layers.append(
                UpBlock3D(ch, ch // 2, num_groups=4)  # 128 -> 64 -> 32 -> 16 -> 8
            )
            ch = ch // 2

        # No final upsample needed - decoder_depth=4 gives exactly 16x upsampling
        self.final_upsample = None

        self.decoder_conv_out = nn.Sequential(
            nn.Conv3d(ch, self.base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, self.base_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(self.base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def _resize_latent(self, z: Tensor, target_size: Tuple[int, int, int]) -> Tensor:
        """
        Resize latent tensor to target spatial size using interpolation.

        Args:
            z: Latent tensor [B, C, D, H, W]
            target_size: Target size (D, H, W)

        Returns:
            Resized tensor
        """
        if z.shape[2:] == target_size:
            return z
        return F.interpolate(z, size=target_size, mode="trilinear", align_corners=False)

    def encode(self, x_dict: Dict[str, Tensor]) -> Tensor:
        """
        Encode multi-modal inputs to unified latent space.

        Args:
            x_dict: Dictionary of modality tensors, e.g.:
                {"t1": tensor, "fmri": tensor, "asl": tensor, ...}
                T1 is required, others are optional.

        Returns:
            Unified latent tensor of shape [B, latent_channels, 8, 8, 6]
        """
        latent_list = []

        # T1 is required
        z_t1 = self.encoder_t1(x_dict["t1"])
        latent_list.append(z_t1)

        # Optional modalities with dropout
        for mod in self.optional_modalities:
            if mod in x_dict and x_dict[mod] is not None:
                z = self.optional_encoders[mod](x_dict[mod])
                z = self.optional_dropouts[mod](z)  # Apply dropout during training
                latent_list.append(z)
            else:
                # Modality not available, add zeros
                z = torch.zeros_like(z_t1)
                latent_list.append(z)

        # Concat all latent tensors
        z_concat = torch.cat(latent_list, dim=1)

        # Project to unified latent space (returns concat for mu + logvar)
        return z_concat

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent to T1 reconstruction.

        Args:
            z: Latent tensor of shape [B, latent_channels, 16, 16, 12]

        Returns:
            Reconstructed T1 of shape [B, 1, 256, 256, 192]
        """
        h = self.decoder_latent_conv(z)

        # OOM fix: when use_checkpointing is on (and we are training),
        # wrap the decoder layer stack in torch.utils.checkpoint. The
        # decoder's intermediate activations (multiple feature maps at
        # full 256^3 resolution) account for ~16GB of autograd-graph
        # memory in the default config. Checkpointing recomputes those
        # activations on the backward pass, trading ~15% of wall time
        # for ~40% peak memory reduction.
        if self.training and self.use_checkpointing and len(self.decoder_layers) > 0:
            from torch.utils.checkpoint import checkpoint_sequential
            # checkpoint_sequential splits the modules into chunks and
            # runs each chunk in a no_grad forward, re-attaching the
            # graph on backward. We use a chunk size of 1 to get the
            # maximum memory savings (one layer at a time).
            h = checkpoint_sequential(self.decoder_layers, 1, h, use_reentrant=False)
        else:
            for layer in self.decoder_layers:
                h = layer(h)

        if self.final_upsample is not None:
            h = self.final_upsample(h)
        recon = self.decoder_conv_out(h)

        return recon

    def classify(self, z: Tensor) -> Tensor:
        """
        Classify disease stage from latent.

        Args:
            z: Latent tensor of shape [B, latent_channels, 16, 16, 12]

        Returns:
            Classification logits of shape [B, num_classes]
        """
        # Pool to single value per channel, then flatten
        pooled = F.adaptive_avg_pool3d(z, output_size=(1, 1, 1))  # [B, latent_channels, 1, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # [B, latent_channels]
        logits = self.classifier(pooled)
        return logits

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick for VAE sampling.

        In training: z = mu + std * epsilon
        In eval: z = mu (deterministic)

        Args:
            mu: Mean [B, C, D, H, W]
            logvar: Log variance [B, C, D, H, W]

        Returns:
            Sampled latent
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        return_components: bool = False,
        age: Optional[Tensor] = None,
        sex: Optional[Tensor] = None,
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """
        Full forward pass.

        Args:
            x_dict: Dictionary of modality tensors. T1 is required.
                    Can optionally contain "age" (float [B]) and "sex" (long [B])
                    keys for demographic conditioning — they will be popped out
                    and applied additively to the latent.
            return_components: If True, return (recon, cls_logits, mu, logvar).
                              If False, return (recon, mu, logvar).
            age: Optional age tensor [B] (float, in years). If provided AND
                 x_dict also has "age", x_dict wins. Used only if
                 use_demographic_cond=True was set at __init__.
            sex: Optional sex tensor [B] (long, 0=unknown/1=male/2=female).
                 If provided AND x_dict also has "sex", x_dict wins.

        Returns:
            When return_components=False:
                Tuple of (reconstruction, mu, logvar)
            When return_components=True:
                Tuple of (reconstruction, cls_logits, mu, logvar)

        Note on DataParallel:
            age/sex CAN be passed as kwargs OR as x_dict["age"]/x_dict["sex"].
            Both are supported, but passing them as x_dict keys is the
            PREFERRED path: MultiModalDataParallel.scatter will then
            split age/sex to match each replica's batch slice. When passed
            as kwargs, kwargs are NOT scattered, so each replica sees the
            full batch — which can cause broadcast issues (mu[B=1] +
            age[B=2] -> mu[B=2], doubling downstream batch size).
        """
        # Demographic inputs can come from either kwargs or x_dict keys.
        # x_dict keys win if both are present (and are popped out so they
        # don't leak into encoders).
        if "age" in x_dict:
            age = x_dict.pop("age")
        if "sex" in x_dict:
            sex = x_dict.pop("sex")

        # Encode multi-modal inputs
        z_concat = self.encode(x_dict)

        # True VAE: learn both mu and logvar
        mu = self.fusion_proj(z_concat)
        logvar = self.logvar_proj(z_concat)

        # Optional demographic conditioning: additive per-channel bias on mu/logvar
        # before reparameterization. Adds age/sex info without disturbing the
        # T1+optional-modality visual signal.
        if self.use_demographic_cond and age is not None and sex is not None:
            demo_bias = self._demographic_bias(age, sex, mu.shape[0])  # [B, C]
            demo_bias = demo_bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1, 1]
            mu = mu + demo_bias
            # logvar shift is conservative (only the bias direction); keeping logvar
            # un-shifted is also valid. We apply a soft 0.1x scale of the bias to
            # logvar to keep gradient flow stable.
            logvar = logvar + 0.1 * demo_bias

        z_sample = self.reparameterize(mu, logvar)

        # Decode to T1 reconstruction
        recon = self.decode(z_sample)

        # Classify disease stage
        cls_logits = self.classify(z_sample)

        if return_components:
            return recon, cls_logits, mu, logvar
        return recon, mu, logvar

    def _demographic_bias(self, age: Tensor, sex: Tensor, batch_size: int) -> Tensor:
        """
        Compute per-channel demographic bias vector.

        DataParallel note: under nn.DataParallel, x_dict (positional input) is
        auto-scattered to each replica's device, but kwargs (age, sex) are
        NOT auto-scattered. The recommended path is to pass them as
        x_dict["age"] / x_dict["sex"] so scatter splits them per replica.
        This method also defensively handles the kwargs path: it slices
        age/sex down to the local replica's batch size to avoid broadcast
        inflation of mu.

        Args:
            age: [B_full] or [B_local] float, in years
            sex: [B_full] or [B_local] long (0=unknown, 1=male, 2=female)
            batch_size: Local batch size of this replica (mu.shape[0]).

        Returns:
            bias: [B_local, latent_channels] on the same device as self.age_mlp
        """
        target_device = self.age_mlp[0].weight.device
        age = age.to(device=target_device, dtype=torch.float32)
        sex = sex.to(device=target_device, dtype=torch.long)

        # Defensive: if age/sex have more samples than the local batch
        # (e.g. kwargs path under DataParallel, full batch is broadcast to
        # all replicas), take the FIRST batch_size samples as a best-effort
        # match. The recommended fix is to pass age/sex via x_dict instead.
        if age.shape[0] != batch_size:
            age = age[:batch_size]
            sex = sex[:batch_size]

        # Normalize age to ~[0, 1] range (AD patients are typically 55-85, divide by 100)
        age_norm = (age / 100.0).unsqueeze(-1)  # [B, 1]
        age_feat = self.age_mlp(age_norm)  # [B, age_emb_dim]
        sex_feat = self.sex_emb(sex)  # [B, sex_emb_dim]
        demo = torch.cat([age_feat, sex_feat], dim=-1)  # [B, age+sex]
        bias = self.demo_proj(demo)  # [B, latent_channels]
        return bias

    def get_latent(
        self,
        x_dict: Dict[str, Tensor],
        age: Optional[Tensor] = None,
        sex: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode and return latent without reparameterization.

        Useful for extracting fixed latent features for downstream tasks
        like disease classification. Returns mu (deterministic in eval).

        Args:
            x_dict: Dictionary of modality tensors
            age: Optional age tensor [B]. Used only if use_demographic_cond=True.
            sex: Optional sex tensor [B]. Used only if use_demographic_cond=True.

        Returns:
            Latent representation [B, latent_channels, 16, 16, 12]
        """
        z_concat = self.encode(x_dict)
        mu = self.fusion_proj(z_concat)
        if self.use_demographic_cond and age is not None and sex is not None:
            demo_bias = self._demographic_bias(age, sex, mu.shape[0])
            demo_bias = demo_bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mu = mu + demo_bias
        return mu