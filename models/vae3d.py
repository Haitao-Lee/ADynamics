"""
3D Variational Autoencoder (VAE) for ADynamics.

Implements a 3D CNN-based VAE for learning compressed latent representations
of T1-weighted MRI scans. Uses GroupNorm (more stable than BatchNorm for small
batch sizes common in 3D medical imaging) and LeakyReLU activation.

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


class ResidualBlock3D(nn.Module):
    """
    3D Residual Block with GroupNorm and LeakyReLU.

    Implements a standard residual block with two 3D convolution layers,
    group normalization, and residual connection.
    """

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
    ) -> None:
        """
        Initialize a 3D upsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (typically in_channels/2)
            num_groups: Number of groups for GroupNorm
            leakyrelu_slope: Negative slope for LeakyReLU
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with upsampling.

        Args:
            x: Input tensor of shape [B, C, D, H, W]

        Returns:
            Output tensor of shape [B, C_out, D*2, H*2, W*2]
        """
        return self.block(x)


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
        """
        super().__init__()

        self.spatial_size = spatial_size
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.use_checkpointing = use_checkpointing

        # With 4 downsampling blocks: 256 -> 128 -> 64 -> 32 -> 16
        # Latent spatial size for HD (256, 256, 192): [16, 16, 12]
        num_downsamples = 4

        # Encoder
        self.encoder_conv_in = nn.Conv3d(
            in_channels, base_channels, kernel_size=3, padding=1
        )
        self.encoder_norm_in = nn.GroupNorm(8, base_channels)
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

        # Decoder
        self.decoder_latent_conv = nn.Conv3d(
            latent_channels, ch, kernel_size=3, padding=1
        )

        self.decoder_layers = nn.ModuleList()
        for _ in range(num_downsamples):
            self.decoder_layers.append(
                ResidualBlock3D(ch, ch, num_groups=8)
            )
            self.decoder_layers.append(
                UpBlock3D(ch, ch // 2, num_groups=8)
            )
            ch //= 2

        self.decoder_conv_out = nn.Sequential(
            nn.Conv3d(ch, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1),
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

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representation back to MRI space.

        Args:
            z: Latent tensor of shape [B, latent_channels, 16, 16, 12]

        Returns:
            Reconstructed MRI of shape [B, 1, 256, 256, 192]

        Note:
            When use_checkpointing=True, gradient checkpointing is applied to
            decoder layers to save GPU memory. This trades compute for memory,
            approximately halving memory usage at the cost of ~20-30% slower training.
        """
        h = self.decoder_latent_conv(z)
        # shape: [B, 512, 16, 16, 12]

        # Apply gradient checkpointing for memory-constrained HD inputs
        # This saves significant GPU memory (~50%) at cost of ~20-30% slower training
        if self.use_checkpointing:
            for layer in self.decoder_layers:
                h = torch.utils.checkpoint.checkpoint(layer, h)
        else:
            for layer in self.decoder_layers:
                h = layer(h)

        recon = self.decoder_conv_out(h)
        # shape: [B, 1, 256, 256, 192]

        return recon

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full VAE forward pass.

        Args:
            x: Input MRI tensor of shape [B, 1, 256, 256, 192]

        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
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