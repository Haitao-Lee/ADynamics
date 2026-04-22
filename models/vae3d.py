"""
3D Variational Autoencoder (VAE) for ADynamics.

Implements a 3D CNN-based VAE for learning compressed latent representations
of T1-weighted MRI scans. Uses GroupNorm (more stable than BatchNorm for small
batch sizes common in 3D medical imaging) and LeakyReLU activation.

Architecture:
    - Encoder: 3D CNN with residual blocks, outputting mu and logvar
    - Reparameterization: z = mu + std * epsilon
    - Decoder: 3D transposed CNN with residual blocks, outputting sigmoid activation
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

        # First conv block
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=leakyrelu_slope, inplace=True)

        # Second conv block
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
        )
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=leakyrelu_slope, inplace=True)

        # Residual connection
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

    The encoder compresses 3D MRI [B, 1, 128, 128, 128] to latent space [B, C, D', H', W'].
    The decoder reconstructs the MRI from latent space.

    Attributes:
        spatial_size: Original spatial dimensions
        in_channels: Number of input channels (1 for T1 MRI)
        latent_channels: Number of channels in latent space
    """

    def __init__(
        self,
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 1,
        latent_channels: int = 64,
    ) -> None:
        """
        Initialize the 3D VAE.

        Args:
            spatial_size: Spatial dimensions of input MRI (D, H, W)
            in_channels: Number of input channels (default: 1 for T1)
            latent_channels: Number of channels in latent representation
        """
        super().__init__()

        self.spatial_size = spatial_size
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Calculate number of downsampling blocks for target latent size of 8x8x8
        # With 4 downsampling blocks: 128 -> 64 -> 32 -> 16 -> 8
        target_latent_size = 8
        num_downsamples = 4
        base_channels = 32

        # Encoder
        self.encoder_conv_in = nn.Conv3d(
            in_channels, base_channels, kernel_size=3, padding=1
        )
        self.encoder_norm_in = nn.GroupNorm(8, base_channels)
        self.encoder_act_in = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Encoder backbone with downsampling and residual blocks
        self.encoder_layers = nn.ModuleList()
        ch = base_channels
        for i in range(num_downsamples):
            self.encoder_layers.append(
                DownBlock3D(ch, ch * 2, num_groups=8)
            )
            self.encoder_layers.append(
                ResidualBlock3D(ch * 2, ch * 2, num_groups=8)
            )
            ch *= 2

        # Latent space conv to produce mu and logvar
        self.latent_conv = nn.Conv3d(
            ch, latent_channels * 2, kernel_size=3, padding=1
        )

        # Decoder
        self.decoder_latent_conv = nn.Conv3d(
            latent_channels, ch, kernel_size=3, padding=1
        )

        # Decoder backbone with upsampling and residual blocks
        self.decoder_layers = nn.ModuleList()
        for i in range(num_downsamples):
            self.decoder_layers.append(
                ResidualBlock3D(ch, ch, num_groups=8)
            )
            self.decoder_layers.append(
                UpBlock3D(ch, ch // 2, num_groups=8)
            )
            ch //= 2

        # Final output conv with Sigmoid activation (data normalized to [0, 1])
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
            x: Input MRI tensor of shape [B, 1, D, H, W]

        Returns:
            Tuple of (mu, logvar) both of shape [B, latent_channels, D', H', W']
        """
        # Input conv
        h = self.encoder_conv_in(x)
        h = self.encoder_norm_in(h)
        h = self.encoder_act_in(h)
        # shape: [B, 32, 128, 128, 128]

        # Encoder backbone
        for layer in self.encoder_layers:
            h = layer(h)

        # Latent projection
        # shape after 4 downsamples: [B, 512, 8, 8, 8]
        latent = self.latent_conv(h)
        # shape: [B, latent_channels*2, 8, 8, 8]

        # Split into mu and logvar
        mu, logvar = latent.chunk(2, dim=1)
        # shape: [B, latent_channels, 8, 8, 8] each

        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick for VAE sampling.

        Args:
            mu: Mean of latent distribution, shape [B, C, D, H, W]
            logvar: Log variance of latent distribution, shape [B, C, D, H, W]

        Returns:
            Sampled latent tensor z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representation back to MRI space.

        Args:
            z: Latent tensor of shape [B, latent_channels, D', H', W']

        Returns:
            Reconstructed MRI of shape [B, 1, D, H, W]
        """
        # Latent projection
        h = self.decoder_latent_conv(z)
        # shape: [B, 512, 8, 8, 8]

        # Decoder backbone
        for layer in self.decoder_layers:
            h = layer(h)

        # Output conv
        recon = self.decoder_conv_out(h)
        # shape: [B, 1, 128, 128, 128]

        return recon

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Full VAE forward pass.

        Args:
            x: Input MRI tensor of shape [B, 1, D, H, W]

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
        like disease classification.

        Args:
            x: Input MRI tensor of shape [B, 1, D, H, W]

        Returns:
            Latent representation of shape [B, latent_channels, D', H', W']
        """
        mu, _ = self.encode(x)
        return mu


def vae_kl_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Compute KL divergence loss for VAE latent regularization.

    KL divergence between N(mu, sigma) and N(0, 1):
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution

    Returns:
        Scalar KL divergence loss
    """
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_div


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
