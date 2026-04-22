"""
Conditional Flow Matching (CFM) Vector Field Network for Stage 3.

Implements a 3D U-Net with FiLM conditioning for learning the velocity field
that morphs NC latent distributions to AD latent distributions.

The network takes:
    - Latent representation z_t (interpolated between NC and AD)
    - Time step t ∈ [0, 1]
    - Clinical conditions (e.g., age)

And outputs the velocity field v_theta(z_t, t, c) that drives the ODE flow.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TimeEmbedding(nn.Module):
    """
    MLP-based Time Embedding for FiLM conditioning.

    Maps time step t ∈ [0, 1] to a conditioning vector that modulates
    the network features via scale (γ) and shift (β) parameters.

    Architecture:
        Linear(t_embed) -> SiLU -> Linear(time_hidden) -> SiLU -> Linear(2 * embed_dim)
        -> Split into γ and β
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initialize time embedding network.

        Args:
            embed_dim: Output embedding dimension (also used for FiLM γ, β)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.embed_dim = embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim * 2),  # γ and β
        )

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-4)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute FiLM parameters γ and β from time step.

        Args:
            t: Time step tensor of shape [B, embed_dim] after sinusoidal embedding

        Returns:
            Tuple of (γ, β) each of shape [B, embed_dim]
        """
        # t shape: [B, embed_dim]
        gamma_beta = self.mlp(t)
        # shape: [B, embed_dim * 2]

        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # each shape: [B, embed_dim]

        return gamma, beta


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding for Time Steps.

    Follows the original Transformer sinusoidal embedding scheme to encode
    time steps t ∈ [0, 1] into a high-dimensional vector.

    Output dimension: embed_dim
    """

    def __init__(
        self,
        embed_dim: int = 128,
    ) -> None:
        """
        Initialize sinusoidal time embedding.

        Args:
            embed_dim: Output embedding dimension. Must be even.
        """
        super().__init__()

        assert embed_dim % 2 == 0, "embed_dim must be even for sinusoidal embedding"

        self.embed_dim = embed_dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Encode time steps to sinusoidal embedding.

        Args:
            t: Time step tensor of shape [B] with values in [0, 1]

        Returns:
            Embedded time tensor of shape [B, embed_dim]
        """
        # t shape: [B]
        B = t.shape[0]

        # Compute frequencies
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(half_dim, device=t.device) / half_dim
        )
        # shape: [half_dim]

        # Compute angle
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0)
        # shape: [B, half_dim]

        # Compute sin and cos
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # shape: [B, embed_dim]

        return emb


class ConditionEmbedding(nn.Module):
    """
    MLP-based Clinical Condition Embedding for FiLM conditioning.

    Takes clinical covariates (e.g., normalized age) and maps them to
    a conditioning vector for feature modulation.

    Architecture:
        Linear(cond_embed) -> SiLU -> Linear(cond_hidden) -> SiLU -> Linear(2 * embed_dim)
        -> Split into γ and β
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_conditions: int = 1,
    ) -> None:
        """
        Initialize condition embedding network.

        Args:
            embed_dim: Output embedding dimension for each condition
            hidden_dim: Hidden layer dimension
            num_conditions: Number of clinical conditions (default: 1 for age)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_conditions = num_conditions

        # Input is num_conditions (e.g., normalized age)
        self.mlp = nn.Sequential(
            nn.Linear(num_conditions, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim * 2),  # γ and β
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-4)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, c: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute FiLM parameters γ and β from clinical conditions.

        Args:
            c: Condition tensor of shape [B, num_conditions] (e.g., normalized age)

        Returns:
            Tuple of (γ, β) each of shape [B, embed_dim]
        """
        # c shape: [B, num_conditions]
        gamma_beta = self.mlp(c)
        # shape: [B, embed_dim * 2]

        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # each shape: [B, embed_dim]

        return gamma, beta


class FiLMLayer3D(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer for 3D Tensors.

    Modulates channel-wise features using scale (γ) and shift (β):
        y = γ * x + β

    The γ and β are computed from conditioning information (time, clinical data).
    """

    def __init__(
        self,
        num_channels: int,
        embed_dim: int,
    ) -> None:
        """
        Initialize FiLM layer.

        Args:
            num_channels: Number of feature channels to modulate
            embed_dim: Dimension of conditioning embedding (γ, β dimension)
        """
        super().__init__()

        self.num_channels = num_channels
        self.embed_dim = embed_dim

        # Project conditioning embedding to γ and β for each channel
        self.gamma_proj = nn.Linear(embed_dim, num_channels)
        self.beta_proj = nn.Linear(embed_dim, num_channels)

        # Initialize projections to identity
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(
        self,
        x: Tensor,
        gamma: Tensor,
        beta: Tensor,
    ) -> Tensor:
        """
        Apply FiLM modulation to input features.

        Args:
            x: Input features of shape [B, C, D, H, W]
            gamma: Scale parameters of shape [B, C] or [B, embed_dim]
            beta: Shift parameters of shape [B, C] or [B, embed_dim]

        Returns:
            Modulated features of shape [B, C, D, H, W]
        """
        # Handle different gamma/beta shapes
        if gamma.dim() == 2 and gamma.shape[1] == self.embed_dim:
            # Project to channel dimension
            gamma = self.gamma_proj(gamma)
            beta = self.beta_proj(beta)
        # else: assume gamma/beta already have shape [B, C]

        # Reshape for broadcasting: [B, C] -> [B, C, 1, 1, 1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Modulate
        y = gamma * x + beta
        # shape: [B, C, D, H, W]

        return y


class ResBlock3D(nn.Module):
    """
    3D Residual Block with GroupNorm and optional FiLM conditioning.

    Architecture:
        GroupNorm -> SiLU -> Conv3d -> GroupNorm -> SiLU -> Conv3d -> (+residual)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: Optional[int] = None,
        num_groups: int = 8,
    ) -> None:
        """
        Initialize 3D residual block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            embed_dim: Embedding dimension for FiLM conditioning. If None, no FiLM
            num_groups: Number of groups for GroupNorm
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim

        # Main path
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        # FiLM conditioning
        if embed_dim is not None:
            self.film = FiLMLayer3D(out_channels, embed_dim)
        else:
            self.film = None

        self.act = nn.SiLU(inplace=True)

    def forward(
        self,
        x: Tensor,
        gamma: Optional[Tensor] = None,
        beta: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with optional FiLM conditioning.

        Args:
            x: Input tensor of shape [B, C, D, H, W]
            gamma: Optional scale parameters for FiLM
            beta: Optional shift parameters for FiLM

        Returns:
            Output tensor of shape [B, C_out, D, H, W]
        """
        residual = self.skip(x)

        # First conv block
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        # shape: [B, C_out, D, H, W]

        # Apply FiLM if conditioning is provided
        if self.film is not None and gamma is not None and beta is not None:
            h = self.film(h, gamma, beta)

        # Second conv block
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        # shape: [B, C_out, D, H, W]

        # Residual connection
        return h + residual


class DownBlock3D(nn.Module):
    """
    3D Downsampling block for U-Net encoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """
        Initialize downsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Downsample input by factor of 2.

        Args:
            x: Input of shape [B, C, D, H, W]

        Returns:
            Output of shape [B, C_out, D/2, H/2, W/2]
        """
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        return h


class UpBlock3D(nn.Module):
    """
    3D Upsampling block for U-Net decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        """
        Initialize upsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, 4, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Upsample input by factor of 2.

        Args:
            x: Input of shape [B, C, D, H, W]

        Returns:
            Output of shape [B, C_out, D*2, H*2, W*2]
        """
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        return h


class VelocityFieldNet(nn.Module):
    """
    Conditional Flow Matching Vector Field Network.

    A 3D U-Net that predicts the velocity field v(z_t, t, c) for morphing
    disease progression in latent space. Uses FiLM conditioning to inject
    time and clinical information.

    Architecture:
        - Encoder: Downsampling with ResBlocks
        - Decoder: Upsampling with skip connections and ResBlocks
        - FiLM: Time and condition embedding injected at each ResBlock

    Input:
        z_t: Latent tensor of shape [B, latent_channels, D, H, W]
        t: Time step of shape [B] (values in [0, 1])
        c: Clinical conditions of shape [B, num_conditions] (e.g., normalized age)

    Output:
        v: Velocity field of shape [B, latent_channels, D, H, W]
    """

    def __init__(
        self,
        latent_channels: int = 64,
        latent_spatial: Tuple[int, int, int] = (8, 8, 8),
        time_embed_dim: int = 128,
        time_hidden_dim: int = 256,
        cond_embed_dim: int = 64,
        cond_hidden_dim: int = 128,
        num_conditions: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, int, int] = (1, 2, 4),
        num_res_blocks: int = 2,
    ) -> None:
        """
        Initialize the Velocity Field Network.

        Args:
            latent_channels: Number of channels in latent representation
            latent_spatial: Spatial dimensions of latent (D, H, W)
            time_embed_dim: Dimension for time embedding
            time_hidden_dim: Hidden dimension for time MLP
            cond_embed_dim: Dimension for condition embedding
            cond_hidden_dim: Hidden dimension for condition MLP
            num_conditions: Number of clinical conditions (e.g., 1 for age)
            base_channels: Base channel count for U-Net
            channel_mults: Channel multipliers for each U-Net level
            num_res_blocks: Number of ResBlocks per U-Net level
        """
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial
        self.time_embed_dim = time_embed_dim
        self.cond_embed_dim = cond_embed_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(embed_dim=time_embed_dim)
        self.time_mlp = TimeEmbedding(embed_dim=time_embed_dim, hidden_dim=time_hidden_dim)

        # Condition embedding
        self.cond_embed = ConditionEmbedding(
            embed_dim=cond_embed_dim,
            hidden_dim=cond_hidden_dim,
            num_conditions=num_conditions,
        )

        # Combined embedding dimension (for FiLM)
        self.film_embed_dim = time_embed_dim + cond_embed_dim

        # Input projection
        self.input_conv = nn.Conv3d(latent_channels, base_channels, 3, padding=1)
        # shape: [B, base_channels, D, H, W]

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()

        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            # ResBlocks
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlock3D(ch, out_ch, embed_dim=self.film_embed_dim)
                )
                ch = out_ch

            # Downsample (except for last level)
            if i < len(channel_mults) - 1:
                self.encoder_downsample.append(DownBlock3D(ch, ch))
                # Store spatial size for skip connections

        # Middle
        self.middle_block = nn.Sequential(
            ResBlock3D(ch, ch, embed_dim=self.film_embed_dim),
            ResBlock3D(ch, ch, embed_dim=self.film_embed_dim),
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult

            # Upsample (except for first level)
            if i > 0:
                self.decoder_upsample.append(UpBlock3D(ch, ch))

            # ResBlocks
            for _ in range(num_res_blocks):
                self.decoder_blocks.append(
                    ResBlock3D(ch, out_ch, embed_dim=self.film_embed_dim)
                )
                ch = out_ch

        # Output projection
        self.output_conv = nn.Conv3d(ch, latent_channels, 3, padding=1)

    def get_time_condition(
        self,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute time conditioning parameters for FiLM.

        Args:
            t: Time step tensor of shape [B] with values in [0, 1]

        Returns:
            Tuple of (gamma, beta) for FiLM conditioning
        """
        # Sinusoidal embedding
        t_embed = self.time_embed(t)
        # shape: [B, time_embed_dim]

        # MLP to get γ and β
        gamma, beta = self.time_mlp(t_embed)
        # each shape: [B, time_embed_dim]

        return gamma, beta

    def get_cond_condition(
        self,
        c: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute condition conditioning parameters for FiLM.

        Args:
            c: Condition tensor of shape [B, num_conditions]

        Returns:
            Tuple of (gamma, beta) for FiLM conditioning
        """
        # MLP to get γ and β
        gamma, beta = self.cond_embed(c)
        # each shape: [B, cond_embed_dim]

        return gamma, beta

    def combine_conditions(
        self,
        time_gamma: Tensor,
        time_beta: Tensor,
        cond_gamma: Tensor,
        cond_beta: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Combine time and condition embeddings for FiLM.

        Concatenates time and condition γ, β and projects to film_embed_dim.

        Args:
            time_gamma: Time scale params [B, time_embed_dim]
            time_beta: Time shift params [B, time_embed_dim]
            cond_gamma: Condition scale params [B, cond_embed_dim]
            cond_beta: Condition shift params [B, cond_embed_dim]

        Returns:
            Combined (gamma, beta) [B, film_embed_dim]
        """
        gamma = torch.cat([time_gamma, cond_gamma], dim=-1)
        beta = torch.cat([time_beta, cond_beta], dim=-1)
        # each shape: [B, time_embed_dim + cond_embed_dim]

        return gamma, beta

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        c: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute velocity field v(z_t, t, c).

        Args:
            z_t: Latent tensor at time t of shape [B, latent_channels, D, H, W]
            t: Time step tensor of shape [B] with values in [0, 1]
            c: Optional clinical conditions of shape [B, num_conditions]

        Returns:
            Velocity field of shape [B, latent_channels, D, H, W]
        """
        # Get conditioning
        time_gamma, time_beta = self.get_time_condition(t)
        # each shape: [B, time_embed_dim]

        if c is not None:
            cond_gamma, cond_beta = self.get_cond_condition(c)
            # each shape: [B, cond_embed_dim]
            gamma, beta = self.combine_conditions(
                time_gamma, time_beta, cond_gamma, cond_beta
            )
            # each shape: [B, time_embed_dim + cond_embed_dim]
        else:
            # Use time conditioning only
            gamma, beta = time_gamma, time_beta
            # each shape: [B, time_embed_dim]

        # Input projection
        h = self.input_conv(z_t)
        # shape: [B, base_channels, D, H, W]

        # Encoder forward with FiLM conditioning
        encoder_outputs = []
        block_idx = 0

        for i, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                h = self.encoder_blocks[block_idx](h, gamma, beta)
                block_idx += 1
            encoder_outputs.append(h)

            if i < len(channel_mults) - 1:
                h = self.encoder_downsample[i](h)

        # Middle
        h = self.middle_block[0](h, gamma, beta)
        h = self.middle_block[1](h, gamma, beta)
        # shape: [B, base_channels * channel_mults[-1], D', H', W']

        # Decoder forward with skip connections and FiLM
        block_idx = 0
        for i, mult in enumerate(reversed(channel_mults)):
            if i > 0:
                h = self.decoder_upsample[i - 1](h)

            for _ in range(num_res_blocks):
                # Concatenate skip connection
                skip_idx = len(channel_mults) - 1 - i
                if block_idx < len(self.decoder_blocks):
                    h = torch.cat([h, encoder_outputs[skip_idx]], dim=1)
                    # This changes channel dim, so we need to project

                h = self.decoder_blocks[block_idx](h, gamma, beta)
                block_idx += 1

        # Output projection
        v = self.output_conv(h)
        # shape: [B, latent_channels, D, H, W]

        return v


def cfm_velocity_loss(
    v_pred: Tensor,
    z0: Tensor,
    z1: Tensor,
) -> Tensor:
    """
    Compute Conditional Flow Matching loss.

    L_CFM = || v_theta(z_t, t) - (z1 - z0) ||^2

    where z_t = (1-t)*z0 + t*z1 is the linear interpolation.

    Args:
        v_pred: Predicted velocity field of shape [B, C, D, H, W]
        z0: Source latent (NC) of shape [B, C, D, H, W]
        z1: Target latent (AD) of shape [B, C, D, H, W]

    Returns:
        Scalar MSE loss
    """
    target_v = z1 - z0
    loss = F.mse_loss(v_pred, target_v)
    return loss
