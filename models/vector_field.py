"""
Conditional Flow Matching (CFM) Vector Field Network for Stage 3.

Implements a 3D U-Net with deep FiLM conditioning for learning the velocity field
that morphs NC latent distributions to AD latent distributions.

The network takes:
    - Latent representation z_t (interpolated between NC and AD), shape [B, C, 16, 16, 12] for HD
    - Time step t ∈ [0, 1]
    - Clinical conditions (e.g., age)

And outputs the velocity field v_theta(z_t, t, c) that drives the ODE flow.

Architecture with deep FiLM:
    - Encoder: DownBlock3D (with FiLM) -> ResBlock3D (with FiLM)
    - Decoder: UpBlock3D (with FiLM) -> ResBlock3D (with FiLM) + skip connection
    - FiLM applied at every spatial scale for better time/condition awareness
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding for Time Steps.

    Follows the original Transformer sinusoidal embedding scheme to encode
    time steps t ∈ [0, 1] into a high-dimensional vector.
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
        B = t.shape[0]
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device, dtype=torch.float16)) * torch.arange(half_dim, device=t.device, dtype=torch.float16) / half_dim
        )
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """
    MLP-based Time Embedding for FiLM conditioning.

    Maps time step t ∈ [0, 1] to a conditioning vector that modulates
    the network features via scale (γ) and shift (β) parameters.

    Architecture:
        Linear(t_embed) -> SiLU -> Linear(time_hidden) -> SiLU -> Linear(2 * embed_dim)
        -> Split into γ and β

    γ, β initialized such that γ ≈ 1.0, β ≈ 0.0 for stable early training.
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

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights with Kaiming normal for better gradient flow.
        γ projection initialized to ~1, β projection initialized to ~0.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        gamma_beta = self.mlp(t)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma, beta


class ConditionEmbedding(nn.Module):
    """
    MLP-based Clinical Condition Embedding for FiLM conditioning.

    Takes clinical covariates (e.g., normalized age) and maps them to
    a conditioning vector for feature modulation.

    Architecture:
        Linear(cond_embed) -> SiLU -> Linear(cond_hidden) -> SiLU -> Linear(2 * embed_dim)
        -> Split into γ and β

    γ, β initialized such that γ ≈ 1.0, β ≈ 0.0 for stable early training.
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

        self.mlp = nn.Sequential(
            nn.Linear(num_conditions, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim * 2),  # γ and β
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        gamma_beta = self.mlp(c)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma, beta


class DemographicsEmbedding(nn.Module):
    """
    Separate embeddings for Age and Sex with ablation support.

    Provides independent learned embeddings for age (continuous) and sex (binary)
    that can be individually enabled/disabled for ablation experiments:
        - age only: both embeddings active, sex embedding masked to zeros
        - sex only: both embeddings active, age embedding masked to zeros
        - both: full demographics conditioning
        - neither: both masked to zeros (baseline)

    Architecture:
        Age: Linear(1) -> SiLU -> Linear(age_embed_dim) -> SiLU -> Linear(2 * age_embed_dim) -> γ_a, β_a
        Sex: Linear(1) -> SiLU -> Linear(sex_embed_dim) -> SiLU -> Linear(2 * sex_embed_dim) -> γ_s, β_s
        Combined: concatenate(γ_a, γ_s), concatenate(β_a, β_s) -> [B, age_embed_dim + sex_embed_dim]

    γ, β initialized such that γ ≈ 1.0, β ≈ 0.0 for stable early training.
    """

    def __init__(
        self,
        age_embed_dim: int = 32,
        sex_embed_dim: int = 16,
        age_hidden_dim: int = 64,
        sex_hidden_dim: int = 32,
    ) -> None:
        """
        Initialize demographics embedding network.

        Args:
            age_embed_dim: Output embedding dimension for age FiLM parameters
            sex_embed_dim: Output embedding dimension for sex FiLM parameters
            age_hidden_dim: Hidden layer dimension for age MLP
            sex_hidden_dim: Hidden layer dimension for sex MLP
        """
        super().__init__()

        self.age_embed_dim = age_embed_dim
        self.sex_embed_dim = sex_embed_dim

        # Age embedding: takes normalized age [0, 1]
        self.age_mlp = nn.Sequential(
            nn.Linear(1, age_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(age_hidden_dim, age_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(age_hidden_dim, age_embed_dim * 2),  # γ and β
        )

        # Sex embedding: takes binary sex (0=female, 1=male)
        self.sex_mlp = nn.Sequential(
            nn.Linear(1, sex_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(sex_hidden_dim, sex_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(sex_hidden_dim, sex_embed_dim * 2),  # γ and β
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming normal, γ ≈ 1, β ≈ 0."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        age: Optional[Tensor] = None,
        sex: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute FiLM parameters γ and β from age and/or sex.

        Args:
            age: Normalized age tensor of shape [B, 1] in [0, 1], or None to mask
            sex: Binary sex tensor of shape [B, 1] (0=female, 1=male), or None to mask
            t: Time tensor [B] to determine batch size when both age and sex are None

        Returns:
            Tuple of (gamma, beta) each of shape [B, age_embed_dim + sex_embed_dim]
            Missing embeddings are masked to zeros (γ=1, β=0 effect via masking)
        """
        B = 0
        if age is not None:
            B = age.shape[0]
        elif sex is not None:
            B = sex.shape[0]
        elif t is not None:
            B = t.shape[0]

        device = None
        dtype = None
        if age is not None:
            device = age.device
            dtype = age.dtype
        elif sex is not None:
            device = sex.device
            dtype = sex.dtype
        elif t is not None:
            device = t.device
            dtype = t.dtype

        # Initialize to zeros (γ=0, β=0 -> identity when multiplied)
        gamma = torch.zeros(B, self.age_embed_dim + self.sex_embed_dim, device=device, dtype=dtype)
        beta = torch.zeros(B, self.age_embed_dim + self.sex_embed_dim, device=device, dtype=dtype)

        offset = 0

        if age is not None:
            gamma_beta_a = self.age_mlp(age)
            gamma_a, beta_a = gamma_beta_a.chunk(2, dim=-1)
            gamma[:, offset:offset + self.age_embed_dim] = gamma_a
            beta[:, offset:offset + self.age_embed_dim] = beta_a
            offset += self.age_embed_dim

        if sex is not None:
            gamma_beta_s = self.sex_mlp(sex)
            gamma_s, beta_s = gamma_beta_s.chunk(2, dim=-1)
            gamma[:, offset:offset + self.sex_embed_dim] = gamma_s
            beta[:, offset:offset + self.sex_embed_dim] = beta_s
            offset += self.sex_embed_dim

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

        self.gamma_proj = nn.Linear(embed_dim, num_channels)
        self.beta_proj = nn.Linear(embed_dim, num_channels)

        # Initialize to identity: γ=1, β=0
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
            gamma: Scale parameters of shape [B, embed_dim]
            beta: Shift parameters of shape [B, embed_dim]

        Returns:
            Modulated features of shape [B, C, D, H, W]
        """
        # Project from embed_dim to channel dimension
        gamma = self.gamma_proj(gamma)
        beta = self.beta_proj(beta)

        # Reshape for broadcasting: [B, C] -> [B, C, 1, 1, 1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        y = gamma * x + beta
        return y


class ResBlock3D(nn.Module):
    """
    3D Residual Block with GroupNorm and FiLM conditioning.

    Architecture:
        PreConv (optional projection) -> GroupNorm -> SiLU -> Conv3d ->
        FiLM -> GroupNorm -> SiLU -> Conv3d -> (+residual)

    FiLM is applied after first convolution to modulate features before
    the second convolution, allowing condition-aware feature refinement.
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

        # Pre-projection for main path channel alignment
        self.pre_conv = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Main path
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv1 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        # Skip connection: project to match output channels for residual addition
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # FiLM conditioning (applied after first conv)
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
        Forward pass with FiLM conditioning.

        Args:
            x: Input tensor of shape [B, C, D, H, W]
            gamma: Optional scale parameters for FiLM
            beta: Optional shift parameters for FiLM

        Returns:
            Output tensor of shape [B, C_out, D, H, W]
        """
        # Pre-projection to match channel dimensions
        h = self.pre_conv(x)

        # First conv block
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)

        # Apply FiLM if conditioning is provided
        if self.film is not None and gamma is not None and beta is not None:
            h = self.film(h, gamma, beta)

        # Second conv block
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        # Residual connection
        return h + self.skip(x)


class DownBlock3D(nn.Module):
    """
    3D Downsampling block for U-Net encoder with FiLM conditioning.

    Applies FiLM modulation after downsampling to inject time/condition
    awareness at each spatial scale.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: Optional[int] = None,
    ) -> None:
        """
        Initialize downsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            embed_dim: Embedding dimension for FiLM conditioning
        """
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU(inplace=True)

        # FiLM after downsampling
        if embed_dim is not None:
            self.film = FiLMLayer3D(out_channels, embed_dim)
        else:
            self.film = None

    def forward(
        self,
        x: Tensor,
        gamma: Optional[Tensor] = None,
        beta: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Downsample input by factor of 2 with FiLM modulation.

        Args:
            x: Input of shape [B, C, D, H, W]
            gamma: Optional scale parameters for FiLM
            beta: Optional shift parameters for FiLM

        Returns:
            Output of shape [B, C_out, D/2, H/2, W/2]
        """
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)

        if self.film is not None and gamma is not None and beta is not None:
            h = self.film(h, gamma, beta)

        return h


class UpBlock3D(nn.Module):
    """
    3D Upsampling block for U-Net decoder with FiLM conditioning.

    Applies FiLM modulation after upsampling to inject time/condition
    awareness at each spatial scale.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: Optional[int] = None,
    ) -> None:
        """
        Initialize upsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            embed_dim: Embedding dimension for FiLM conditioning
        """
        super().__init__()

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, 4, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU(inplace=True)

        # FiLM after upsampling
        if embed_dim is not None:
            self.film = FiLMLayer3D(out_channels, embed_dim)
        else:
            self.film = None

    def forward(
        self,
        x: Tensor,
        gamma: Optional[Tensor] = None,
        beta: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Upsample input by factor of 2 with FiLM modulation.

        Args:
            x: Input of shape [B, C, D, H, W]
            gamma: Optional scale parameters for FiLM
            beta: Optional shift parameters for FiLM

        Returns:
            Output of shape [B, C_out, D*2, H*2, W*2]
        """
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)

        if self.film is not None and gamma is not None and beta is not None:
            h = self.film(h, gamma, beta)

        return h


class VelocityFieldNet(nn.Module):
    """
    Conditional Flow Matching Vector Field Network.

    A 3D U-Net that predicts the velocity field v(z_t, t, c) for morphing
    disease progression in latent space. Uses deep FiLM conditioning
    injected at every encoder/decoder block.

    HD Input: [B, latent_channels, 16, 16, 12] (from VAE with 256x256x192 input)

    Architecture:
        - Encoder: DownBlock3D (FiLM) -> ResBlock3D (FiLM) -> ... (3 levels)
        - Middle: ResBlock3D (FiLM) x 2
        - Decoder: UpBlock3D (FiLM) -> ResBlock3D (FiLM) + skip -> ... (3 levels)
        - FiLM: Time and condition embedding injected at EVERY block

    Channel progression for HD latent [16,16,12] with base=64, mults=(1,2,4):
        Level 0: 64 channels, spatial [16, 16, 12]
        Level 1: 128 channels, spatial [8, 8, 6]
        Level 2: 256 channels, spatial [4, 4, 3]
    """

    def __init__(
        self,
        latent_channels: int = 64,
        latent_spatial: Tuple[int, int, int] = (16, 16, 12),
        time_embed_dim: int = 128,
        time_hidden_dim: int = 256,
        cond_embed_dim: int = 64,
        cond_hidden_dim: int = 128,
        num_conditions: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, int, int] = (1, 2, 4),
        num_res_blocks: int = 2,
        # Demographics embedding options for ablation
        use_demographics: bool = False,
        age_embed_dim: int = 32,
        sex_embed_dim: int = 16,
        age_hidden_dim: int = 64,
        sex_hidden_dim: int = 32,
    ) -> None:
        """
        Initialize the Velocity Field Network.

        Args:
            latent_channels: Number of channels in latent representation
            latent_spatial: Spatial dimensions of latent (D, H, W). Default: (16, 16, 12) for HD
            time_embed_dim: Dimension for time embedding
            time_hidden_dim: Hidden dimension for time MLP
            cond_embed_dim: Dimension for condition embedding
            cond_hidden_dim: Hidden dimension for condition MLP
            num_conditions: Number of clinical conditions (e.g., 1 for age)
            base_channels: Base channel count for U-Net
            channel_mults: Channel multipliers for each U-Net level
            num_res_blocks: Number of ResBlocks per U-Net level
            use_demographics: If True, use separate age/sex embeddings instead of generic condition
            age_embed_dim: Embedding dim for age (only used if use_demographics=True)
            sex_embed_dim: Embedding dim for sex (only used if use_demographics=True)
            age_hidden_dim: Hidden dim for age MLP (only used if use_demographics=True)
            sex_hidden_dim: Hidden dim for sex MLP (only used if use_demographics=True)
        """
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial
        self.time_embed_dim = time_embed_dim
        self.cond_embed_dim = cond_embed_dim
        self.num_res_blocks = num_res_blocks
        self.channel_mults = channel_mults
        self.use_demographics = use_demographics

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(embed_dim=time_embed_dim)
        self.time_mlp = TimeEmbedding(embed_dim=time_embed_dim, hidden_dim=time_hidden_dim)

        # Condition embedding (legacy or when use_demographics=False)
        self.cond_embed = ConditionEmbedding(
            embed_dim=cond_embed_dim,
            hidden_dim=cond_hidden_dim,
            num_conditions=num_conditions,
        )

        # Demographics embedding (age + sex) for ablation experiments
        if use_demographics:
            self.demographics_embed = DemographicsEmbedding(
                age_embed_dim=age_embed_dim,
                sex_embed_dim=sex_embed_dim,
                age_hidden_dim=age_hidden_dim,
                sex_hidden_dim=sex_hidden_dim,
            )
            self.cond_embed_dim = age_embed_dim + sex_embed_dim
        else:
            self.demographics_embed = None

        # Combined embedding dimension (for FiLM)
        self.film_embed_dim = time_embed_dim + self.cond_embed_dim

        # Input projection: [B, C, 16, 16, 12] -> [B, base, 16, 16, 12]
        self.input_conv = nn.Conv3d(latent_channels, base_channels, 3, padding=1)

        # Build encoder with FiLM-aware blocks
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()

        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            # ResBlocks with FiLM: first block uses ch, subsequent blocks use previous out_ch
            for j in range(num_res_blocks):
                if j == 0:
                    in_ch = ch  # First block uses input from previous level
                else:
                    in_ch = out_ch  # Subsequent blocks use previous block's output
                self.encoder_blocks.append(
                    ResBlock3D(in_ch, out_ch, embed_dim=self.film_embed_dim)
                )

            # Downsample with FiLM (except last level)
            if i < len(channel_mults) - 1:
                self.encoder_downsample.append(
                    DownBlock3D(out_ch, out_ch, embed_dim=self.film_embed_dim)
                )
                ch = out_ch

        # Middle with FiLM
        mid_ch = base_channels * channel_mults[-1]
        self.middle_block = nn.ModuleList([
            ResBlock3D(mid_ch, mid_ch, embed_dim=self.film_embed_dim),
            ResBlock3D(mid_ch, mid_ch, embed_dim=self.film_embed_dim),
        ])

        # Build decoder with correct channel handling for skip connections
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        self.decoder_proj = nn.ModuleList()  # Project skip connections to match channels

        # Track decoder block input channels for proper skip connection handling
        # After upsample + concat, channels = up_out_ch + skip_ch (from encoder)
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult

            # Upsample with FiLM (except first level)
            if i > 0:
                self.decoder_upsample.append(
                    UpBlock3D(ch, out_ch, embed_dim=self.film_embed_dim)
                )
                # Projection for skip connection to match channel dimensions
                # At each decoder level, we need skip_ch to match out_ch for proper concat
                # skip_idx maps decoder level i to encoder level (len-1-i)
                skip_idx = len(channel_mults) - 1 - i
                skip_ch = base_channels * channel_mults[skip_idx]
                if skip_ch != out_ch:
                    self.decoder_proj.append(nn.Conv3d(skip_ch, out_ch, 1))
                else:
                    self.decoder_proj.append(nn.Identity())

            for _ in range(num_res_blocks):
                # Compute actual in_channels based on whether we upsample or not
                if i == 0:
                    # First block: no upsample, just middle output
                    block_in_ch = mid_ch
                else:
                    # After upsample: upsample output + projected skip
                    block_in_ch = out_ch + out_ch  # upsample output + skip (projected to out_ch)

                self.decoder_blocks.append(
                    ResBlock3D(block_in_ch, out_ch, embed_dim=self.film_embed_dim)
                )
                ch = out_ch

        # Output projection
        self.output_conv = nn.Conv3d(ch, latent_channels, 3, padding=1)

    def get_time_condition(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute time conditioning parameters for FiLM.

        Args:
            t: Time step tensor of shape [B] with values in [0, 1]

        Returns:
            Tuple of (gamma, beta) for FiLM conditioning
        """
        t_embed = self.time_embed(t)
        gamma, beta = self.time_mlp(t_embed)
        return gamma, beta

    def get_cond_condition(self, c: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute condition conditioning parameters for FiLM.

        Args:
            c: Condition tensor of shape [B, num_conditions]

        Returns:
            Tuple of (gamma, beta) for FiLM conditioning
        """
        gamma, beta = self.cond_embed(c)
        return gamma, beta

    def get_demographics_condition(
        self,
        age: Optional[Tensor] = None,
        sex: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute demographics conditioning parameters for FiLM.

        Supports ablation by allowing age and/or sex to be None.

        Args:
            age: Normalized age tensor [B, 1] in [0, 1], or None to mask
            sex: Binary sex tensor [B, 1] (0=female, 1=male), or None to mask
            t: Time tensor [B] to determine batch size when both age and sex are None

        Returns:
            Tuple of (gamma, beta) for FiLM conditioning
        """
        if self.demographics_embed is None:
            raise RuntimeError(
                "Demographics embedding not enabled. "
                "Set use_demographics=True in constructor."
            )
        return self.demographics_embed(age=age, sex=sex, t=t)

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
        return gamma, beta

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        c: Optional[Tensor] = None,
        age: Optional[Tensor] = None,
        sex: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute velocity field v(z_t, t, c) or v(z_t, t, age, sex).

        Args:
            z_t: Latent tensor at time t of shape [B, latent_channels, 16, 16, 12]
            t: Time step tensor of shape [B] with values in [0, 1]
            c: Clinical conditions of shape [B, num_conditions] (legacy, used when use_demographics=False)
            age: Normalized age tensor [B, 1] in [0, 1] (used when use_demographics=True)
            sex: Binary sex tensor [B, 1], 0=female 1=male (used when use_demographics=True)

        Returns:
            Velocity field of shape [B, latent_channels, 16, 16, 12]
        """
        # Get conditioning - always use combined time+condition (FiLM requires consistent embed_dim)
        time_gamma, time_beta = self.get_time_condition(t)

        if self.use_demographics:
            # Use separate age/sex embeddings for ablation
            cond_gamma, cond_beta = self.get_demographics_condition(age=age, sex=sex, t=t)
        elif c is not None:
            # Use legacy condition embedding
            cond_gamma, cond_beta = self.get_cond_condition(c)
        else:
            # Use zero conditioning when not provided to keep embed_dim consistent
            cond_gamma = torch.zeros(t.size(0), self.cond_embed_dim, device=t.device, dtype=t.dtype)
            cond_beta = torch.zeros(t.size(0), self.cond_embed_dim, device=t.device, dtype=t.dtype)

        gamma, beta = self.combine_conditions(
            time_gamma, time_beta, cond_gamma, cond_beta
        )

        # Input projection
        h = self.input_conv(z_t)
        # shape: [B, base_channels, 16, 16, 12]

        # Encoder forward with FiLM at every block
        encoder_outputs = []
        block_idx = 0

        for i, mult in enumerate(self.channel_mults):
            # ResBlocks
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[block_idx](h, gamma, beta)
                block_idx += 1
            encoder_outputs.append(h)

            # Downsample with FiLM (except last level)
            if i < len(self.channel_mults) - 1:
                h = self.encoder_downsample[i](h, gamma, beta)

        # Middle with FiLM
        for block in self.middle_block:
            h = block(h, gamma, beta)

        # Decoder forward with skip connections and FiLM
        block_idx = 0
        upsample_idx = 0
        proj_idx = 0

        for i, mult in enumerate(reversed(self.channel_mults)):
            # Upsample with FiLM (except first level)
            if i > 0:
                h = self.decoder_upsample[upsample_idx](h, gamma, beta)
                upsample_idx += 1

            # ResBlocks with skip connections
            for _ in range(self.num_res_blocks):
                # At level 0 (deepest), no concatenation - just process middle output
                if i == 0:
                    # No skip connection at deepest level
                    h = self.decoder_blocks[block_idx](h, gamma, beta)
                else:
                    # Concatenate skip connection (with projection if needed)
                    skip_idx = len(self.channel_mults) - 1 - i
                    skip = encoder_outputs[skip_idx]
                    if proj_idx < len(self.decoder_proj):
                        skip = self.decoder_proj[proj_idx](skip)
                        proj_idx += 1
                    h = torch.cat([h, skip], dim=1)
                    h = self.decoder_blocks[block_idx](h, gamma, beta)
                block_idx += 1

        # Output projection
        v = self.output_conv(h)
        # shape: [B, latent_channels, 16, 16, 12]

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