"""
Spatial Transformation Module for ADynamics.

Implements:
    - DeformationGenerator: 3D CNN that generates displacement fields from latent
    - SpatialTransformer: 3D STN that applies deformation to MRI images

Critical for medical imaging:
    - Uses align_corners=False for proper physical coordinate handling
    - Jacobian determinant penalty to prevent folding
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DeformationGenerator(nn.Module):
    """
    3D Deformation Field Generator.

    Takes evolved latent representation z_final from CFM and generates
    a 3D displacement field that warps the NC image toward AD.

    Architecture:
        Latent -> 3D U-Net style upsampling -> 3-channel displacement field

    Output:
        displacement field of shape [B, 3, D, H, W] where:
            channel 0 = displacement in D dimension (depth)
            channel 1 = displacement in H dimension (height)
            channel 2 = displacement in W dimension (width)

    The displacement is in voxel units, can be multiplied by spacing for physical units.
    """

    def __init__(
        self,
        latent_channels: int = 64,
        latent_spatial: Tuple[int, int, int] = (8, 8, 8),
        output_spatial: Tuple[int, int, int] = (128, 128, 128),
        base_channels: int = 64,
        channel_mults: Tuple[int, int, int, int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        max_displacement: float = 10.0,
    ) -> None:
        """
        Initialize Deformation Generator.

        Args:
            latent_channels: Number of channels in latent representation
            latent_spatial: Spatial dimensions of latent (D, H, W)
            output_spatial: Output deformation field spatial dimensions
            base_channels: Base channel count for network
            channel_mults: Channel multipliers for each upsampling level
            num_res_blocks: Number of residual blocks per level
            max_displacement: Maximum displacement in voxels (clamping range)
        """
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_spatial = latent_spatial
        self.output_spatial = output_spatial
        self.max_displacement = max_displacement

        # Initial projection from latent to feature map
        # Start from latent spatial size and upsampe to full size
        self.latent_proj = nn.Sequential(
            nn.Conv3d(latent_channels, base_channels * channel_mults[-1], 3, padding=1),
            nn.GroupNorm(8, base_channels * channel_mults[-1]),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Build decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()

        ch = base_channels * channel_mults[-1]
        self.num_levels = len(channel_mults)

        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult

            # Residual blocks
            for _ in range(num_res_blocks):
                self.decoder_blocks.append(
                    nn.Sequential(
                        nn.GroupNorm(8, ch),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv3d(ch, out_ch, 3, padding=1),
                    )
                )
                ch = out_ch

            # Upsample (except for last level)
            if i < len(channel_mults) - 1:
                self.decoder_upsample.append(
                    nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1)
                )

        # Output head to generate 3-channel displacement
        self.output_head = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ch, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(base_channels, 3, 3, padding=1),
            # Output: [B, 3, D, H, W] displacement field
        )

    def forward(self, z_final: Tensor) -> Tensor:
        """
        Generate deformation field from evolved latent.

        Args:
            z_final: Evolved latent from CFM of shape [B, latent_channels, D', H', W']

        Returns:
            Deformation field of shape [B, 3, D, H, W]
            Values are clamped to [-max_displacement, max_displacement]
        """
        # Project latent to feature space
        h = self.latent_proj(z_final)
        # shape: [B, base_channels * channel_mults[-1], D', H', W']

        # Decoder with upsampling
        block_idx = 0
        for i in range(len(self.decoder_upsample)):
            # Upsample
            h = self.decoder_upsample[i](h)
            # shape: [B, ch, D'*2, H'*2, W'*2]

            # Apply two res blocks
            for _ in range(2):
                h = h + self.decoder_blocks[block_idx](h)
                block_idx += 1

        # Final blocks (no more upsampling)
        for _ in range(2):
            h = h + self.decoder_blocks[block_idx](h)
            block_idx += 1

        # Generate displacement
        flow = self.output_head(h)
        # shape: [B, 3, D, H, W]

        # Clamp displacement to prevent extreme deformations
        flow = torch.clamp(flow, min=-self.max_displacement, max=self.max_displacement)

        return flow


class SpatialTransformer(nn.Module):
    """
    3D Spatial Transformer Network (STN) for applying deformations.

    Uses torch.nn.functional.grid_sample to apply a displacement field
    to a 3D image. The displacement field specifies where to sample from
    in the input image to produce the output.

    CRITICAL: Uses align_corners=False for proper physical coordinate handling.
    With align_corners=False, coordinates are in [-1, 1] representing the
    spatial extent of the input, with 0 being the center.

    The deformation field flow specifies the displacement from the current
    position in normalized coordinates [-1, 1].
    """

    def __init__(
        self,
        mode: str = "bilinear",
        padding_mode: str = "border",
    ) -> None:
        """
        Initialize Spatial Transformer.

        Args:
            mode: Interpolation mode ("bilinear" or "nearest")
            padding_mode: Padding mode for out-of-bound coordinates
                "border" = replicate edge values
                "zeros" = fill with zeros
                "reflection" = reflect values
        """
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(
        self,
        image: Tensor,
        flow: Tensor,
        return_grid: bool = False,
    ) -> Tensor:
        """
        Apply deformation field to image.

        Args:
            image: Input image of shape [B, 1, D, H, W] or [B, C, D, H, W]
            flow: Deformation field of shape [B, 3, D, H, W]
                flow[:, 0] = displacement in D dimension
                flow[:, 1] = displacement in H dimension
                flow[:, 2] = displacement in W dimension
                Values are in voxels, will be converted to normalized coords
            return_grid: Whether to return the sampling grid

        Returns:
            Warped image of shape [B, C, D, H, W]
        """
        B, C, D, H, W = image.shape

        # Create normalized sampling grid [-1, 1]
        # With align_corners=False:
        # -1 corresponds to -D/2, D/2, -H/2, H/2, -W/2, W/2 from center
        # This represents the spatial extent properly

        # Create base grid
        d = torch.linspace(-1, 1, D, device=image.device, dtype=flow.dtype)
        h = torch.linspace(-1, 1, H, device=image.device, dtype=flow.dtype)
        w = torch.linspace(-1, 1, W, device=image.device, dtype=flow.dtype)

        # Create meshgrid
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing="ij")
        # each shape: [D, H, W]

        # Stack to create full grid
        base_grid = torch.stack([grid_d, grid_h, grid_w], dim=-1)
        # shape: [D, H, W, 3]

        # Expand to batch
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
        # shape: [B, D, H, W, 3]

        # Normalize flow to [-1, 1] range
        # Flow is in voxels, need to convert to normalized coordinates
        # Grid spacing in normalized coords is 2/(size-1) ≈ 2/size for large sizes
        flow_d_normalized = 2.0 * flow[:, 0] / max(D - 1, 1)
        flow_h_normalized = 2.0 * flow[:, 1] / max(H - 1, 1)
        flow_w_normalized = 2.0 * flow[:, 2] / max(W - 1, 1)

        # Add displacement to base grid
        # flow specifies where to sample FROM, so we add it to the target coordinates
        sampling_grid = base_grid.clone()
        sampling_grid[:, :, :, :, 0] = sampling_grid[:, :, :, :, 0] + flow_d_normalized
        sampling_grid[:, :, :, :, 1] = sampling_grid[:, :, :, :, 1] + flow_h_normalized
        sampling_grid[:, :, :, :, 2] = sampling_grid[:, :, :, :, 2] + flow_w_normalized
        # shape: [B, D, H, W, 3]

        # grid_sample expects grid of shape [B, H_out, W_out, D_out, 3] for 4D
        # or [B, D_out, H_out, W_out, 3] for 5D - need to permute
        sampling_grid = sampling_grid.permute(0, 4, 1, 2, 3)
        # shape: [B, 3, D, H, W] -> [B, 3, D, H, W] for grid_sample

        # Apply spatial transformer
        # grid_sample with align_corners=False:
        # - The grid values are in [-1, 1] representing the normalized spatial extent
        # - Coordinates are centered at 0, with -1 at one edge and 1 at the opposite
        warped = F.grid_sample(
            image,
            sampling_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=False,
        )
        # shape: [B, C, D, H, W]

        if return_grid:
            return warped, sampling_grid
        return warped

    def inverse_warp(
        self,
        image: Tensor,
        flow: Tensor,
    ) -> Tensor:
        """
        Apply inverse deformation (useful for registration).

        The flow defines the forward displacement, so we negate it
        to get the inverse warp.

        Args:
            image: Input image [B, C, D, H, W]
            flow: Forward deformation field [B, 3, D, H, W]

        Returns:
            Inversely warped image
        """
        inverse_flow = -flow
        return self.forward(image, inverse_flow)


class CompositionTransformer(nn.Module):
    """
    Composes multiple deformations for sequential warping.

    Useful for applying intermediate deformations during ODE integration.
    """

    def __init__(
        self,
        mode: str = "bilinear",
        padding_mode: str = "border",
    ) -> None:
        super().__init__()
        self.stn = SpatialTransformer(mode=mode, padding_mode=padding_mode)

    def compose_flows(
        self,
        flow1: Tensor,
        flow2: Tensor,
    ) -> Tensor:
        """
        Compose two deformation fields.

        flow_composed(x) = flow1(x) + flow2(x + flow1(x))

        Args:
            flow1: First flow [B, 3, D, H, W]
            flow2: Second flow [B, 3, D, H, W]

        Returns:
            Composed flow [B, 3, D, H, W]
        """
        # Apply flow2 at locations displaced by flow1
        # This requires a grid sample operation
        warped_flow2 = self.stn(flow2.permute(0, 1, 4, 3, 2), flow1)
        warped_flow2 = warped_flow2.permute(0, 1, 4, 3, 2)
        return flow1 + warped_flow2

    def forward(
        self,
        image: Tensor,
        flows: list,
    ) -> Tensor:
        """
        Apply sequential deformations.

        Args:
            image: Input image [B, C, D, H, W]
            flows: List of deformation fields to apply in sequence

        Returns:
            Warped image after all deformations
        """
        result = image
        for flow in flows:
            result = self.stn(result, flow)
        return result


def create_identity_flow(
    spatial_size: Tuple[int, int, int],
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Create an identity deformation field (zero displacement).

    Args:
        spatial_size: Tuple of (D, H, W)
        device: Device to create tensor on

    Returns:
        Identity flow of shape [1, 3, D, H, W] (all zeros)
    """
    D, H, W = spatial_size
    flow = torch.zeros(1, 3, D, H, W, device=device)
    return flow


def flow_to_displacement_voxel(
    flow: Tensor,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tensor:
    """
    Convert flow from normalized to voxel displacement.

    Args:
        flow: Flow in normalized coords [-1, 1] representing [-spacing*D/2, spacing*D/2]
        spacing: Voxel spacing in mm

    Returns:
        Displacement in voxels
    """
    # Convert normalized flow to physical displacement then to voxels
    # Normalized coord: [-1, 1] maps to physical extent of size * spacing
    # So normalized 2.0 corresponds to size * spacing total range
    # And 1.0 corresponds to size * spacing / 2

    D, H, W = flow.shape[2], flow.shape[3], flow.shape[4]
    physical_extent = torch.tensor(
        [D * spacing[0], H * spacing[1], W * spacing[2]],
        device=flow.device,
        dtype=flow.dtype,
    )
    # Normalization factor: with align_corners=False, grid goes from -1 to 1
    # covering the full extent, so 2.0 in normalized coords = physical_extent
    norm_factor = physical_extent / 2.0

    displacement = flow * norm_factor.view(1, 3, 1, 1, 1)
    return displacement


def compute_determinant_jacobian(
    flow: Tensor,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tensor:
    """
    Compute Jacobian determinant for a deformation field.

    For deformation φ(x) = x + u(x), J = I + ∂u/∂x

    Args:
        flow: Deformation field [B, 3, D, H, W]
        spacing: Voxel spacing for gradient computation

    Returns:
        Jacobian determinant [B, D-2, H-2, W-2]
    """
    B, _, D, H, W = flow.shape

    # Compute gradients using central differences
    # u = flow (displacement)
    # ∂u_i/∂x_j ≈ (u_i(x+Δx_j) - u_i(x-Δx_j)) / (2 * spacing_j)

    # Gradients for each component of displacement
    # du/dD
    du0_dd = (flow[:, 0, 2:, 1:-1, 1:-1] - flow[:, 0, :-2, 1:-1, 1:-1]) / (2 * spacing[0])
    du0_dh = (flow[:, 0, 1:-1, 2:, 1:-1] - flow[:, 0, 1:-1, :-2, 1:-1]) / (2 * spacing[1])
    du0_dw = (flow[:, 0, 1:-1, 1:-1, 2:] - flow[:, 0, 1:-1, 1:-1, :-2]) / (2 * spacing[2])

    du1_dd = (flow[:, 1, 2:, 1:-1, 1:-1] - flow[:, 1, :-2, 1:-1, 1:-1]) / (2 * spacing[0])
    du1_dh = (flow[:, 1, 1:-1, 2:, 1:-1] - flow[:, 1, 1:-1, :-2, 1:-1]) / (2 * spacing[1])
    du1_dw = (flow[:, 1, 1:-1, 1:-1, 2:] - flow[:, 1, 1:-1, 1:-1, :-2]) / (2 * spacing[2])

    du2_dd = (flow[:, 2, 2:, 1:-1, 1:-1] - flow[:, 2, :-2, 1:-1, 1:-1]) / (2 * spacing[0])
    du2_dh = (flow[:, 2, 1:-1, 2:, 1:-1] - flow[:, 2, 1:-1, :-2, 1:-1]) / (2 * spacing[1])
    du2_dw = (flow[:, 2, 1:-1, 1:-1, 2:] - flow[:, 2, 1:-1, 1:-1, :-2]) / (2 * spacing[2])

    # Jacobian: J = I + ∂u/∂x
    # det(J) = (1 + du0_d0) * ((1 + du1_d1) * (1 + du2_d2) - du1_d2 * du2_d1)
    #          - (du0_d1) * (du1_d0 * (1 + du2_d2) - du1_d2 * du2_d0)
    #          + (du0_d2) * (du1_d0 * du2_d1 - (1 + du1_d1) * du2_d0)

    det = (
        (1 + du0_dd) * ((1 + du1_dh) * (1 + du2_dw) - du1_dw * du2_dh)
        - du0_dh * (du1_dd * (1 + du2_dw) - du1_dw * du2_dd)
        + du0_dw * (du1_dd * du2_dh - (1 + du1_dh) * du2_dd)
    )

    return det
