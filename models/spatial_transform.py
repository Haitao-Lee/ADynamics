"""
Spatial Transformation Module for ADynamics.

Implements:
    - DeformationGenerator: 3D CNN that generates displacement fields from latent
    - SpatialTransformer: 3D STN that applies deformation to MRI images

Critical for medical imaging:
    - Uses align_corners=False for proper physical coordinate handling
    - Jacobian determinant penalty to prevent folding

PyTorch grid_sample coordinate convention for 3D:
    - Input tensor shape: [B, C, D, H, W] where D=depth, H=height, W=width
    - Grid shape: [B, D, H, W, 3] where last dim is (x, y, z)
    - x corresponds to W (width), y corresponds to H (height), z corresponds to D (depth)
    - This means when creating meshgrid with indexing="ij", we stack [grid_w, grid_h, grid_d]
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

    For HD inputs [256, 256, 192], base_channels is reduced to 16 to prevent OOM.

    Architecture:
        Latent -> 3D U-Net style upsampling -> 3-channel displacement field

    Output:
        displacement field of shape [B, 3, D, H, W] where:
            channel 0 = displacement in x direction (W dimension)
            channel 1 = displacement in y direction (H dimension)
            channel 2 = displacement in z direction (D dimension)

    The displacement is in voxel units, can be multiplied by spacing for physical units.

    Initialization: Output conv initialized to near-zero for identity-like behavior
    at early training stages, which stabilizes convergence.
    """

    def __init__(
        self,
        latent_channels: int = 64,
        latent_spatial: Tuple[int, int, int] = (16, 16, 12),
        output_spatial: Tuple[int, int, int] = (256, 256, 192),
        base_channels: int = 16,
        channel_mults: Tuple[int, int, int, int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        max_displacement: float = 10.0,
    ) -> None:
        """
        Initialize Deformation Generator.

        Args:
            latent_channels: Number of channels in latent representation
            latent_spatial: Spatial dimensions of latent (D, H, W)
            output_spatial: Output deformation field spatial dimensions for HD: (256, 256, 192)
            base_channels: Base channel count. Default 16 for HD memory efficiency.
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

            for _ in range(num_res_blocks):
                self.decoder_blocks.append(
                    nn.Sequential(
                        nn.GroupNorm(8, ch),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv3d(ch, out_ch, 3, padding=1),
                    )
                )
                ch = out_ch

            if i < len(channel_mults):
                self.decoder_upsample.append(
                    nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1)
                )

        # Output head to generate 3-channel displacement
        # Initialized to near-zero for identity-like behavior at start
        self.output_head = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(ch, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(base_channels, 3, 3, padding=1),
        )

        # Initialize output conv to near-zero for stable initial deformation
        self._init_output_layer()

    def _init_output_layer(self) -> None:
        """Initialize output conv to near-zero for identity-like initial behavior."""
        for m in self.output_head.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_final: Tensor) -> Tensor:
        """
        Generate deformation field from evolved latent.

        Args:
            z_final: Evolved latent from CFM of shape [B, latent_channels, 16, 16, 12]

        Returns:
            Deformation field of shape [B, 3, 256, 256, 192]
            Values are clamped to [-max_displacement, max_displacement]
        """
        h = self.latent_proj(z_final)

        block_idx = 0
        for i in range(len(self.decoder_upsample)):
            h = self.decoder_upsample[i](h)

            for _ in range(2):
                h = h + self.decoder_blocks[block_idx](h)
                block_idx += 1

        for _ in range(2):
            h = h + self.decoder_blocks[block_idx](h)
            block_idx += 1

        flow = self.output_head(h)
        flow = torch.clamp(flow, min=-self.max_displacement, max=self.max_displacement)

        return flow


class SpatialTransformer(nn.Module):
    """
    3D Spatial Transformer Network (STN) for applying deformations.

    Uses torch.nn.functional.grid_sample to apply a displacement field
    to a 3D image. The displacement field specifies where to sample from
    in the input image to produce the output.

    CRITICAL: Uses align_corners=False for proper physical coordinate handling.

    Coordinate convention for PyTorch grid_sample:
        - Input: [B, C, D, H, W] where D=depth, H=height, W=width
        - Grid: [B, D, H, W, 3] where last dim is (x, y, z)
        - x corresponds to W dimension, y corresponds to H, z corresponds to D
        - Therefore, when creating meshgrid with indexing="ij",
          we stack [grid_w, grid_h, grid_d] to get correct (x, y, z) order
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
            image: Input image of shape [B, C, D, H, W]
            flow: Deformation field of shape [B, 3, D, H, W]
                flow[:, 0] = displacement in x (W dimension)
                flow[:, 1] = displacement in y (H dimension)
                flow[:, 2] = displacement in z (D dimension)
                Values are in voxels, will be converted to normalized coords
            return_grid: Whether to return the sampling grid

        Returns:
            Warped image of shape [B, C, D, H, W]
        """
        B, C, D, H, W = image.shape

        # Create normalized base grid [-1, 1]
        # Use linspace to create coordinate arrays
        d = torch.linspace(-1, 1, D, device=image.device, dtype=flow.dtype)
        h = torch.linspace(-1, 1, H, device=image.device, dtype=flow.dtype)
        w = torch.linspace(-1, 1, W, device=image.device, dtype=flow.dtype)

        # Create meshgrid with indexing="ij" for (D, H, W) order
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing="ij")
        # each shape: [D, H, W]

        # Stack to [x, y, z] order = [W, H, D] for grid_sample convention
        # PyTorch grid_sample expects last dim to be (x, y, z) where x=W, y=H, z=D
        base_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)
        # shape: [D, H, W, 3] where [:,:,:,0]=x=W, [:,:,:,1]=y=H, [:,:,:,2]=z=D

        # Expand to batch: [B, D, H, W, 3]
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)

        # Normalize flow from voxel units to [-1, 1] range
        # flow[:, 0] is x-displacement (W), flow[:, 1] is y-displacement (H), flow[:, 2] is z-displacement (D)
        flow_x_norm = 2.0 * flow[:, 0] / max(W - 1, 1)
        flow_y_norm = 2.0 * flow[:, 1] / max(H - 1, 1)
        flow_z_norm = 2.0 * flow[:, 2] / max(D - 1, 1)

        # Add displacement: base_grid[..., 0] is x (W), base_grid[..., 1] is y (H), base_grid[..., 2] is z (D)
        sampling_grid = base_grid.clone()
        sampling_grid[..., 0] = sampling_grid[..., 0] + flow_x_norm
        sampling_grid[..., 1] = sampling_grid[..., 1] + flow_y_norm
        sampling_grid[..., 2] = sampling_grid[..., 2] + flow_z_norm
        # shape: [B, D, H, W, 3] - ready for grid_sample (NO permute needed!)

        # Apply spatial transformer with align_corners=False for physical coords
        warped = F.grid_sample(
            image,
            sampling_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=False,
        )

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

        With properly aligned underlying STN, no manual axis swap needed.

        Args:
            flow1: First flow [B, 3, D, H, W]
            flow2: Second flow [B, 3, D, H, W]

        Returns:
            Composed flow [B, 3, D, H, W]
        """
        # Apply flow2 at locations displaced by flow1
        warped_flow2 = self.stn(flow2, flow1)
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
    D, H, W = flow.shape[2], flow.shape[3], flow.shape[4]
    physical_extent = torch.tensor(
        [D * spacing[0], H * spacing[1], W * spacing[2]],
        device=flow.device,
        dtype=flow.dtype,
    )
    norm_factor = physical_extent / 2.0

    displacement = flow * norm_factor.view(1, 3, 1, 1, 1)
    return displacement


def compute_determinant_jacobian(
    flow: Tensor,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tensor:
    """
    Compute Jacobian determinant for a deformation field using torch.gradient.

    For deformation φ(x) = x + u(x), J = I + ∂u/∂x

    Uses torch.gradient for more accurate gradient computation that handles
    boundary conditions better than manual slicing.

    Args:
        flow: Deformation field [B, 3, D, H, W]
        spacing: Voxel spacing for gradient computation

    Returns:
        Jacobian determinant [B, D, H, W] (same size as input, edge handling at boundaries)
    """
    B, _, D, H, W = flow.shape

    # Use torch.gradient for accurate gradient computation
    # gradient returns [B, 3, spatial_dims] for each spatial dimension

    # Compute gradients along each dimension using torch.gradient
    # Flow channel 0 = x displacement (W), channel 1 = y displacement (H), channel 2 = z displacement (D)

    # Gradient w.r.t. x (W dimension) - note: x coordinate index in PyTorch is the last spatial index
    dx0, dx1, dx2 = torch.gradient(flow[:, 0], dim=(2, 3, 4), spacing=(spacing[0], spacing[1], spacing[2]))
    # dx0 = du_x/dx (wrt D), dx1 = du_x/dy (wrt H), dx2 = du_x/dz (wrt W)

    dy0, dy1, dy2 = torch.gradient(flow[:, 1], dim=(2, 3, 4), spacing=(spacing[0], spacing[1], spacing[2]))
    # dy0 = du_y/dx (wrt D), dy1 = du_y/dy (wrt H), dy2 = du_y/dz (wrt W)

    dz0, dz1, dz2 = torch.gradient(flow[:, 2], dim=(2, 3, 4), spacing=(spacing[0], spacing[1], spacing[2]))
    # dz0 = du_z/dx (wrt D), dz1 = du_z/dy (wrt H), dz2 = du_z/dz (wrt W)

    # Jacobian matrix J for each spatial location:
    # J[i,j] = ∂φ_i / ∂x_j = δ_ij + ∂u_i / ∂x_j
    #
    # For 3D: J = [[1+du_x/dx, du_x/dy, du_x/dz],
    #              [du_y/dx, 1+du_y/dy, du_y/dz],
    #              [du_z/dx, du_z/dy, 1+du_z/dz]]
    #
    # Where x=W, y=H, z=D in tensor indexing

    # Compute determinant: det(J) = (1+dx0)*((1+dy1)*(1+dz2) - dy2*dz1)
    #                          - dx1*(dy0*(1+dz2) - dy2*dz0)
    #                          + dx2*(dy0*dz1 - (1+dy1)*dz0)

    det = (
        (1 + dx0) * ((1 + dy1) * (1 + dz2) - dy2 * dz1)
        - dx1 * (dy0 * (1 + dz2) - dy2 * dz0)
        + dx2 * (dy0 * dz1 - (1 + dy1) * dz0)
    )

    return det


def compute_jacobian_penalty(
    flow: Tensor,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tensor:
    """
    Compute Jacobian determinant penalty to prevent folding.

    Penalizes negative determinants which indicate folding/invalid deformations.

    Args:
        flow: Deformation field [B, 3, D, H, W]
        spacing: Voxel spacing for gradient computation

    Returns:
        Scalar penalty loss
    """
    det = compute_determinant_jacobian(flow, spacing)

    # Penalize negative determinants (folding regions)
    # L_jac = mean(max(0, -det(J))^2)
    folding_penalty = torch.clamp(-det, min=0)
    loss = torch.mean(folding_penalty ** 2)

    return loss