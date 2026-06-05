"""
ADynamics Models Module

Neural network architectures for AD progression modeling:
- 3D VAE for latent representation learning
- Conditional Flow Matching vector field
- Spatial transformer for deformation
"""

from models.vae3d import ADynamicsVAE3D, MultiModalVAE3D
from models.vector_field import VelocityFieldNet, cfm_velocity_loss
from models.spatial_transform import (
    DeformationGenerator,
    SpatialTransformer,
    CompositionTransformer,
    create_identity_flow,
    flow_to_displacement_voxel,
    compute_determinant_jacobian,
)
from models.attention_3d import AxialAttention3D, MultiAxisAttention3D

__all__ = [
    "ADynamicsVAE3D",
    "MultiModalVAE3D",
    "VelocityFieldNet",
    "cfm_velocity_loss",
    "DeformationGenerator",
    "SpatialTransformer",
    "CompositionTransformer",
    "create_identity_flow",
    "flow_to_displacement_voxel",
    "compute_determinant_jacobian",
    "AxialAttention3D",
    "MultiAxisAttention3D",
]
