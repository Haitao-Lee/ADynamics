"""
ADynamics Models Module

Neural network architectures for AD progression modeling:
- 3D VAE for latent representation learning
- Disease stage classifier
- Conditional Flow Matching vector field
- Spatial transformer for deformation
"""

from models.vae3d import ADynamicsVAE3D
from models.classifier import DiseaseClassifier, classifier_ce_loss, classifier_accuracy
from models.vector_field import VelocityFieldNet, cfm_velocity_loss
from models.spatial_transform import (
    DeformationGenerator,
    SpatialTransformer,
    CompositionTransformer,
    create_identity_flow,
    flow_to_displacement_voxel,
    compute_determinant_jacobian,
)

__all__ = [
    "ADynamicsVAE3D",
    "DiseaseClassifier",
    "classifier_ce_loss",
    "classifier_accuracy",
    "VelocityFieldNet",
    "cfm_velocity_loss",
    "DeformationGenerator",
    "SpatialTransformer",
    "CompositionTransformer",
    "create_identity_flow",
    "flow_to_displacement_voxel",
    "compute_determinant_jacobian",
]
