"""
ADynamics Training Engine.

Loss functions and per-stage trainers:
    losses          -- VAE (recon, KL, contrastive, gradient, SSIM),
                       CFM (velocity, rectified flow), deformation (similarity,
                       smooth, Jacobian), ordinal CE, ordinal regression.
    trainer_vae     -- VAETrainer (single-modal) and MultiModalVAETrainer
                       (4-class + 5-modality with multi-GPU DataParallel).
    trainer_cfm     -- CFMTrainer for the MMSE-conditional flow matching.

Public API (most-used names):
    total_vae_loss, cfm_loss, vae_reconstruction_loss, vae_kl_loss,
    ordinal_cross_entropy_loss, ordinal_contrastive_loss, ordinal_regression_loss,
    gradient_loss, ssim_loss, rectified_flow_regularization, GradientSmoothingLoss,
    MultiModalVAETrainer, CFMTrainer
"""

from engine.losses import (
    total_vae_loss,
    cfm_loss,
    vae_reconstruction_loss,
    vae_kl_loss,
    ordinal_cross_entropy_loss,
    ordinal_contrastive_loss,
    ordinal_regression_loss,
    gradient_loss,
    ssim_loss,
    rectified_flow_regularization,
    GradientSmoothingLoss,
    MemoryBank,
    supervised_contrastive_loss,
)
from engine.trainer_vae import VAETrainer, MultiModalVAETrainer
from engine.trainer_cfm import CFMTrainer

__all__ = [
    # losses
    "total_vae_loss",
    "cfm_loss",
    "vae_reconstruction_loss",
    "vae_kl_loss",
    "ordinal_cross_entropy_loss",
    "ordinal_contrastive_loss",
    "ordinal_regression_loss",
    "gradient_loss",
    "ssim_loss",
    "rectified_flow_regularization",
    "GradientSmoothingLoss",
    "MemoryBank",
    "supervised_contrastive_loss",
    # trainers
    "VAETrainer",
    "MultiModalVAETrainer",
    "CFMTrainer",
]
