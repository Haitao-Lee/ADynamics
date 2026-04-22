"""
ADynamics Training Engine Module

Core training logic for each stage:
- VAE trainer for Stage 1
- CFM trainer for Stage 3
- Loss functions
"""

from engine.losses import (
    cfm_loss,
    deformation_smooth_loss,
    total_vae_loss,
)
from engine.trainer_vae import VAETrainer
from engine.trainer_cfm import CFMTrainer

__all__ = [
    "VAETrainer",
    "CFMTrainer",
    "total_vae_loss",
    "cfm_loss",
    "deformation_smooth_loss",
]
