"""
ADynamics Training Engine Module

Core training logic for each stage:
- VAE trainer for Stage 1
- CFM trainer for Stage 3
- Loss functions
"""

from engine.losses import (
    cfm_loss,
    total_vae_loss,
)
from engine.trainer_vae import MultiModalVAETrainer
from engine.trainer_cfm import CFMTrainer

__all__ = [
    "MultiModalVAETrainer",
    "CFMTrainer",
    "total_vae_loss",
    "cfm_loss",
]
