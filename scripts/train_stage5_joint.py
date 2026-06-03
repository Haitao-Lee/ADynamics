"""
Stage 5: Joint Fine-tuning for Multi-Modal VAE.

Fine-tunes all pretrained modules together end-to-end:
    - Multi-Modal VAE Encoder + Decoder (from Stage 1/2)
    - CFM Velocity Field (from Stage 3)
    - Deformation Generator (from Stage 4)

Combined loss:
    L_total = lambda_recon * L_recon + lambda_cfm * L_cfm + lambda_def * L_def

Where:
    L_recon: L1 reconstruction loss
    L_cfm: CFM velocity field loss
    L_def: Deformation smoothness + similarity loss

All modules are unfrozen but use lower learning rates for pretrained weights.

Usage:
    python scripts/train_stage5_joint.py \
        --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \
        --decoder_checkpoint ./checkpoints/stage2_decoder/decoder_best.pt \
        --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt \
        --deform_checkpoint ./checkpoints/stage4_def/def_best.pt \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --output_dir ./checkpoints/stage5_joint
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_train_transforms, get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D
from models.vector_field import VelocityFieldNet
from models.spatial_transform import (
    DeformationGenerator,
    SpatialTransformer,
    compute_jacobian_penalty,
)
from engine.losses import GradientSmoothingLoss


def parse_args():
    from utils.config_loader import apply_yaml_defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    pre_args, _ = pre.parse_known_args()

    mapping = [
        (("input", "encoder_checkpoint"), "encoder_checkpoint"),
        (("input", "cfm_checkpoint"), "cfm_checkpoint"),
        (("input", "deform_checkpoint"), "deform_checkpoint"),
        (("input", "num_classes"), "num_classes"),
        (("model", "latent_channels"), "latent_channels"),
        (("model", "base_channels"), "base_channels"),
        (("model", "decoder_depth"), "decoder_depth"),
        (("model", "in_channels"), "in_channels"),
        (("model", "spatial_size"), "spatial_size"),
        (("model", "dropout_rate"), "dropout_rate"),
        (("training", "batch_size"), "batch_size"),
        (("training", "learning_rate"), "learning_rate"),
        (("training", "weight_decay"), "weight_decay"),
        (("training", "epochs"), "epochs"),
        (("training", "early_stopping_patience"), "early_stopping"),
        (("training", "num_gpus"), "num_gpus"),
        (("training", "use_amp"), "no_amp"),
        (("loss", "recon_weight"), "recon_weight"),
        (("loss", "cfm_weight"), "cfm_weight"),
        (("loss", "def_weight"), "def_weight"),
        (("loss", "smooth_weight"), "smooth_weight"),
        (("loss", "jacobian_weight"), "jacobian_weight"),
        (("output", "dir"), "output_dir"),
        (("seed",), "seed"),
    ]
    config_defaults = apply_yaml_defaults(pre_args.config, mapping) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Stage 5 Joint Fine-tuning", parents=[pre])
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage5_joint")
    parser.add_argument("--encoder_checkpoint", type=str,
                        default="./checkpoints/stage1_multimodal/vae_best.pt")
    parser.add_argument("--decoder_checkpoint", type=str, default=None)
    parser.add_argument("--cfm_checkpoint", type=str,
                        default="./checkpoints/stage3_cfm/cfm_best.pt")
    parser.add_argument("--deform_checkpoint", type=str,
                        default="./checkpoints/stage4_def/def_best.pt")

    # Model params (must match Stage 1)
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD)")
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--cfm_weight", type=float, default=0.1)
    parser.add_argument("--def_weight", type=float, default=0.1)
    parser.add_argument("--smooth_weight", type=float, default=0.05)
    parser.add_argument("--jacobian_weight", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--early_stopping", type=int, default=30)
    parser.add_argument("--no_amp", action="store_true", default=False)
    # Apply YAML config defaults AFTER all add_argument calls
    # (set_defaults must come last so it isn't overridden by argparse defaults)
    parser.set_defaults(**config_defaults)

    return parser.parse_args()
