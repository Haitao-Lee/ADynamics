"""
Baseline Comparison for ADynamics CFM Pipeline.

Compares the learned CFM flow against simpler interpolation baselines to validate
that flow matching provides genuine value over naive approaches.

Baselines:
    1. Linear Interpolation: z_interp = (1-t)*z_NC + t*z_AD (no learned model)
    2. KNN Interpolation: Average of K nearest neighbors in each class
    3. Supervised Regression: Direct NC->AD mapping via trained MLP
    4. CFM (Ours): Learned velocity field via Conditional Flow Matching

Evaluation Metrics:
    - Trajectory straightness (curvature)
    - Latent space alignment (cosine similarity to true class centroids)
    - ODE integration efficiency (steps needed for convergence)
    - Classification consistency (do intermediate states classify correctly?)

Usage:
    python scripts/run_baseline_comparison.py \
        --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \
        --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt \
        --output_dir ./inference_results/baseline_comparison
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D
from models.vector_field import VelocityFieldNet


def parse_args():
    from utils.config_loader import apply_yaml_defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    pre_args, _ = pre.parse_known_args()

    mapping = [
        (("input", "num_classes"), "num_classes"),
        (("model", "latent_channels"), "latent_channels"),
        (("model", "base_channels"), "base_channels"),
        (("model", "decoder_depth"), "decoder_depth"),
        (("baseline_comparison", "output_dir"), "output_dir"),
        (("baseline_comparison", "num_test"), "num_test"),
    ]
    config_defaults = apply_yaml_defaults(pre_args.config, mapping) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Baseline Comparison", parents=[pre])
    # Apply YAML config defaults AFTER all add_argument calls
    # (set_defaults must come last so it isn't overridden by argparse defaults)
    parser.set_defaults(**config_defaults)
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--encoder_checkpoint", type=str, required=True)
    parser.add_argument("--cfm_checkpoint", type=str, default=None)
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD)")
    parser.add_argument("--output_dir", type=str, default="./inference_results/baseline_comparison")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()
