"""
ADynamics Utilities Module.

Cross-cutting helpers used across configs, models, training, and inference:
    io_utils          -- NIfTI load/save, affine handling, resampling
    config_loader     -- YAML config -> argparse defaults bridge
    multi_gpu         -- MultiModalDataParallel for dict inputs (DataParallel-safe)
    preprocessing/    -- N4, denoise, registration, tissue segmentation (FSL-FAST,
                         HD-BET, ANTsPy)

Public API:
    load_nifti, save_tensor_to_nifti, tensor_to_nifti_data, create_identity_affine,
    resample_nifti
    apply_yaml_defaults, load_yaml_config, merge_config, remap_labels_3class
    setup_data_parallel, MultiModalDataParallel
"""

from utils.io_utils import (
    load_nifti,
    save_tensor_to_nifti,
    tensor_to_nifti_data,
    create_identity_affine,
    resample_nifti,
)
from utils.config_loader import (
    load_yaml_config,
    apply_yaml_defaults,
    merge_config,
    remap_labels_3class,
)
from utils.multi_gpu import (
    setup_data_parallel,
    MultiModalDataParallel,
)

__all__ = [
    # io_utils
    "load_nifti",
    "save_tensor_to_nifti",
    "tensor_to_nifti_data",
    "create_identity_affine",
    "resample_nifti",
    # config_loader
    "load_yaml_config",
    "apply_yaml_defaults",
    "merge_config",
    "remap_labels_3class",
    # multi_gpu
    "setup_data_parallel",
    "MultiModalDataParallel",
]
