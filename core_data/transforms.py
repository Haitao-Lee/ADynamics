"""
MONAI-based preprocessing transforms for 3D MRI.

Supports:
- Single-modal: T1 MRI only
- Multi-modal: T1 + optional fMRI, ASL, QSM, FLAIR

Preprocessing pipeline:
- Load NIfTI files
- Ensure channel-first format
- Reorient to RAS orientation
- Crop foreground (remove black borders) - BEFORE resampling to save CPU
- Resample to 1mm isotropic (on cropped brain region only)
- Intensity normalization with outlier removal
- Resize/pad to target spatial size (fixed output dimensions)
"""

from typing import Any, Dict, List, Sequence, Optional

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    ResizeWithPadOrCropd,
    Compose,
)


def get_train_transforms(
    spatial_size: Sequence[int] = (256, 256, 192),
) -> Compose:
    """
    Get training data transforms pipeline using MONAI Dictionary Transforms.

    Applies the following preprocessing steps in order:
        1. LoadImaged: Load NIfTI file from disk (preserves affine for Orientationd)
        2. EnsureChannelFirstd: Ensure channel dimension is first
        3. Orientationd: Reorient image to RAS (Right-Anterior-Superior)
        4. CropForegroundd: Remove zero-intensity borders (crop BEFORE resampling)
        5. Spacingd: Resample to 1x1x1 mm isotropic resolution (on valid brain region)
        6. ScaleIntensityRangePercentilesd: Normalize intensity, clip outliers
        7. ResizeWithPadOrCropd: Force fixed output size (pad small, crop large)

    Args:
        spatial_size: Target spatial dimensions (D, H, W). Default: (256, 256, 192)

    Returns:
        MONAI Compose object with all training transforms
    """
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["image"],
                reader="NibabelReader",
            ),
            EnsureChannelFirstd(
                keys=["image"],
            ),
            Orientationd(
                keys=["image"],
                axcodes="RAS",
            ),
            CropForegroundd(
                keys=["image"],
                source_key="image",
                margin_cut=0,
            ),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
                align_corners=False,
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                relative=False,
            ),
            ResizeWithPadOrCropd(
                keys=["image"],
                spatial_size=spatial_size,
                mode="constant",
            ),
        ]
    )
    return train_transforms


def get_val_transforms(
    spatial_size: Sequence[int] = (256, 256, 192),
) -> Compose:
    """
    Get validation data transforms pipeline using MONAI Dictionary Transforms.

    Validation transforms are identical to training transforms to ensure
    consistency between train and val data processing.

    Applies the following preprocessing steps in order:
        1. LoadImaged: Load NIfTI file from disk (preserves affine for Orientationd)
        2. EnsureChannelFirstd: Ensure channel dimension is first
        3. Orientationd: Reorient image to RAS (Right-Anterior-Superior)
        4. CropForegroundd: Remove zero-intensity borders (crop BEFORE resampling)
        5. Spacingd: Resample to 1x1x1 mm isotropic resolution (on valid brain region)
        6. ScaleIntensityRangePercentilesd: Normalize intensity, clip outliers
        7. ResizeWithPadOrCropd: Force fixed output size (pad small, crop large)

    Args:
        spatial_size: Target spatial dimensions (D, H, W). Default: (256, 256, 192)

    Returns:
        MONAI Compose object with all validation transforms
    """
    val_transforms = Compose(
        [
            LoadImaged(
                keys=["image"],
                reader="NibabelReader",
            ),
            EnsureChannelFirstd(
                keys=["image"],
            ),
            Orientationd(
                keys=["image"],
                axcodes="RAS",
            ),
            CropForegroundd(
                keys=["image"],
                source_key="image",
                margin_cut=0,
            ),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
                align_corners=False,
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                relative=False,
            ),
            ResizeWithPadOrCropd(
                keys=["image"],
                spatial_size=spatial_size,
                mode="constant",
            ),
        ]
    )
    return val_transforms


# Multi-modal transform sizes.
# Sizes were derived from the actual data shape survey (June 2026):
#   T1    actual (197, 233, 189) → (256, 256, 192) — MUST match decoder
#         output (the VAE's internal latent grid is (16,16,12) and
#         decoder_depth=4 → 16x upsample to (256,256,192)). Resize/crop
#         the real T1 to that size so the recon loss has matching shapes.
#   fMRI  actual (64, 64, 34, T)  → (64, 64, 34)    — 3D part; T=200 separately
#   ASL   actual (128, 128, 32)   → (64, 64, 32)    — perfusion, half-res
#   QSM   actual (256, 256, 124)  → (128, 128, 96)  — venous detail, half D/H
#   FLAIR actual (256, 256, 22)   → (128, 128, 32)  — slice-thin, upsample W
# Per-modality resize happens in MultiModalDataset._resize_spatial_3d;
# this dict drives the MONAI transform for T1 only (which is the only
# modality that needs intensity normalization + crop).
MULTI_MODAL_SPATIAL_SIZES = {
    "t1":   (128, 128, 128),
    "fmri": (64, 64, 34),
    "asl":  (64, 64, 32),
    "qsm":  (128, 128, 96),
    "flair": (128, 128, 32),
}


def get_multimodal_train_transforms(
    spatial_sizes: Optional[Dict[str, Sequence[int]]] = None,
) -> Compose:
    """
    Get multi-modal training transforms.

    Only T1 gets full preprocessing pipeline. Optional modalities get simplified
    transforms (load + resize) since they may be missing.

    Args:
        spatial_sizes: Dict mapping modality names to spatial sizes.
                      Default uses MULTI_MODAL_SPATIAL_SIZES.

    Returns:
        MONAI Compose object with multi-modal transforms
    """
    if spatial_sizes is None:
        spatial_sizes = MULTI_MODAL_SPATIAL_SIZES

    # T1 transforms (full pipeline)
    t1_size = spatial_sizes.get("t1", (128, 128, 128))
    t1_transforms = Compose([
        LoadImaged(keys=["t1"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["t1"]),
        Orientationd(keys=["t1"], axcodes="RAS"),
        CropForegroundd(keys=["t1"], source_key="t1", margin_cut=0),
        Spacingd(keys=["t1"], pixdim=(1.0, 1.0, 1.0), mode="bilinear", align_corners=False),
        ScaleIntensityRangePercentilesd(keys=["t1"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, relative=False),
        ResizeWithPadOrCropd(keys=["t1"], spatial_size=t1_size, mode="constant"),
    ])

    return t1_transforms


def get_multimodal_train_transforms_safe(
    spatial_sizes: Optional[Dict[str, Sequence[int]]] = None,
) -> Compose:
    """
    A more robust version of get_multimodal_train_transforms that skips
    Spacingd and CropForegroundd (which fail on some samples where the
    brain mask collapses to 0×0×0). The trade-off: no resampling to
    1mm isotropic, no foreground cropping. Use this when the multi-modality
    data is already preprocessed (manifest paths point to preprocessed NIfTI).

    Only does:
      - LoadImaged
      - EnsureChannelFirstd
      - Orientationd (RAS)
      - ScaleIntensityRangePercentilesd
      - ResizeWithPadOrCropd
    """
    if spatial_sizes is None:
        from core_data.transforms import MULTI_MODAL_SPATIAL_SIZES as _sizes
        spatial_sizes = _sizes

    t1_size = spatial_sizes.get("t1", (128, 128, 128))
    t1_transforms = Compose([
        LoadImaged(keys=["t1"], reader="NibabelReader"),
        EnsureChannelFirstd(keys=["t1"]),
        Orientationd(keys=["t1"], axcodes="RAS"),
        ScaleIntensityRangePercentilesd(keys=["t1"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, relative=False),
        ResizeWithPadOrCropd(keys=["t1"], spatial_size=t1_size, mode="constant"),
    ])

    return t1_transforms


def get_multimodal_val_transforms(
    spatial_sizes: Optional[Dict[str, Sequence[int]]] = None,
) -> Compose:
    """
    Get multi-modal validation transforms.
    Same as training transforms since we don't apply augmentation.

    Args:
        spatial_sizes: Dict mapping modality names to spatial sizes.
                      Default uses MULTI_MODAL_SPATIAL_SIZES.

    Returns:
        MONAI Compose object with multi-modal transforms
    """
    return get_multimodal_train_transforms(spatial_sizes)