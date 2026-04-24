"""
MONAI-based preprocessing transforms for 3D T1 MRI.

Applies standardized preprocessing pipeline:
- Load NIfTI files
- Ensure channel-first format
- Reorient to RAS orientation
- Crop foreground (remove black borders) - BEFORE resampling to save CPU
- Resample to 1mm isotropic (on cropped brain region only)
- Intensity normalization with outlier removal
- Resize/pad to target spatial size (fixed output dimensions)
"""

from typing import Any, Dict, Sequence

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