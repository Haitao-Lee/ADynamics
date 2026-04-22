"""
MONAI-based preprocessing transforms for 3D T1 MRI.

Applies standardized preprocessing pipeline:
- Load NIfTI files
- Ensure channel-first format
- Reorient to RAS orientation
- Resample to 1mm isotropic
- Intensity normalization with outlier removal
- Crop foreground (remove black borders)
- Pad or resize to target spatial size
"""

from typing import Any, Dict, Sequence

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    SpatialPadd,
    Resized,
    Compose,
)


def get_train_transforms(
    spatial_size: Sequence[int] = (128, 128, 128),
) -> Compose:
    """
    Get training data transforms pipeline using MONAI Dictionary Transforms.

    Applies the following preprocessing steps in order:
        1. LoadImaged: Load NIfTI file from disk
        2. EnsureChannelFirstd: Ensure channel dimension is first
        3. Orientationd: Reorient image to RAS (Right-Anterior-Superior)
        4. Spacingd: Resample to 1x1x1 mm isotropic resolution
        5. ScaleIntensityRangePercentilesd: Normalize intensity, clip outliers
        6. CropForegroundd: Remove zero-intensity borders
        7. SpatialPadd: Pad to minimum size or resize to target

    Args:
        spatial_size: Target spatial dimensions (D, H, W). Default: (128, 128, 128)

    Returns:
        MONAI Compose object with all training transforms
    """
    train_transforms = Compose(
        [
            LoadImaged(
                keys=["image"],
                reader="NibabelReader",
                image_only=True,
            ),
            EnsureChannelFirstd(
                keys=["image"],
            ),
            Orientationd(
                keys=["image"],
                axcodes="RAS",
            ),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
                align_corners=True,
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                relative=False,
            ),
            CropForegroundd(
                keys=["image"],
                source_key="image",
                margin_cut=0,
            ),
            SpatialPadd(
                keys=["image"],
                spatial_size=spatial_size,
                mode="constant",
                constant_values=0,
            ),
        ]
    )
    return train_transforms


def get_val_transforms(
    spatial_size: Sequence[int] = (128, 128, 128),
) -> Compose:
    """
    Get validation data transforms pipeline using MONAI Dictionary Transforms.

    Validation transforms are identical to training transforms to ensure
    consistency between train and val data processing.

    Args:
        spatial_size: Target spatial dimensions (D, H, W). Default: (128, 128, 128)

    Returns:
        MONAI Compose object with all validation transforms
    """
    val_transforms = Compose(
        [
            LoadImaged(
                keys=["image"],
                reader="NibabelReader",
                image_only=True,
            ),
            EnsureChannelFirstd(
                keys=["image"],
            ),
            Orientationd(
                keys=["image"],
                axcodes="RAS",
            ),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
                align_corners=True,
            ),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                relative=False,
            ),
            CropForegroundd(
                keys=["image"],
                source_key="image",
                margin_cut=0,
            ),
            SpatialPadd(
                keys=["image"],
                spatial_size=spatial_size,
                mode="constant",
                constant_values=0,
            ),
        ]
    )
    return val_transforms
