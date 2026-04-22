"""
Dataset and DataLoader utilities for ADynamics.

Provides functions to create train/validation dataloaders from
a list of data dictionaries containing paths and labels.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset
from torch.utils.data import Subset


def get_dataloader(
    data_list: List[Dict[str, Any]],
    train_transforms,
    val_transforms,
    batch_size: int = 2,
    num_workers: int = 4,
    val_split: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders from a list of data dictionaries.

    The data_list should contain dictionaries with the following structure:
        {"image": "/path/to/image.nii.gz", "label": stage_int}

    The data is split into train/val sets based on val_split ratio.

    Args:
        data_list: List of dictionaries, each containing:
            - "image": str, path to the NIfTI image file
            - "label": int, disease stage label (0=NC, 1=SCD, 2=MCI, 3=AD)
        train_transforms: MONAI transforms for training data
        val_transforms: MONAI transforms for validation data
        batch_size: Number of samples per batch. Default: 2
        num_workers: Number of worker processes for data loading. Default: 4
        val_split: Fraction of data to use for validation. Default: 0.2
        shuffle: Whether to shuffle training data. Default: True
        seed: Random seed for reproducible train/val split. Default: 42

    Returns:
        Tuple of (train_loader, val_loader)

    Raises:
        ValueError: If data_list is empty or val_split is invalid
    """
    if len(data_list) == 0:
        raise ValueError("data_list cannot be empty")
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")

    # Set seed for reproducible split
    np.random.seed(seed)
    indices = np.random.permutation(len(data_list)).tolist()

    # Calculate split index
    n_val = int(len(data_list) * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Create subsets
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]

    # Create MONAI datasets
    train_dataset = CacheDataset(
        data=train_data,
        transform=train_transforms,
        cache_num=len(train_data),
        num_workers=num_workers,
    )

    val_dataset = CacheDataset(
        data=val_data,
        transform=val_transforms,
        cache_num=len(val_data),
        num_workers=num_workers,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def create_dummy_dataset(
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    num_samples: int = 10,
) -> List[Dict[str, Any]]:
    """
    Create a dummy dataset for testing pipeline functionality.

    Generates random 3D numpy arrays and saves them as NIfTI files
    in a temporary directory. Useful for pipeline testing without
    real MRI data.

    Args:
        spatial_size: Spatial dimensions (D, H, W) of dummy images
        num_samples: Number of dummy samples to create

    Returns:
        List of data dictionaries with paths to dummy NIfTI files
    """
    import tempfile

    import nibabel as nib

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="adynamics_dummy_")
    data_list = []

    for i in range(num_samples):
        # Generate random 3D MRI-like data (brain-shaped ellipsoid with noise)
        D, H, W = spatial_size
        dummy_data = np.random.rand(D, H, W).astype(np.float32)

        # Create ellipsoid brain mask
        x = np.linspace(-1, 1, D)
        y = np.linspace(-1, 1, H)
        z = np.linspace(-1, 1, W)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        ellipsoid = (xx**2 + yy**2 + zz**2) <= 0.8

        # Apply brain mask and add some structure
        dummy_data = dummy_data * ellipsoid + 0.3 * ellipsoid
        dummy_data = (dummy_data / dummy_data.max()).astype(np.float32)

        # Create NIfTI image with identity affine
        affine = np.eye(4)
        nii_image = nib.Nifti1Image(dummy_data, affine)

        # Save to file
        filename = os.path.join(temp_dir, f"dummy_T1_{i:04d}.nii.gz")
        nib.save(nii_image, filename)

        # Random disease stage label
        label = np.random.randint(0, 4)

        data_list.append({
            "image": filename,
            "label": int(label),
        })

    return data_list
