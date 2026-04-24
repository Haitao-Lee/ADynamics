"""
Dataset and DataLoader utilities for ADynamics.

Provides functions to create train/validation/test dataloaders from
a list of data dictionaries containing paths and labels.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset


def _load_split_from_json(split_save_dir: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Load dataset splits from JSON checkpoint if exists.

    Args:
        split_save_dir: Directory containing dataset_splits.json

    Returns:
        Dictionary with "train", "val", "test" keys if file exists, None otherwise
    """
    json_path = os.path.join(split_save_dir, "dataset_splits.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _save_split_to_json(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    split_save_dir: str,
) -> None:
    """
    Save dataset splits to JSON checkpoint.

    Args:
        train_data: Training subset
        val_data: Validation subset
        test_data: Test subset
        split_save_dir: Directory to save dataset_splits.json
    """
    os.makedirs(split_save_dir, exist_ok=True)
    split_dict = {
        "train": [{"image": item["image"], "label": item["label"]} for item in train_data],
        "val": [{"image": item["image"], "label": item["label"]} for item in val_data],
        "test": [{"image": item["image"], "label": item["label"]} for item in test_data],
    }
    json_path = os.path.join(split_save_dir, "dataset_splits.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2, ensure_ascii=False)


def get_train_val_test_dataloaders(
    data_list: List[Dict[str, Any]],
    train_transforms,
    val_transforms,
    test_transforms,
    batch_size: int = 8,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
    use_cache: bool = False,
    cache_rate: float = 0.1,
    split_save_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test DataLoaders with stratified splitting.

    Uses stratified sampling to preserve label distribution across splits,
    preventing class imbalance in Train/Val/Test sets.

    Memory-safe: Uses standard Dataset by default to avoid OOM with large datasets.
    Optionally enable caching with a safe cache_rate.

    Deterministic: Enforces alphabetical sorting on image paths before splitting
    to ensure reproducibility regardless of filesystem ordering.

    Split persistence: When split_save_dir is provided, splits are saved to JSON
    and reused on subsequent calls, ensuring identical train/val/test divisions
    even after code changes or dataset modifications.

    Args:
        data_list: List of dictionaries, each containing:
            - "image": str, path to the NIfTI image file
            - "label": int, disease stage label (0=NC, 1=SCD, 2=MCI, 3=AD)
        train_transforms: MONAI transforms for training data
        val_transforms: MONAI transforms for validation data
        test_transforms: MONAI transforms for test data
        batch_size: Number of samples per batch. Default: 8
        num_workers: Number of worker processes for data loading. Default: 4
        train_split: Fraction of data for training. Default: 0.7 (70%)
        val_split: Fraction of data for validation. Default: 0.15 (15%)
        shuffle: Whether to shuffle training data. Default: True
        seed: Random seed for reproducible split. Default: 42
        use_cache: If True, use CacheDataset with cache_rate. Default: False (memory-safe)
        cache_rate: Fraction of data to cache (0.0 to 1.0). Only used if use_cache=True.
            Default: 0.1 (caches 10% of training data)
        split_save_dir: Optional directory to save/load dataset splits as JSON.
            If directory contains dataset_splits.json, loads from it instead of splitting.
            If not exists, creates the file after splitting. Default: None (no persistence)

    Returns:
        Tuple of (train_loader, val_loader, test_loader) where test_loader may be None
        if test_split is 0.0.

    Raises:
        ValueError: If data_list is empty or splits don't sum to 1.0

    Example:
        >>> train_loader, val_loader, test_loader = get_train_val_test_dataloaders(
        ...     data_list=data_list,
        ...     train_transforms=train_transforms,
        ...     val_transforms=val_transforms,
        ...     test_transforms=test_transforms,
        ...     batch_size=4,
        ...     train_split=0.7,
        ...     val_split=0.15,
        ...     split_save_dir="./checkpoints",
        ... )
    """
    from sklearn.model_selection import train_test_split

    if len(data_list) == 0:
        raise ValueError("data_list cannot be empty")

    # Step 1: Enforce deterministic alphabetical sorting on image paths
    # This ensures reproducibility regardless of filesystem ordering (e.g., os.listdir)
    data_list = sorted(data_list, key=lambda x: str(x["image"]))

    test_split = round(1.0 - train_split - val_split, 6)
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError(
            f"Splits must sum to 1.0, got train={train_split}, val={val_split}, test={test_split}"
        )

    # Step 2: Try to load existing split from JSON checkpoint
    if split_save_dir is not None:
        cached_splits = _load_split_from_json(split_save_dir)
        if cached_splits is not None:
            train_data = cached_splits.get("train", [])
            val_data = cached_splits.get("val", [])
            test_data = cached_splits.get("test", [])
        else:
            # Compute splits and save to JSON
            train_data, val_data, test_data = _compute_stratified_splits(
                data_list, train_split, val_split, test_split, seed
            )
            _save_split_to_json(train_data, val_data, test_data, split_save_dir)
    else:
        # No persistence requested, compute splits directly
        train_data, val_data, test_data = _compute_stratified_splits(
            data_list, train_split, val_split, test_split, seed
        )

    # Build DataLoaders from split data
    train_dataset, val_dataset, test_dataset = _build_datasets(
        train_data, val_data, test_data,
        train_transforms, val_transforms, test_transforms,
        use_cache, cache_rate, num_workers,
    )

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
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if test_dataset is not None
        else None
    )

    return train_loader, val_loader, test_loader


def _compute_stratified_splits(
    data_list: List[Dict[str, Any]],
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute stratified train/val/test splits using sklearn.

    Args:
        data_list: Sorted list of data dictionaries
        train_split: Training fraction
        val_split: Validation fraction
        test_split: Test fraction
        seed: Random seed

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    from sklearn.model_selection import train_test_split

    labels = np.array([item["label"] for item in data_list])

    # Handle test_split == 0.0 to avoid sklearn ValueError
    if test_split == 0.0:
        train_val_data: List[Dict[str, Any]] = data_list
        test_data: List[Dict[str, Any]] = []
    else:
        train_val_data, test_data = train_test_split(
            data_list,
            test_size=test_split,
            stratify=labels,
            random_state=seed,
        )

    # Recalculate labels for remaining data
    train_val_labels = np.array([item["label"] for item in train_val_data])

    # Second split: separate train and val from remaining
    val_ratio_in_remaining = round(val_split / (train_split + val_split), 6)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_ratio_in_remaining,
        stratify=train_val_labels,
        random_state=seed,
    )

    return train_data, val_data, test_data


def _build_datasets(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    train_transforms,
    val_transforms,
    test_transforms,
    use_cache: bool,
    cache_rate: float,
    num_workers: int,
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    """
    Build MONAI datasets from split data lists.

    Args:
        train_data: Training subset
        val_data: Validation subset
        test_data: Test subset
        train_transforms: Transform for training
        val_transforms: Transform for validation
        test_transforms: Transform for test
        use_cache: Whether to use CacheDataset
        cache_rate: Cache rate for CacheDataset
        num_workers: Number of workers

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset or None)
    """
    if use_cache:
        train_dataset = CacheDataset(
            data=train_data,
            transform=train_transforms,
            cache_num=max(1, int(len(train_data) * cache_rate)),
            num_workers=num_workers,
        )
        val_dataset = CacheDataset(
            data=val_data,
            transform=val_transforms,
            cache_num=max(1, int(len(val_data) * cache_rate)),
            num_workers=num_workers,
        )
        test_dataset = (
            CacheDataset(
                data=test_data,
                transform=test_transforms,
                cache_num=max(1, int(len(test_data) * cache_rate)),
                num_workers=num_workers,
            )
            if test_data
            else None
        )
    else:
        train_dataset = Dataset(data=train_data, transform=train_transforms)
        val_dataset = Dataset(data=val_data, transform=val_transforms)
        test_dataset = Dataset(data=test_data, transform=test_transforms) if test_data else None

    return train_dataset, val_dataset, test_dataset


def cleanup_dummy_dataset(data_list: List[Dict[str, Any]]) -> None:
    """
    Clean up dummy NIfTI files created by create_dummy_dataset.

    Deletes the temporary NIfTI files and their parent directory.
    Should be called after testing to prevent disk space leakage.

    Args:
        data_list: List of data dictionaries with "image" paths from create_dummy_dataset

    Example:
        >>> dummy_data = create_dummy_dataset(num_samples=5)
        >>> # ... use dummy_data for testing ...
        >>> cleanup_dummy_dataset(dummy_data)  # Clean up temp files
    """
    if not data_list:
        return

    temp_dirs: set[str] = set()

    for item in data_list:
        image_path = item.get("image")
        if not image_path:
            continue

        image_path = str(image_path)
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass

        temp_dir = os.path.dirname(image_path)
        if temp_dir:
            temp_dirs.add(temp_dir)

    for temp_dir in temp_dirs:
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


def create_dummy_dataset(
    spatial_size: Tuple[int, int, int] = (256, 256, 192),
    num_samples: int = 10,
) -> List[Dict[str, Any]]:
    """
    Create a dummy dataset for testing pipeline functionality.

    Generates random 3D numpy arrays and saves them as NIfTI files
    in a temporary directory. Useful for pipeline testing without
    real MRI data.

    WARNING: This function creates real NIfTI files on disk.
    After testing, you MUST call cleanup_dummy_dataset(data_list) to delete
    the temporary files and prevent disk space leakage.

    Args:
        spatial_size: Spatial dimensions (D, H, W) of dummy images
        num_samples: Number of dummy samples to create

    Returns:
        List of data dictionaries with paths to dummy NIfTI files

    Example:
        >>> dummy_data = create_dummy_dataset(num_samples=5)
        >>> # ... use dummy_data for testing ...
        >>> cleanup_dummy_dataset(dummy_data)  # Clean up temp files
    """
    import tempfile

    import nibabel as nib

    temp_dir = tempfile.mkdtemp(prefix="adynamics_dummy_")
    data_list: List[Dict[str, Any]] = []

    for i in range(num_samples):
        D, H, W = spatial_size
        dummy_data = np.random.rand(D, H, W).astype(np.float32)

        x = np.linspace(-1, 1, D)
        y = np.linspace(-1, 1, H)
        z = np.linspace(-1, 1, W)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        ellipsoid = (xx**2 + yy**2 + zz**2) <= 0.8

        dummy_data = dummy_data * ellipsoid + 0.3 * ellipsoid
        dummy_data = (dummy_data / dummy_data.max()).astype(np.float32)

        affine = np.eye(4)
        nii_image = nib.Nifti1Image(dummy_data, affine)

        filename = os.path.join(temp_dir, f"dummy_T1_{i:04d}.nii.gz")
        nib.save(nii_image, filename)

        label = np.random.randint(0, 4)

        data_list.append({
            "image": filename,
            "label": int(label),
        })

    return data_list