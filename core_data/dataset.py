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
import torch.nn.functional as F
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
    use_cache: bool = True,
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
        use_cache: If True, use CacheDataset with cache_rate. Default: True (caching enabled)
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

    # Note: test_split is derived from train_split + val_split complement.
    # When train_split + val_split = 1.0, test_split = 0.0 (no test set created).
    # This is intentional for train-only workflows; test_transforms parameter
    # becomes required by API signature but is unused when test_split=0.0.
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


class MultiModalDataset(Dataset):
    """
    Multi-Modal MRI Dataset for ADynamics.

    Supports T1 (required) + optional modalities (fMRI, ASL, QSM, FLAIR).
    Handles missing modalities gracefully by returning None for missing files.

    Each sample returns:
        - x_dict: Dict[str, Tensor] - modality tensors (T1 required, others optional)
        - label: int - disease stage (0=NC, 1=SCD, 2=MCI, 3=AD)
        - patient_id: str
        - available_modalities: List[str] - which optional modalities were loaded

    Example:
        >>> dataset = MultiModalDataset(
        ...     data_list=data_list,
        ...     transform=transforms,
        ...     spatial_sizes={'t1': (256,256,192), 'fmri': (34,64,64), ...}
        ... )
        >>> sample = dataset[0]
        >>> print(sample.keys())  # ['t1', 'fmri', 'asl', 'qsm', 'flair', 'label', 'patient_id', 'available']
    """

    # Default per-modality spatial target sizes, derived from real data survey:
    #   T1    actual (197, 233, 189) → target (256, 256, 192) — must match
    #         the decoder output (the VAE's internal latent grid is hard-
    #         coded to (16,16,12) and decoder_depth=4 → 16x upsample to
    #         (256,256,192)). T1 is padded/cropped to that size so the
    #         reconstruction loss has matching shapes.
    #   fMRI  actual (64, 64, 34, T)  → target (64, 64, 34, 200) — keeps
    #         native spatial (BOLD is 3.5mm), trims/pads T to 200.
    #   ASL   actual (128, 128, 32)   → target (64, 64, 32)   — perfusion
    #         spatial is coarse, halve for memory.
    #   QSM   actual (256, 256, 124)  → target (128, 128, 96) — venous
    #         detail worth keeping, halve D/H to free memory.
    #   FLAIR actual (256, 256, 22)   → target (128, 128, 32) — only 22
    #         slices, upsample W to 32 and halve D/H.
    DEFAULT_SPATIAL_SIZES: Dict[str, Tuple[int, int, int]] = {
        "t1":   (256, 256, 192),
        "fmri": (64, 64, 34),       # 3D part of 4D fMRI
        "asl":  (64, 64, 32),
        "qsm":  (128, 128, 96),
        "flair": (128, 128, 32),
    }
    # fMRI-specific temporal target (number of BOLD volumes to keep)
    DEFAULT_FMRI_T_TARGET: int = 200

    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        transform: Optional[Any] = None,
        spatial_sizes: Optional[Dict[str, Tuple[int, int, int]]] = None,
        fmri_t_target: Optional[int] = None,
        required_modality: str = "t1",
        optional_modalities: Optional[List[str]] = None,
        preserve_temporal_dim: bool = True,
    ) -> None:
        """
        Initialize multi-modal dataset.

        Args:
            data_list: List of data dictionaries with modality paths and labels
            transform: MONAI transform to apply
            spatial_sizes: Dict mapping modality -> (D, H, W) target size.
                If None, uses DEFAULT_SPATIAL_SIZES (per-modality, NOT all
                forced to T1's size). This was changed from the original
                "all modalities → (256,256,192)" because each modality's
                physical spatial resolution is different:
                  T1=1mm, fMRI=3.5mm, ASL≈4mm, QSM≈1mm, FLAIR≈5mm.
                Forcing fMRI (3.5mm) to 256×256×192 wastes 10x memory on
                artificial super-resolution.
            fmri_t_target: Number of BOLD timepoints to normalize to. If a
                sample has more, take the middle segment; if fewer, zero-pad
                at the end. Default 200 (covers ~95% of files in survey).
            required_modality: The required modality (default: "t1")
            optional_modalities: List of optional modality names (default:
                ["fmri", "asl", "qsm", "flair"])
            preserve_temporal_dim: If True (default), 4D fMRI keeps its time
                dimension and is returned as 5D tensor [1, D, H, W, T] for
                the fMRITemporalEncoder. If False, time is averaged (legacy).
        """
        super().__init__(data=data_list, transform=transform)
        # Per-modality spatial targets. Each modality uses its own — see
        # DEFAULT_SPATIAL_SIZES docstring for the survey justification.
        self.spatial_sizes = dict(self.DEFAULT_SPATIAL_SIZES)
        if spatial_sizes:
            self.spatial_sizes.update(spatial_sizes)
        self.fmri_t_target = fmri_t_target if fmri_t_target is not None else self.DEFAULT_FMRI_T_TARGET
        self.required_modality = required_modality
        self.optional_modalities = optional_modalities or ["fmri", "asl", "qsm", "flair"]
        # Legacy attribute: target_size is T1's target (kept for any caller
        # that still reads it). New code should use spatial_sizes["t1"].
        self.target_size = self.spatial_sizes.get("t1", (192, 192, 160))
        self.preserve_temporal_dim = preserve_temporal_dim

    def _resize_spatial_3d(self, data: np.ndarray, target_dhw: Tuple[int, int, int]) -> np.ndarray:
        """Resize a 3D numpy array (D, H, W) to (target_D, target_H, target_W)
        using trilinear interpolation. If shapes already match, returns as-is.
        """
        if data.shape == target_dhw:
            return data
        t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
        t = F.interpolate(t, size=target_dhw, mode="trilinear", align_corners=False)
        return t.squeeze(0).squeeze(0).numpy()

    def _normalize_fmri_t(self, data: np.ndarray, training: bool = True) -> np.ndarray:
        """Normalize a 4D fMRI (D, H, W, T) tensor:

        1. Resize spatial (D, H, W) to self.spatial_sizes['fmri'] (default 64,64,34).
        2. Resample T to self.fmri_t_target (default 200):
           - T > target: take the middle segment (deterministic, keeps
             steady-state BOLD; drops calibration frames at start/end).
           - T < target: zero-pad at the end.
           - T == target: as-is.
           - 4D: random segment during training (augmentation); middle
             segment during eval/test (deterministic). Caller passes
             `training` accordingly.
        3. Per-volume (per-timepoint) z-score: for each t, subtract
           mean over (D,H,W) and divide by std over (D,H,W). This kills
           global BOLD drift and makes BOLD units comparable across
           subjects / scanners.

        Returns: np.ndarray of shape (D', H', W', T_target).
        """
        target_dhw = tuple(self.spatial_sizes["fmri"])
        T_target = self.fmri_t_target

        # Step 1: spatial resize via per-volume loop (F.interpolate doesn't
        # accept 5D input with a different mode per dim).
        if data.shape[:3] != target_dhw:
            # data is (D, H, W, T). Loop over T slices.
            T_src = data.shape[3]
            resized = np.empty((*target_dhw, T_src), dtype=np.float32)
            for t in range(T_src):
                resized[..., t] = self._resize_spatial_3d(data[..., t], target_dhw)
            data = resized

        # Step 2: temporal trim or pad to T_target
        T_src = data.shape[3]
        if T_src > T_target:
            if training:
                # random crop in training for augmentation
                max_start = T_src - T_target
                start = int(np.random.randint(0, max_start + 1))
            else:
                # deterministic middle segment at eval/test
                start = (T_src - T_target) // 2
            data = data[..., start:start + T_target]
        elif T_src < T_target:
            pad = T_target - T_src
            data = np.pad(data, ((0, 0), (0, 0), (0, 0), (0, pad)), mode="constant", constant_values=0.0)

        # Step 3: per-volume z-score
        # mean / std over (D, H, W) per timepoint, keepdims.
        mean = data.mean(axis=(0, 1, 2), keepdims=True)  # (1, 1, 1, T)
        std = data.std(axis=(0, 1, 2), keepdims=True) + 1e-8
        data = (data - mean) / std
        return data.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a multi-modal sample.

        Each modality is resized to its own target size (not forced to T1's).
        fMRI is additionally normalized in time (trim/pad to T=200) and
        per-volume z-scored to kill BOLD drift.

        Returns:
            Dictionary containing:
                - t1: preprocessed T1 tensor (resize to spatial_sizes['t1'])
                - other modalities: per-modality-sized tensors
                - label: disease stage
                - patient_id
                - available_modalities: list of available modality names
        """
        data_item = self.data[idx]

        result = {}
        available_modalities = []

        # T1 path (required) - validate before passing to MONAI transforms
        t1_path = data_item.get("t1") or data_item.get(self.required_modality)
        if not t1_path or not os.path.exists(t1_path):
            raise FileNotFoundError(f"Required modality not found: {t1_path}")

        # Validate T1 file dimensions (catch corrupted [0,0,0] files)
        try:
            import nibabel as nib
            t1_img = nib.load(str(t1_path))
            t1_data = t1_img.get_fdata()
            if t1_data.ndim != 3 or any(s == 0 for s in t1_data.shape):
                raise ValueError(f"Corrupted T1 file: {t1_path} has shape {t1_data.shape}")
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise FileNotFoundError(f"Cannot load T1 file: {t1_path}") from e

        # Build dict for MONAI transforms (T1 only — we don't run transforms
        # on optional modalities; their per-modality resize happens here)
        data_dict = {"t1": str(t1_path)}

        # Optional modalities - per-modality resize + (fMRI) time normalization
        for mod in self.optional_modalities:
            path = data_item.get(mod)
            if not path or not os.path.exists(path):
                result[mod] = None
                continue
            try:
                import nibabel as nib
                img = nib.load(str(path))
                # Read as float32 to avoid 2x memory for float64 → cast.
                data = np.asarray(img.dataobj, dtype=np.float32)
                # Skip corrupted files with zero dimensions
                if any(s == 0 for s in data.shape):
                    print(f"[WARN] {mod} {path} has zero dim: {data.shape}")
                    result[mod] = None
                    continue

                target_dhw = self.spatial_sizes.get(mod, self.target_size)

                if mod == "fmri" and data.ndim == 4 and self.preserve_temporal_dim:
                    # 4D fMRI: spatial resize + temporal normalize to T_target
                    # + per-volume z-score. Returns [1, D', H', W', T_target].
                    normalized = self._normalize_fmri_t(data, training=self.transform is not None)
                    result[mod] = torch.from_numpy(normalized).unsqueeze(0)
                    available_modalities.append(mod)
                elif mod == "fmri" and data.ndim == 4 and not self.preserve_temporal_dim:
                    # Legacy: average over time, resize spatial, return [1, D, H, W].
                    data = data.mean(axis=-1)
                    data = self._resize_spatial_3d(data, target_dhw)
                    result[mod] = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).squeeze(0)
                    available_modalities.append(mod)
                elif data.ndim == 3:
                    # 3D modality (ASL/QSM/FLAIR or fallback for fMRI)
                    data = self._resize_spatial_3d(data, target_dhw)
                    result[mod] = torch.from_numpy(data).unsqueeze(0)
                    available_modalities.append(mod)
                else:
                    # Unsupported rank
                    print(f"[WARN] {mod} {path} unsupported rank {data.ndim}D shape={data.shape}")
                    result[mod] = None
                    continue
            except Exception as e:
                print(f"[WARN] failed to load {mod} {path}: {e}")
                result[mod] = None

        # Apply MONAI transforms to T1 (resize, intensity norm, etc.)
        if self.transform is not None:
            data_dict = self.transform(data_dict)
            result["t1"] = data_dict["t1"]
        else:
            raise ValueError("Transform is required for T1 preprocessing")

        # Label and metadata
        result["label"] = data_item.get("label", 0)
        result["patient_id"] = data_item.get("patient_id", f"unknown_{idx}")
        result["available_modalities"] = available_modalities

        # Demographics: always include keys (with safe defaults) so collate is
        # uniform regardless of whether use_demographic_cond is on or off.
        # Missing values become safe defaults (age=0, sex=0[unknown]).
        raw_age = data_item.get("age", None)
        raw_sex = data_item.get("sex", None)
        if raw_age is None or raw_age == "":
            age_val = 0.0
        else:
            try:
                age_val = float(raw_age)
            except (TypeError, ValueError):
                age_val = 0.0
        if raw_sex is None or raw_sex == "":
            sex_val = 0  # 0=unknown
        else:
            try:
                sex_val = int(raw_sex)
            except (TypeError, ValueError):
                sex_val = 0
        result["age"] = torch.tensor(age_val, dtype=torch.float32)
        result["sex"] = torch.tensor(sex_val, dtype=torch.long)

        return result


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple collate function that stacks all tensors and handles missing modalities.

    Handles the case where fMRI is 5D [B, 1, D, H, W, T] (preserve_temporal_dim=True)
    vs 3D [B, 1, D, H, W] (legacy time-averaged). Within a single batch, all fMRI
    samples must have the same rank — the dataset guarantees this via
    preserve_temporal_dim. Missing-modality zero-fills use the reference shape.
    """
    result = {}
    keys = ["t1", "fmri", "asl", "qsm", "flair", "label", "patient_id", "available_modalities", "age", "sex"]

    for key in keys:
        values = [item.get(key) for item in batch]

        if key == "label":
            result[key] = torch.tensor(values, dtype=torch.long)
        elif key == "age":
            # Stack as float [B]; missing values default to 0.0
            arr = []
            for v in values:
                if v is None:
                    arr.append(0.0)
                elif isinstance(v, torch.Tensor):
                    arr.append(float(v.item()))
                else:
                    try:
                        arr.append(float(v))
                    except (TypeError, ValueError):
                        arr.append(0.0)
            result[key] = torch.tensor(arr, dtype=torch.float32)
        elif key == "sex":
            # Stack as long [B]; missing values default to 0 (unknown)
            arr = []
            for v in values:
                if v is None:
                    arr.append(0)
                elif isinstance(v, torch.Tensor):
                    arr.append(int(v.item()))
                else:
                    try:
                        arr.append(int(v))
                    except (TypeError, ValueError):
                        arr.append(0)
            result[key] = torch.tensor(arr, dtype=torch.long)
        elif key == "patient_id":
            result[key] = values
        elif key == "available_modalities":
            result[key] = values
        elif key in ["t1", "fmri", "asl", "qsm", "flair"]:
            # Stack valid tensors, use zeros for None
            valid_vals = [v for v in values if v is not None and isinstance(v, torch.Tensor)]
            if valid_vals:
                ref_shape = valid_vals[0].shape
                tensors = []
                for v in values:
                    if v is None or not isinstance(v, torch.Tensor):
                        tensors.append(torch.zeros(ref_shape, dtype=torch.float32))
                    elif v.shape != ref_shape:
                        # Shape mismatch (e.g. fMRI rank differs across samples).
                        # Defensive: broadcast / reshape to ref_shape.
                        # If v has fewer dims, unsqueeze on the right until they match.
                        v_fixed = v
                        while v_fixed.dim() < len(ref_shape):
                            v_fixed = v_fixed.unsqueeze(-1)
                        # If v has more dims, this is unrecoverable; raise clearly.
                        if v_fixed.dim() != len(ref_shape):
                            raise ValueError(
                                f"Modality {key!r} has mismatched shape in batch: "
                                f"ref={tuple(ref_shape)}, got={tuple(v.shape)}. "
                                f"Check preserve_temporal_dim consistency in dataset."
                            )
                        tensors.append(v_fixed)
                    else:
                        tensors.append(v)
                result[key] = torch.stack(tensors)
            else:
                # Defensive: when ALL samples in the batch lack this modality
                # (rare; only happens with extreme class imbalance + modality
                # dropout), fall back to a known-good shape. We default to a
                # batch-size-1 zero tensor using the per-modality default size
                # (T1D/T1H/T1W for 3D modalities; +T for fMRI). The trainer
                # expects a non-None tensor so it can route through
                # `_normalize_fmri_batch` and the model forward.
                # We rebuild a default from the dataset class if we can find it.
                default_shape = {
                    "t1": (1, 192, 192, 160),
                    "fmri": (1, 64, 64, 34, 200),
                    "asl": (1, 64, 64, 32),
                    "qsm": (1, 128, 128, 96),
                    "flair": (1, 128, 128, 32),
                }.get(key)
                if default_shape is None:
                    result[key] = None
                else:
                    result[key] = torch.zeros((len(values),) + default_shape, dtype=torch.float32)
        else:
            result[key] = values

    return result


def create_multimodal_dataloaders(
    data_list: List[Dict[str, Any]],
    train_transforms,
    val_transforms,
    batch_size: int = 2,
    num_workers: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple:
    """
    Create train/val dataloaders for multi-modal dataset.

    Args:
        data_list: List of data dictionaries
        train_transforms: Transform for training
        val_transforms: Transform for validation
        batch_size: Batch size
        num_workers: Number of workers
        train_split: Training fraction
        val_split: Validation fraction
        shuffle: Shuffle training data
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from sklearn.model_selection import train_test_split

    # Stratified split
    labels = np.array([item.get("label", 0) for item in data_list])
    train_data, val_data = train_test_split(
        data_list,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )

    # Create datasets
    train_dataset = MultiModalDataset(train_data, transform=train_transforms)
    val_dataset = MultiModalDataset(val_data, transform=val_transforms)

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
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader