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

    # Default per-modality spatial target sizes.
    # Design goal: all 3D modalities share the same spatial size (128, 128, 128)
    # so that 4x downsampling produces [B, C, 8, 8, 8] for every modality,
    # matching the T1 latent grid exactly. This eliminates AdaptiveAvgPool3d
    # interpolation on auxiliary modalities, which previously caused severe
    # information loss (e.g., FLAIR 32→2→8 layers, ASL 32→2→8 layers).
    #
    #   T1    (128, 128, 128) — structural backbone
    #   fMRI  (64, 64, 34)    — fMRIDeepEncoder has dedicated architecture
    #   ASL   (128, 128, 128) — unified: was (64, 64, 32)
    #   QSM   (128, 128, 128) — unified: was (128, 128, 96)
    #   FLAIR (128, 128, 128) — unified: was (128, 128, 32)
    DEFAULT_SPATIAL_SIZES: Dict[str, Tuple[int, int, int]] = {
        "t1":    (128, 128, 128),
        "fmri":  (64, 64, 34),       # fMRIDeepEncoder native (3D spatial part)
        "asl":   (128, 128, 128),    # unified cube
        "qsm":   (128, 128, 128),    # unified cube
        "flair": (128, 128, 128),    # unified cube
    }
    # fMRI-specific temporal target (number of BOLD volumes to keep)
    # Literature: ≥60 volumes (≈2-3 min at TR=3s) is bare minimum for
    # deep-learning classification of rs-fMRI (Van Dijk 2010, Meier 2024).
    # ADNI3 basic protocol: ~200 volumes at TR=3s. T=60 captures central
    # ~3 min — sufficient for the fMRIDeepEncoder's learned features.
    DEFAULT_FMRI_T_TARGET: int = 60

    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        transform: Optional[Any] = None,
        spatial_sizes: Optional[Dict[str, Tuple[int, int, int]]] = None,
        fmri_t_target: Optional[int] = None,
        required_modality: str = "t1",
        optional_modalities: Optional[List[str]] = None,
        preserve_temporal_dim: bool = True,
        use_npy_cache: bool = True,
        npy_cache_dir: Optional[str] = None,
        preload_in_memory: bool = False,
        ram_dtype: str = "float16",
        precomputed_path: Optional[str] = None,
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
            use_npy_cache: If True (default), every .nii.gz read is cached
                as a sibling .npy file the first time it's read, and
                subsequent reads load the .npy directly. Skips the
                gzip-decompress + nibabel header parse, the two biggest
                costs in the data loader. First epoch is a bit slower
                (writing caches), later epochs are 5-10x faster.
            npy_cache_dir: If given, .npy files are written to this dir
                using a hash of the original path as the filename. Useful
                when the source data dir is read-only. If None (default),
                the cache lives next to the original .nii.gz.
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
        # .npy cache: skip the gzip-decompress + nibabel-header cost on
        # subsequent reads. Each cache file is a flat float32 array matching
        # the underlying voxel data of the source .nii.gz.
        self.use_npy_cache = use_npy_cache
        self.npy_cache_dir = npy_cache_dir
        if npy_cache_dir is not None:
            os.makedirs(npy_cache_dir, exist_ok=True)

        # In-RAM preload: load every sample's arrays into a dict at
        # __init__ so __getitem__ becomes a pure dict lookup (no I/O,
        # no nibabel parse). The fp32 .npy cache is ~211 GB which
        # doesn't fit in 128 GB, so we cast to ram_dtype (default fp16,
        # ~110 GB). The model trains in fp32, so __getitem__ casts
        # back. Trades ~1ms of cast time per sample for zero disk I/O
        # during training — keeps the GPU saturated instead of bursty
        # 70-100% while DataLoader workers fight the disk.
        self.preload_in_memory = preload_in_memory
        self.ram_dtype = np.dtype(ram_dtype)
        self._in_memory_pool: Optional[List[Dict[str, Optional[np.ndarray]]]] = None
        if self.preload_in_memory:
            self._preload_all()

        # Precomputed cache: all transforms already applied.
        # Supports single .pt file or chunked directory.
        # Chunked mode uses LRU cache to avoid loading all data into RAM.
        self._precomputed: Optional[Dict[int, Dict[str, Any]]] = None
        self._precomputed_dir: Optional[str] = None
        self._precomputed_index: Optional[dict] = None
        if precomputed_path is not None:
            if os.path.isdir(precomputed_path):
                # Chunked precomputed cache
                index_path = os.path.join(precomputed_path, "index.json")
                if os.path.exists(index_path):
                    with open(index_path) as _f:
                        self._precomputed_index = json.load(_f)
                    self._precomputed_dir = precomputed_path
                    print(f"[Data] Using chunked precomputed cache: {precomputed_path} ({self._precomputed_index['ok']} samples)")
            elif os.path.exists(precomputed_path):
                print(f"[Data] Loading precomputed cache: {precomputed_path}")
                self._precomputed = torch.load(precomputed_path, map_location="cpu", weights_only=False)
                print(f"[Data] Loaded {len(self._precomputed)} precomputed samples")

        # Lazy transform cache: caches each sample's transform result on first
        # access. Epoch 1 is slow (transforms run), epoch 2+ is fast (dict lookup).
        # Memory: ~128MB/sample × 1468 samples ≈ 187 GB (fp16) — too much for RAM.
        # So we cache to disk instead (precomputed_path), OR use this in-memory
        # dict for small datasets only.
        self._lazy_cache: Optional[Dict[int, Dict[str, Any]]] = None

    def _load_precomputed(self, idx: int) -> Optional[Dict[str, Any]]:
        """Load a precomputed sample from single-file or chunked cache.

        Uses _manifest_idx from the data item to look up the correct sample
        in the precomputed cache (which is keyed by manifest index, not dataset index).
        Falls back to idx if _manifest_idx is not set (backward compat).
        """
        if self._precomputed is None and self._precomputed_dir is None:
            return None

        # Map dataset index -> manifest index
        item = self.data[idx]
        if isinstance(item, dict) and "_manifest_idx" in item:
            manifest_idx = item["_manifest_idx"]
        else:
            manifest_idx = idx

        if self._precomputed is not None:
            return self._precomputed.get(manifest_idx)

        if self._precomputed_dir is not None and self._precomputed_index is not None:
            # Chunked: find which chunk contains this manifest_idx
            chunk_size = 50
            chunk_start = (manifest_idx // chunk_size) * chunk_size
            chunk_path = os.path.join(self._precomputed_dir, f"chunk_{chunk_start:05d}.pt")
            if not os.path.exists(chunk_path):
                return None
            # LRU cache: avoid re-loading the same chunk from disk
            if not hasattr(self, '_chunk_cache'):
                self._chunk_cache: Dict[int, Dict] = {}
                self._chunk_cache_order: list = []
            if chunk_start not in self._chunk_cache:
                chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
                self._chunk_cache[chunk_start] = chunk
                self._chunk_cache_order.append(chunk_start)
                # Keep at most 10 chunks in memory (~20GB)
                while len(self._chunk_cache) > 10:
                    old = self._chunk_cache_order.pop(0)
                    del self._chunk_cache[old]
            return self._chunk_cache[chunk_start].get(manifest_idx)
        return None

    def _load_npy_cached(self, nifti_path: str) -> np.ndarray:
        """
        Load a NIfTI file as a float32 numpy array, with a .npy on-disk
        cache to skip the gzip-decompress + nibabel-header cost on
        subsequent reads.

        Cache layout:
          - Default: cache lives next to the .nii.gz, named
            `<basename>.npy` (the .nii.gz extension is dropped). E.g.
            `sub-001_T1w.nii.gz` → `sub-001_T1w.npy`.
          - If `npy_cache_dir` is set: cache is
            `<npy_cache_dir>/<sha1-of-path>.npy` so different paths to
            the same file (e.g. /E: vs /D: on Windows) share a cache.

        Concurrency:
          - Read is O(1) when cache exists.
          - First read decodes .nii.gz via nibabel, writes the cache
            atomically (write to `<cache>.tmp`, rename) so partial
            writes from a crashed worker don't leave a corrupt cache
            that subsequent reads would happily load.

        Returns: float32 ndarray of shape matching the source NIfTI
        data array. Returns the same thing nibabel.get_fdata() /
        np.asarray(img.dataobj, dtype=float32) would, but ~5-10x
        faster on a warm cache.
        """
        if not self.use_npy_cache:
            # Cache disabled — fall through to nibabel.
            import nibabel as nib
            img = nib.load(str(nifti_path))
            return np.asarray(img.dataobj, dtype=np.float32)

        if self.npy_cache_dir is not None:
            import hashlib
            # Normalize the path before hashing so that different
            # string representations of the same file (`E:\...` vs
            # `E:/...` vs `E:\\...`) all hash to the same key.
            norm = os.path.normpath(nifti_path).replace(os.sep, "/")
            h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
            cache_path = os.path.join(self.npy_cache_dir, h + ".npy")
        else:
            # Sibling to original: foo.nii.gz → foo.npy
            if nifti_path.endswith(".nii.gz"):
                cache_path = nifti_path[:-7] + ".npy"
            elif nifti_path.endswith(".nii"):
                cache_path = nifti_path[:-4] + ".npy"
            else:
                cache_path = nifti_path + ".npy"

        # Fast path: cache hit
        if os.path.exists(cache_path):
            try:
                arr = np.load(cache_path, mmap_mode=None)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32, copy=False)
                return arr
            except (ValueError, OSError):
                # Corrupt cache file — rebuild.
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        # Slow path: read .nii.gz, write cache atomically.
        import nibabel as nib
        img = nib.load(str(nifti_path))
        arr = np.asarray(img.dataobj, dtype=np.float32)
        try:
            # IMPORTANT: np.save() auto-appends ".npy" if the path doesn't
            # already end with .npy. So `np.save("foo.npy.tmp", arr)` would
            # write to "foo.npy.tmp.npy" — broken. We write the tmp file
            # to a path that ALREADY ends in .npy, then atomically rename
            # it to the final cache path.
            tmp_path = cache_path + ".tmp.npy"
            np.save(tmp_path, arr)
            # Atomic rename (Windows: os.replace is atomic on same volume)
            os.replace(tmp_path, cache_path)
        except OSError:
            # Cache write failure (read-only disk, etc.) — non-fatal.
            pass
        return arr

    def _preload_all(self) -> None:
        """
        Walk every sample, load every existing .npy into a dict, cast to
        `self.ram_dtype` (default fp16) to fit in RAM. Reports a memory
        budget estimate first and aborts cleanly if it'd exceed 80% of
        total RAM (so the system stays responsive).

        Only available with use_npy_cache=True. num_workers MUST be 0
        in the DataLoader — workers on Windows use spawn, which would
        copy the entire pool into each worker and OOM.

        Expected cost on a 128 GB host, fp16:
            T1:    ~24 GB  (1472 × 16.5 MB)
            fMRI:  ~77 GB  (~1200 × 64 MB)
            ASL:   ~0.4 GB
            QSM:   ~4.5 GB
            FLAIR: ~3.5 GB
            Total: ~110 GB
        """
        import psutil  # type: ignore
        import time as _t
        vm = psutil.virtual_memory()
        avail_gb = vm.available / 1024**3
        total_gb = vm.total / 1024**3
        # First pass: count how many samples have each modality
        n_samples = len(self.data)
        n_mods_per_sample: Dict[str, int] = {m: 0 for m in ["t1"] + list(self.optional_modalities)}
        for s in self.data:
            for m in n_mods_per_sample:
                p = s.get(m)
                if isinstance(p, str) and os.path.exists(self._cache_path_for(p)):
                    n_mods_per_sample[m] += 1
        # Estimate size from a 1-2 sample probe (cheaper than scanning all)
        probe_arrs = []
        for s in self.data[:3]:
            for m in n_mods_per_sample:
                p = s.get(m)
                if isinstance(p, str) and os.path.exists(self._cache_path_for(p)):
                    probe_arrs.append(np.load(self._cache_path_for(p), mmap_mode="r"))
        if probe_arrs:
            bytes_per_elem = np.dtype(self.ram_dtype).itemsize
            est_gb = sum(a.nbytes for a in probe_arrs) / len(probe_arrs) / 1024**3
            est_gb *= n_samples
            est_gb *= bytes_per_elem / 4.0  # adjust for dtype ratio (probe is fp32)
        else:
            est_gb = 0
        # Safety: refuse if estimate > 80% of total RAM
        if est_gb > 0.8 * total_gb:
            raise RuntimeError(
                f"[preload] estimated {est_gb:.1f} GB needed but only {total_gb:.0f} GB total RAM. "
                f"Drop a modality or switch ram_dtype='float16' to 'float32' if you have more RAM."
            )
        print(f"[preload] loading {n_samples} samples into RAM (dtype={self.ram_dtype.name}, "
              f"~{est_gb:.1f} GB estimated, {avail_gb:.0f} GB available)...", flush=True)
        t0 = _t.time()
        loaded_bytes = 0
        n_loaded = 0
        self._in_memory_pool = []
        # Progress every 100 samples
        for i, s in enumerate(self.data):
            entry: Dict[str, Optional[np.ndarray]] = {"_t1_meta": s}  # keep raw paths
            for m in ["t1"] + list(self.optional_modalities):
                p = s.get(m)
                if not isinstance(p, str) or not os.path.exists(self._cache_path_for(p)):
                    entry[m] = None
                    continue
                arr = np.load(self._cache_path_for(p), mmap_mode=None)
                if arr.dtype != self.ram_dtype:
                    arr = arr.astype(self.ram_dtype, copy=False)
                entry[m] = arr
                loaded_bytes += arr.nbytes
            self._in_memory_pool.append(entry)
            n_loaded += 1
            if (i + 1) % 200 == 0 or (i + 1) == n_samples:
                dt = _t.time() - t0
                rate = (i + 1) / max(dt, 0.01)
                eta = (n_samples - i - 1) / max(rate, 0.01)
                print(f"[preload] {i+1}/{n_samples}  loaded={n_loaded}  "
                      f"bytes={loaded_bytes/1024**3:.1f} GB  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)
        dt = _t.time() - t0
        print(f"[preload] DONE in {dt:.1f}s  ({loaded_bytes/1024**3:.1f} GB in RAM)", flush=True)

    def _cache_path_for(self, nifti_path: str) -> str:
        """Same hash logic as _load_npy_cached — exposed for _preload_all."""
        if self.npy_cache_dir is not None:
            import hashlib
            norm = os.path.normpath(nifti_path).replace(os.sep, "/")
            h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
            return os.path.join(self.npy_cache_dir, h + ".npy")
        if nifti_path.endswith(".nii.gz"):
            return nifti_path[:-7] + ".npy"
        if nifti_path.endswith(".nii"):
            return nifti_path[:-4] + ".npy"
        return nifti_path + ".npy"

    def _resize_spatial_3d(self, data: np.ndarray, target_dhw: Tuple[int, int, int]) -> np.ndarray:
        """Resize a 3D numpy array (D, H, W) to (target_D, target_H, target_W)
        using trilinear interpolation. If shapes already match, returns as-is.
        """
        if data.shape == target_dhw:
            return data
        t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
        t = F.interpolate(t, size=target_dhw, mode="trilinear", align_corners=False)
        return t.squeeze(0).squeeze(0).numpy()

    def _normalize_fmri_t(self, data: np.ndarray, training: bool = True,
                           in_dtype: Optional[np.dtype] = None) -> np.ndarray:
        """Normalize a 4D fMRI (D, H, W, T) tensor:

        1. Resize spatial (D, H, W) to self.spatial_sizes['fmri'] (default 64,64,34).
        2. Resample T to self.fmri_t_target (default 60):
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
        # When called from the in-RAM preload path, the input is already
        # in fp16 (or whichever ram_dtype). Keep it in that dtype all the
        # way through — casting to fp32 needs ~4x memory for the
        # temporaries below, which OOMs when the 89 GB preload pool
        # is also in RAM. The model casts to fp32 internally when it
        # sees the tensor in `forward()`.
        out_dtype = in_dtype if in_dtype is not None else np.float32

        # Step 1: spatial resize via per-volume loop (F.interpolate doesn't
        # accept 5D input with a different mode per dim).
        if data.shape[:3] != target_dhw:
            # data is (D, H, W, T). Loop over T slices.
            T_src = data.shape[3]
            resized = np.empty((*target_dhw, T_src), dtype=out_dtype)
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
        std = data.std(axis=(0, 1, 2), keepdims=True).astype(out_dtype) + np.float32(1e-8)
        data = (data - mean.astype(out_dtype)) / std
        # Cast to fp32 at the end if we started in fp16 (model wants fp32
        # tensors; the in-flight arithmetic on the 200 T-points stays in
        # fp16 to halve the temporaries). The single 111 MB output cast
        # is one allocation, not four.
        if out_dtype != np.float32:
            data = data.astype(np.float32, copy=False)
        return data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a multi-modal sample.

        All 3D modalities (T1, ASL, QSM, FLAIR) share the same spatial size
        (128, 128, 128) so that 4x downsampling produces identical latent grids
        without AdaptiveAvgPool3d interpolation. fMRI keeps (64, 64, 34) with
        its own temporal handling via fMRIDeepEncoder.

        Returns:
            Dictionary containing:
                - t1: preprocessed T1 tensor [1, 128, 128, 128]
                - asl/qsm/flair: tensors [1, 128, 128, 128] (or None if missing)
                - fmri: tensor [1, 64, 64, 34, T] (or None if missing)
                - label: disease stage
                - patient_id
                - available_modalities: list of available modality names
        """
        # Fast path: precomputed cache (all transforms already applied)
        if self._precomputed is not None or self._precomputed_dir is not None:
            entry = self._load_precomputed(idx)
            if entry is not None:
                if "error" in entry:
                    raise RuntimeError(f"Precomputed sample {idx} had error: {entry['error']}")

                # One-time shape validation: detect stale cache (old spatial sizes)
                if not getattr(self, '_precomputed_validated', False):
                    self._precomputed_validated = True
                    for mod, expected_size in self.spatial_sizes.items():
                        if mod == "fmri":
                            continue  # fMRI has special handling
                        val = entry.get(mod)
                        if val is not None and isinstance(val, torch.Tensor):
                            # Shape is [D,H,W] or [1,D,H,W]
                            shape = val.shape
                            spatial = shape[-3:] if len(shape) >= 3 else shape
                            if tuple(spatial) != expected_size:
                                print(f"[WARN] Precomputed cache shape mismatch for {mod}: "
                                      f"got {tuple(spatial)}, expected {expected_size}. "
                                      f"Cache may be stale. Rebuild with scripts/precompute_cache.py "
                                      f"or pass --no_precomputed to skip the cache.")
                                break

                result = {}
                for key, val in entry.items():
                    if isinstance(val, torch.Tensor):
                        t = val.float()  # cast fp16 → fp32 for model
                        # Ensure channel dim: 3D modalities need [C,D,H,W],
                        # fMRI 4D needs [C,D,H,W,T]. Add if missing.
                        if key == "fmri" and t.ndim == 4:
                            t = t.unsqueeze(0)  # [D,H,W,T] -> [1,D,H,W,T]
                            # Truncate/pad time to fmri_t_target if needed
                            if self.fmri_t_target and t.shape[-1] != self.fmri_t_target:
                                T_cur = t.shape[-1]
                                if T_cur > self.fmri_t_target:
                                    t = t[..., :self.fmri_t_target]
                                elif T_cur < self.fmri_t_target:
                                    pad = torch.zeros(*t.shape[:-1], self.fmri_t_target - T_cur)
                                    t = torch.cat([t, pad], dim=-1)
                        elif key != "fmri" and t.ndim == 3:
                            t = t.unsqueeze(0)  # [D,H,W] -> [1,D,H,W]
                        result[key] = t
                    else:
                        result[key] = val
                return result

        data_item = self.data[idx]

        result = {}
        available_modalities = []

        # In-RAM preload fast path: data is already in self._in_memory_pool.
        # Skip the I/O entirely. We still validate the T1 shape here so
        # corrupted samples get the same exception as the on-disk path.
        if self._in_memory_pool is not None:
            entry = self._in_memory_pool[idx]
            t1_arr = entry.get("t1")
            t1_path = data_item.get("t1") or data_item.get(self.required_modality)
            if t1_arr is None or t1_arr.ndim != 3 or any(s == 0 for s in t1_arr.shape):
                raise ValueError(f"Corrupted T1 in pool idx={idx}, path={t1_path}, shape={t1_arr.shape if t1_arr is not None else None}")
            # Cast back to fp32 (the model trains in fp32). T1 is ~33 MB
            # so this 1 ms cast is negligible. For optional modalities we
            # ALSO do fp16-to-fp32 here when the modality is small (ASL /
            # QSM / FLAIR — at most 33 MB). The big one is fMRI 4D: at
            # fp32 it would be 111 MB and the downstream _normalize_fmri_t
            # would need 3-4 fp32 temporaries (resized, mean, std, zscore
            # result) = ~500 MB on top of the already-89 GB pool. With the
            # 128 GB host's pagefile commit + working-set pressure this
            # trips _ArrayMemoryError. Solution: keep fMRI in fp16 ALL
            # THE WAY through _normalize_fmri_t — the model casts the
            # final tensor to fp32 internally when it sees a fp16 input.
            result["t1"] = torch.from_numpy(t1_arr.astype(np.float32, copy=False)).unsqueeze(0)
            for mod in self.optional_modalities:
                arr = entry.get(mod)
                if arr is None:
                    result[mod] = None
                    continue
                if mod == "fmri" and arr.ndim == 4 and self.preserve_temporal_dim:
                    # Keep fp16 throughout the normalization — avoids
                    # 4x 111 MB fp32 temporaries.
                    normalized = self._normalize_fmri_t(arr,
                                                         training=self.transform is not None,
                                                         in_dtype=arr.dtype)
                    result[mod] = torch.from_numpy(normalized).unsqueeze(0)
                    available_modalities.append(mod)
                elif mod == "fmri" and arr.ndim == 4 and not self.preserve_temporal_dim:
                    data_mean = arr.mean(axis=-1)
                    data_resized = self._resize_spatial_3d(data_mean, self.spatial_sizes["fmri"])
                    result[mod] = torch.from_numpy(data_resized).unsqueeze(0).unsqueeze(0).squeeze(0)
                    available_modalities.append(mod)
                elif arr.ndim == 3:
                    data_resized = self._resize_spatial_3d(arr.astype(np.float32, copy=False),
                                                          self.spatial_sizes.get(mod, self.target_size))
                    result[mod] = torch.from_numpy(data_resized).unsqueeze(0)
                    available_modalities.append(mod)
                else:
                    result[mod] = None
            # Apply MONAI transforms to T1 (resize, intensity norm, etc.)
            if self.transform is not None:
                data_dict = self.transform({"t1": t1_path})  # transform reads t1_path via nibabel/MONAI
                result["t1"] = data_dict["t1"]
            else:
                raise ValueError("Transform is required for T1 preprocessing")
            # Label + metadata
            result["label"] = data_item.get("label", 0)
            result["patient_id"] = data_item.get("patient_id", f"unknown_{idx}")
            result["available_modalities"] = available_modalities
            # Demographics (same as the slow path)
            raw_age = data_item.get("age", None)
            if raw_age is None or raw_age == "":
                age_val = 0.0
            else:
                try: age_val = float(raw_age)
                except (TypeError, ValueError): age_val = 0.0
            raw_sex = data_item.get("sex", None)
            if raw_sex is None or raw_sex == "":
                sex_val = 0
            else:
                try: sex_val = int(raw_sex)
                except (TypeError, ValueError): sex_val = 0
            result["age"] = torch.tensor(age_val, dtype=torch.float32)
            result["sex"] = torch.tensor(sex_val, dtype=torch.long)
            return result

        # T1 path (required) - validate before passing to MONAI transforms
        t1_path = data_item.get("t1") or data_item.get(self.required_modality)
        if not t1_path or not os.path.exists(t1_path):
            raise FileNotFoundError(f"Required modality not found: {t1_path}")

        # Validate T1 file dimensions (catch corrupted [0,0,0] files)
        try:
            t1_data = self._load_npy_cached(str(t1_path))
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
                data = self._load_npy_cached(str(path))
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
                    "t1": (1, 128, 128, 128),
                    "fmri": (1, 64, 64, 34, 100),
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