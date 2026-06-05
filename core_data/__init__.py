"""
ADynamics Core Data Module.

Multi-modal MRI dataset loading, preprocessing transforms, and dataloader
construction for AD progression modeling.

Public API:
    MultiModalDataset       -- Dataset yielding dicts {t1, fmri?, asl?, qsm?, flair?, label, ...}
    multimodal_collate_fn  -- Collate that handles missing modalities (zeros)
    get_multimodal_train_transforms  / get_multimodal_val_transforms
    MULTI_MODAL_SPATIAL_SIZES  -- canonical per-modality target shapes

Legacy single-modal API (still exported for backward compat):
    get_train_val_test_dataloaders
    get_train_transforms, get_val_transforms
"""

from core_data.dataset import (
    MultiModalDataset,
    multimodal_collate_fn,
    create_multimodal_dataloaders,
    # legacy single-modal API
    get_train_val_test_dataloaders,
    create_dummy_dataset,
    cleanup_dummy_dataset,
)
from core_data.transforms import (
    get_multimodal_train_transforms,
    get_multimodal_val_transforms,
    get_train_transforms,
    get_val_transforms,
    MULTI_MODAL_SPATIAL_SIZES,
)

__all__ = [
    # Multi-modal (canonical 4-class + CFM path)
    "MultiModalDataset",
    "multimodal_collate_fn",
    "create_multimodal_dataloaders",
    "get_multimodal_train_transforms",
    "get_multimodal_val_transforms",
    "MULTI_MODAL_SPATIAL_SIZES",
    # Legacy single-modal (kept for backward compat)
    "get_train_val_test_dataloaders",
    "get_train_transforms",
    "get_val_transforms",
    "create_dummy_dataset",
    "cleanup_dummy_dataset",
]
