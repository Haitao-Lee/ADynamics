"""
Stage 1 Multi-Modal VAE Training for ADynamics.

Trains a multi-modal VAE that encodes T1 (required) + optional modalities (fMRI, ASL, QSM, FLAIR)
into a unified latent space, then decodes back to T1 reconstruction while training
a disease classifier.

Usage:
    python scripts/train_stage1_multimodal.py \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --batch_size 2 \
        --epochs 300 \
        --learning_rate 0.0002 \
        --output_dir ./checkpoints/stage1_multimodal
"""

# Must be at very top - before any other imports
import os
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Additional warning suppression for main process
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*allow_smaller.*")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_train_transforms, get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from engine.trainer_vae import MultiModalVAETrainer
from models.vae3d import MultiModalVAE3D
from utils.multi_gpu import setup_data_parallel


def _load_yaml_defaults(config_path: str) -> dict:
    """Load YAML config and map nested keys to argparse argument names."""
    from utils.config_loader import apply_yaml_defaults
    mapping = [
        (("data", "json"), "json"),
        (("data", "num_classes"), "num_classes"),
        (("data", "use_fmri"), "use_fmri"),
        (("data", "use_asl"), "use_asl"),
        (("data", "use_qsm"), "use_qsm"),
        (("data", "use_flair"), "use_flair"),
        (("data", "use_demographic"), "use_demographic"),
        (("data", "t1_only"), "t1_only"),
        (("model", "latent_channels"), "latent_channels"),
        (("model", "base_channels"), "base_channels"),
        (("model", "decoder_depth"), "decoder_depth"),
        (("model", "dropout_rate"), "dropout_rate"),
        (("model", "use_attention"), "use_attention"),
        (("model", "attention_heads"), "attention_heads"),
        (("model", "use_fmri_temporal"), "use_fmri_temporal"),
        (("model", "use_fmri_deep"), "use_fmri_deep"),
        (("model", "fmri_in_channels"), "fmri_in_channels"),
        (("model", "fmri_hidden_dim"), "fmri_hidden_dim"),
        (("model", "fmri_num_pool"), "fmri_num_pool"),
        (("model", "fmri_num_transformer_layers"), "fmri_num_transformer_layers"),
        (("model", "fmri_num_heads"), "fmri_num_heads"),
        (("model", "fmri_deep_n_soft_roi"), "fmri_deep_n_soft_roi"),
        (("model", "fmri_deep_n_transformer_layers"), "fmri_deep_n_transformer_layers"),
        (("model", "fmri_deep_n_heads"), "fmri_deep_n_heads"),
        (("model", "fmri_deep_fc_compression"), "fmri_deep_fc_compression"),
        (("model", "use_t1_centric_fusion"), "use_t1_centric_fusion"),
        (("model", "fmri_t_target"), "fmri_t_target"),
        (("model", "use_checkpointing"), "use_checkpointing"),
        (("training", "batch_size"), "batch_size"),
        (("training", "learning_rate"), "learning_rate"),
        (("training", "weight_decay"), "weight_decay"),
        (("training", "epochs"), "epochs"),
        (("training", "early_stopping_patience"), "early_stopping"),
        (("training", "save_interval"), "save_interval"),
        (("training", "num_gpus"), "num_gpus"),
        (("training", "use_amp"), "use_amp"),
        (("loss", "recon_loss_type"), "recon_loss_type"),
        (("loss", "cls_weight"), "cls_weight"),
        (("loss", "kl_weight"), "kl_weight"),
        (("loss", "kl_warmup_epochs"), "kl_warmup_epochs"),
        (("loss", "free_bits"), "free_bits"),
        (("loss", "contrastive_weight"), "contrastive_weight"),
        (("loss", "gradient_weight"), "gradient_weight"),
        (("loss", "ssim_weight"), "ssim_weight"),
        (("loss", "encoder_grad_boost"), "encoder_grad_boost"),
        (("loss", "ordinal_reg_weight"), "ordinal_reg_weight"),
        (("loss", "class_weights"), "class_weights"),
        # v10: cyclical KL schedule + latent mixup (missing in original mapping
        # so trainer fell back to defaults: kl_strategy="linear", mixup_alpha=0.0).
        # Without these 5 entries, self.config.get(...) in trainer returned
        # defaults and the v10 improvements were silently inactive.
        (("loss", "kl_strategy"), "kl_strategy"),
        (("loss", "kl_cycle_len"), "kl_cycle_len"),
        (("loss", "kl_cycle_low_frac"), "kl_cycle_low_frac"),
        (("loss", "mixup_alpha"), "mixup_alpha"),
        (("loss", "mixup_prob"), "mixup_prob"),
        (("output", "dir"), "output_dir"),
        (("seed",), "seed"),
    ]
    return apply_yaml_defaults(config_path, mapping)


# Local thin alias for the shared helper so the rest of this file is unchanged.
def _resolve_optional_modalities(args) -> list:
    """Wrapper around utils.stage23_compat.resolve_optional_modalities."""
    from utils.stage23_compat import resolve_optional_modalities
    return resolve_optional_modalities(args)


def parse_args():
    # Pre-parse config file (if provided) to set defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    pre_args, _ = pre.parse_known_args()

    config_defaults = {}
    if pre_args.config and os.path.exists(pre_args.config):
        config_defaults = _load_yaml_defaults(pre_args.config)

    parser = argparse.ArgumentParser(description="Stage 1 Multi-Modal VAE Training", parents=[pre])

    # Data
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json",
                        help="Path to dataset JSON manifest")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage1",
                        help="Output directory for checkpoints")

    # Modality toggles (T1 is always required). Default: all 4 ON.
    # Use --no_X to drop a single modality, or --t1_only to drop all 4.
    from utils.stage23_compat import add_modality_args
    add_modality_args(parser)

    # Model
    parser.add_argument("--latent_channels", type=int, default=32,
                        help="Latent channels per modality encoder")
    parser.add_argument("--base_channels", type=int, default=16,
                        help="Base channel count for encoder")
    parser.add_argument("--decoder_depth", type=int, default=4,
                        help="Decoder depth (4 for full upsampling)")
    parser.add_argument("--dropout_rate", type=float, default=0.2,
                        help="Modality dropout rate during training")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of disease classes (4: NC, SCD, MCI, AD)")

    # Encoder: multi-axis 3D attention (from NeuroQuant, CVPR Findings 2026)
    parser.add_argument("--use_attention", action="store_true", default=True,
                        help="Insert multi-axis 3D attention into encoder (default ON)")
    parser.add_argument("--no_attention", action="store_true", default=False,
                        help="Disable multi-axis 3D attention (revert to plain ResNet)")
    parser.add_argument("--attention_levels", type=str, default="3",
                        help="Comma-separated 0-indexed stage numbers for attention, e.g. '3' or '2,3'")
    parser.add_argument("--attention_heads", type=int, default=8,
                        help="Number of heads per axial attention block (auto-reduced if it doesn't divide channels)")

    # fMRI temporal encoder (preserves BOLD time series instead of static mean).
    # When enabled, dataset returns 5D fMRI [B, 1, D, H, W, T] and the model
    # uses fMRITemporalEncoder (1D conv + Transformer) instead of the 3D CNN.
    parser.add_argument("--use_fmri_temporal", action="store_true", default=True,
                        help="Use fMRITemporalEncoder (preserves BOLD time series). Default ON.")
    parser.add_argument("--no_fmri_temporal", action="store_true", default=False,
                        help="Use static 3D CNN for fMRI (legacy time-averaged path).")
    parser.add_argument("--use_fmri_deep", action="store_true", default=True,
                        help="Use fMRIDeepEncoder (deep multi-scale + FC). RECOMMENDED. Default ON.")
    parser.add_argument("--no_fmri_deep", action="store_true", default=False,
                        help="Disable deep fMRI encoder (fall back to fMRITemporalEncoder or static).")
    parser.add_argument("--fmri_deep_n_soft_roi", type=int, default=32,
                        help="Number of learned soft-ROI factors for fMRI deep encoder.")
    parser.add_argument("--fmri_deep_n_transformer_layers", type=int, default=3,
                        help="Number of TransformerEncoder layers in fMRI deep encoder.")
    parser.add_argument("--fmri_deep_n_heads", type=int, default=4,
                        help="Number of attention heads in fMRI deep encoder Transformer.")
    parser.add_argument("--fmri_deep_fc_compression", type=int, default=32,
                        help="Output dim of functional connectivity head in fMRI deep encoder.")
    parser.add_argument("--use_t1_centric_fusion", action="store_true", default=True,
                        help="Use T1-centric fusion (T1 trunk + gated aux deltas). "
                             "Guarantees T1+aux >= T1-only. RECOMMENDED.")
    parser.add_argument("--no_t1_centric_fusion", action="store_true", default=False,
                        help="Fall back to legacy concat fusion (for ablation).")
    parser.add_argument("--fmri_in_channels", type=int, default=34,
                        help="Spatial channels for fMRI temporal encoder (W axis of (D,H,W) fMRI).")
    parser.add_argument("--fmri_hidden_dim", type=int, default=128,
                        help="Hidden dim of fMRI 1D conv stack.")
    parser.add_argument("--fmri_num_pool", type=int, default=3,
                        help="Number of 1D conv blocks (each halves T).")
    parser.add_argument("--fmri_num_transformer_layers", type=int, default=2,
                        help="TransformerEncoder depth for fMRI temporal modeling.")
    parser.add_argument("--fmri_num_heads", type=int, default=4,
                        help="Multi-head attention heads in the fMRI transformer.")
    parser.add_argument("--fmri_t_target", type=int, default=200,
                        help="Number of BOLD timepoints to normalize fMRI to. "
                             "T>target → middle segment (training: random); "
                             "T<target → zero-pad at end.")

    # OOM fix: gradient checkpointing on the decoder (saves ~40% peak
    # memory, costs ~15% wall time). Default ON given the 24GB GPU
    # budget; set --no_checkpointing to disable.
    parser.add_argument("--use_checkpointing", action="store_true", default=True,
                        help="Wrap the decoder in torch.utils.checkpoint.sequential (default ON).")
    parser.add_argument("--no_checkpointing", action="store_true", default=False,
                        help="Disable decoder gradient checkpointing.")

    # Training
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accumulation_steps)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--cls_weight", type=float, default=2.0,
                        help="Classification loss weight (higher = more discriminative latent)")
    parser.add_argument("--kl_weight", type=float, default=0.1,
                        help="KL divergence loss weight")
    parser.add_argument("--kl_warmup_epochs", type=int, default=20,
                        help="Epochs for KL weight annealing from 0 to kl_weight")
    parser.add_argument("--free_bits", type=float, default=0.0,
                        help="Free bits per latent dimension (minimum KL, prevents collapse)")
    parser.add_argument("--recon_loss_type", type=str, default="l1",
                        help="Reconstruction loss type (l1 or l2)")
    parser.add_argument("--contrastive_weight", type=float, default=0.0,
                        help="Ordinal contrastive loss weight (0=disabled, try 0.05)")
    parser.add_argument("--gradient_weight", type=float, default=0.0,
                        help="Gradient/texture loss weight (0=disabled, try 0.1)")
    parser.add_argument("--ssim_weight", type=float, default=0.0,
                        help="SSIM loss weight (0=disabled, try 0.1)")
    parser.add_argument("--encoder_grad_boost", type=float, default=1.0,
                        help="Scale factor for encoder gradients (default 1.0; with fMRI fix, no boost needed)")
    parser.add_argument("--ordinal_reg_weight", type=float, default=0.1,
                        help="Weight for ordinal regression loss on latent mean")

    # Hardware
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of GPUs for DataParallel (default 2; canonical setup is 2x RTX 3090)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Checkpoint save interval")
    parser.add_argument("--early_stopping", type=int, default=100,
                        help="Early stopping patience")

    # AMP
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--no_amp", action="store_true", default=False,
                        help="Disable AMP")
    parser.add_argument("--no_precomputed", action="store_true", default=False,
                        help="Disable precomputed cache, use on-the-fly transforms")

    # v10: KL schedule strategy and cyclical-KL params
    parser.add_argument("--kl_strategy", type=str, default="linear",
                        choices=["linear", "cyclical"],
                        help="KL weight schedule: 'linear' (warmup) or 'cyclical' (0<->peak cycles)")
    parser.add_argument("--kl_cycle_len", type=int, default=15,
                        help="Epochs per cycle when kl_strategy=cyclical")
    parser.add_argument("--kl_cycle_low_frac", type=float, default=0.1,
                        help="Min KL weight (fraction of target) during cyclical off-phase")

    # v10: latent-space mixup
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="Latent mixup Beta(alpha, alpha) parameter; 0 disables")
    parser.add_argument("--mixup_prob", type=float, default=0.5,
                        help="Per-batch probability of applying mixup")

    # Apply YAML config defaults AFTER all add_argument calls
    # (set_defaults must come last so it isn't overridden by argparse defaults)
    parser.set_defaults(**config_defaults)

    return parser.parse_args()


def load_data(json_path: str, num_classes: int = 4) -> list:
    """Load and validate multi-modal dataset."""
    import nibabel as nib
    from monai.transforms import LoadImaged, EnsureChannelFirstd, Orientationd, CropForegroundd, Spacingd, ScaleIntensityRangePercentilesd, ResizeWithPadOrCropd, Compose

    with open(json_path, "r") as f:
        data = json.load(f)

    # Quick validation transforms (skip expensive CropForegroundd/Spacingd for speed)
    quick_transforms = Compose([
        LoadImaged(keys=['t1'], reader='NibabelReader'),
        EnsureChannelFirstd(keys=['t1']),
        Orientationd(keys=['t1'], axcodes='RAS'),
    ])

    valid_data = []
    corrupted_t1 = 0
    for item in data:
        # T1 is required
        t1_path = item.get("t1")
        if not t1_path or not os.path.exists(t1_path):
            continue

        # Validate T1 file dimensions and data content (catch corrupted [0,0,0] files and all-zero files)
        try:
            img = nib.load(t1_path)
            shape = img.shape
            if any(s == 0 for s in shape):
                corrupted_t1 += 1
                continue
            # Check data is not all zeros (CropForegroundd would produce empty output)
            img_data = img.get_fdata()
            if img_data.min() == img_data.max():  # All zeros or constant value
                corrupted_t1 += 1
                continue
            # Quick transform check (loadable by MONAI)
            data_dict = {'t1': str(t1_path)}
            quick_transforms(data_dict)
        except Exception:
            corrupted_t1 += 1
            continue

        # Check at least one optional modality exists
        has_optional = False
        for mod in ["fmri", "asl", "qsm", "flair"]:
            path = item.get(mod)
            if path and os.path.exists(path):
                has_optional = True
                break

        # For now, accept samples with T1 even if no optional modalities
        valid_data.append(item)

    # Conditionally remap labels: only when num_classes=3 (merge SCD+MCI)
    # When num_classes=4, keep all 4 stages: NC=0, SCD=1, MCI=2, AD=3
    if num_classes == 3:
        for item in valid_data:
            label = item.get("label", 0)
            if label in [1, 2]:  # SCD or MCI -> merged class
                item["label"] = 1
            elif label == 3:  # AD -> class 2
                item["label"] = 2
            # NC (0) stays 0
        print("Remapped labels to 3-class (NC / SCD+MCI / AD)")
    else:
        print(f"Keeping labels as {num_classes}-class (NC / SCD / MCI / AD)")

    if corrupted_t1 > 0:
        print(f"Warning: Skipped {corrupted_t1} corrupted T1 files")
    print(f"Loaded {len(valid_data)} valid multi-modal samples")

    # Print class distribution
    from collections import Counter
    label_counts = Counter(item.get("label", 0) for item in valid_data)
    if num_classes == 3:
        class_names = ["NC", "SCD+MCI", "AD"]
    else:
        class_names = ["NC", "SCD", "MCI", "AD"]
    for c in range(num_classes):
        print(f"  {class_names[c]}: {label_counts.get(c, 0)}")

    return valid_data


def main():
    args = parse_args()

    # Handle AMP flag
    use_amp = args.use_amp and not args.no_amp

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_list = load_data(args.json, num_classes=args.num_classes)
    print(f"Total samples: {len(data_list)}")

    # Transforms
    train_transforms = get_multimodal_train_transforms()
    val_transforms = get_multimodal_val_transforms()

    # Split data
    # Store manifest index in each sample so precomputed cache can look up correctly
    for i, item in enumerate(data_list):
        item["_manifest_idx"] = i

    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(
        data_list, test_size=0.15, stratify=[d.get("label", 0) for d in data_list], random_state=42
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Datasets
    # Resolve which optional modalities are active so the dataset doesn't bother
    # trying to load files for modalities the model has no encoder for.
    optional_modalities = _resolve_optional_modalities(args)
    use_demographic = bool(getattr(args, "use_demographic", True)) and not bool(
        getattr(args, "no_demographic", False)
    )
    print(f"[Modality switches] optional={optional_modalities}  "
          f"demographic={use_demographic}  t1_only={bool(getattr(args, 't1_only', False))}")
    # Per-modality target sizes from MULTI_MODAL_SPATIAL_SIZES (one entry
    # per modality; the dataset enforces these in _resize_spatial_3d for
    # 3D modalities and _normalize_fmri_t for 4D fMRI). fMRI's T=200
    # is read from args.fmri_t_target with a 200 default.
    spatial_sizes = dict(MULTI_MODAL_SPATIAL_SIZES)
    print(f"[Per-modality target sizes] {spatial_sizes}")
    fmri_t_target = int(getattr(args, "fmri_t_target", 200))
    print(f"[fMRI T target] {fmri_t_target}")
    # Use npy cache on C: for ~5-10x faster data loading (pre-built from .nii.gz)
    _npy_cache = "C:/ADynamics_npy_cache" if os.path.isdir("C:/ADynamics_npy_cache") else None
    if _npy_cache:
        print(f"[Data] Using npy cache: {_npy_cache}")

    # Precomputed cache: use chunked loading to avoid OOM
    # Each chunk is ~2GB, loaded on demand with LRU cache
    _precomputed_path = getattr(args, 'precomputed_cache', None)
    if getattr(args, 'no_precomputed', False):
        _precomputed_path = None
        print("[Data] Precomputed cache disabled by --no_precomputed")
    elif _precomputed_path is None and _npy_cache:
        _chunked = os.path.join(_npy_cache, "precomputed")
        if os.path.isdir(_chunked) and os.path.exists(os.path.join(_chunked, "index.json")):
            _precomputed_path = _chunked

    train_dataset = MultiModalDataset(
        train_data,
        transform=train_transforms,
        optional_modalities=optional_modalities,
        spatial_sizes=spatial_sizes,
        fmri_t_target=fmri_t_target,
        npy_cache_dir=_npy_cache,
        precomputed_path=_precomputed_path,
    )
    val_dataset = MultiModalDataset(
        val_data,
        transform=val_transforms,
        optional_modalities=optional_modalities,
        spatial_sizes=spatial_sizes,
        fmri_t_target=fmri_t_target,
        npy_cache_dir=_npy_cache,
        precomputed_path=_precomputed_path,
    )

    # Dataloaders: num_workers=0 on Windows (multi-worker + DataParallel crashes).
    from core_data.dataset import multimodal_collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate_fn,
        drop_last=True,
    )

    # Model
    # Parse attention_levels: accept either CLI comma-separated string ("2,3")
    # or YAML list ([2, 3]) or single int / str.
    use_attention = args.use_attention and not args.no_attention
    use_fmri_temporal = args.use_fmri_temporal and not args.no_fmri_temporal
    al_raw = args.attention_levels
    if isinstance(al_raw, (list, tuple)):
        attn_levels = tuple(int(x) for x in al_raw)
    else:
        try:
            attn_levels = tuple(int(x.strip()) for x in str(al_raw).split(",") if str(x).strip())
        except (ValueError, AttributeError):
            raise ValueError(f"--attention_levels must be comma-separated ints, got {al_raw!r}")
    print(f"[Encoder] use_attention={use_attention}  attention_levels={attn_levels}  attention_heads={args.attention_heads}")
    model = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        decoder_depth=args.decoder_depth,
        optional_modalities=optional_modalities,   # ← was hardcoded; now driven by switches
        use_attention=use_attention,
        attention_levels=attn_levels,
        attention_heads=args.attention_heads,
        use_fmri_temporal=use_fmri_temporal,
        use_fmri_deep=(args.use_fmri_deep and not args.no_fmri_deep),
        fmri_in_channels=args.fmri_in_channels,
        fmri_t_target=fmri_t_target,
        fmri_hidden_dim=args.fmri_hidden_dim,
        fmri_num_pool=args.fmri_num_pool,
        fmri_num_transformer_layers=args.fmri_num_transformer_layers,
        fmri_num_heads=args.fmri_num_heads,
        fmri_deep_n_soft_roi=args.fmri_deep_n_soft_roi,
        fmri_deep_n_transformer_layers=args.fmri_deep_n_transformer_layers,
        fmri_deep_n_heads=args.fmri_deep_n_heads,
        fmri_deep_fc_compression=args.fmri_deep_fc_compression,
        use_t1_centric_fusion=(args.use_t1_centric_fusion and not args.no_t1_centric_fusion),
        use_demographic_cond=use_demographic,
        # OOM fix: gradient checkpointing on the decoder. 256^3 decoder
        # activations consume ~16GB of autograd-graph memory; checkpoint
        # drops peak by ~40% at the cost of ~15% slower training. Set
        # this to False if you have more VRAM than peak and need speed.
        use_checkpointing=bool(getattr(args, "use_checkpointing", True)),
    )

    # Multi-GPU support via shared utils (replaces buggy local DataParallel)
    print(f"[DEBUG] args.num_gpus = {args.num_gpus}, cuda.device_count = {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    model = setup_data_parallel(model, args.num_gpus)

    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Config
    config = {
        "accumulation_steps": args.accumulation_steps,
        "cls_weight": args.cls_weight,
        "kl_weight": args.kl_weight,
        "kl_warmup_epochs": args.kl_warmup_epochs,
        "free_bits": args.free_bits,
        "recon_loss_type": args.recon_loss_type,
        "contrastive_weight": args.contrastive_weight,
        "gradient_weight": args.gradient_weight,
        "ssim_weight": args.ssim_weight,
        "encoder_grad_boost": args.encoder_grad_boost,
        "ordinal_reg_weight": args.ordinal_reg_weight,
        "num_classes": args.num_classes,
        "use_amp": use_amp,
        "use_fmri_temporal": use_fmri_temporal,
        "use_demographic_cond": use_demographic,
        "optional_modalities": optional_modalities,
        # v10: cyclical KL schedule (was missing — trainer fell back to "linear")
        "kl_strategy": getattr(args, "kl_strategy", "linear"),
        "kl_cycle_len": getattr(args, "kl_cycle_len", 15),
        "kl_cycle_low_frac": getattr(args, "kl_cycle_low_frac", 0.1),
        # v10: latent mixup (was missing — trainer fell back to mixup_alpha=0.0)
        "mixup_alpha": getattr(args, "mixup_alpha", 0.0),
        "mixup_prob": getattr(args, "mixup_prob", 0.5),
        # Class names for diagnostic output
        "class_names": ["NC", "SCD", "MCI", "AD"],
    }

    # Trainer
    trainer = MultiModalVAETrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        scheduler=scheduler,
    )

    # Resume from checkpoint
    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = checkpoint["model_state_dict"]

        # Handle DataParallel prefix
        model_ref = model.module if hasattr(model, "module") else model
        model_sd = model_ref.state_dict()
        has_dp = any(k.startswith("module.") for k in sd)

        if has_dp:
            sd = {k[7:]: v for k, v in sd.items()}

        # Filter: only load keys that exist and shape matches
        filtered_sd = {}
        skipped = []
        for k, v in sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                filtered_sd[k] = v
            else:
                skipped.append(k)

        # Load into underlying model (bypass DataParallel)
        model_ref.load_state_dict(filtered_sd, strict=False)
        print(f"  Loaded {len(filtered_sd)} params, skipped {len(skipped)}")
        if skipped:
            print(f"  Skipped (shape mismatch): {skipped}")

        trainer.current_epoch = checkpoint.get("epoch", 0)
        trainer.best_val_loss = float("inf")
        trainer.best_cls_acc = 0.0

        # Optimizer state: always skip when resuming with different num_classes
        # (old optimizer has stale buffers for classifier head that waste GPU memory)
        if skipped:
            print("  Optimizer state skipped (classifier changed, starting fresh)")
        elif "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("  Optimizer state restored")
            except Exception as e:
                print(f"  Optimizer state incompatible, starting fresh: {e}")

        # Free checkpoint from memory
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Train
    print(f"\n{'='*60}")
    print("Starting Multi-Modal VAE Training")
    print(f"{'='*60}")
    print(f"Output directory: {args.output_dir}")
    print(f"Latent channels: {args.latent_channels}")
    print(f"Base channels: {args.base_channels}")
    print(f"Classification weight: {args.cls_weight}")
    print(f"Modality dropout rate: {args.dropout_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Clear CUDA cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

    try:
        history = trainer.train(
            num_epochs=args.epochs,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
            early_stopping_patience=args.early_stopping,
        )

        print("\nTraining complete!")
        print(f"Best val_cls_acc: {trainer.best_cls_acc:.4f}")
        print(f"Checkpoints saved to: {args.output_dir}")
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"TRAINING FAILED: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
