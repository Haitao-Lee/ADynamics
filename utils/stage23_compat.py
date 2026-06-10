"""
Shared compatibility helpers for Stage 2-5 trainers.

These helpers close the gap between Stage 1's recent upgrades (Plan A
fMRI 4D fix + Plan C fMRITemporalEncoder) and the downstream stages that
load Stage 1 checkpoints and consume its outputs.

Functions:
    normalize_fmri_batch:  5D/6D fMRI shape normalization, mirroring
        the Stage 1 trainer's _normalize_fmri_batch. When the dataset
        returns 5D fMRI [B, 1, D, H, W, T] (Plan C preserve_temporal_dim
        path), this is a no-op. When it returns legacy 3D [B, 1, D, H, W],
        unsqueezes the trailing time dim to 1.
    shape_filtered_load_state_dict:  Load a checkpoint, keeping only the
        keys that match the current model in both name AND shape. Avoids
        noise warnings when Stage 1's optional_encoders contain encoders
        (e.g. fMRITemporalEncoder) that downstream stages don't build.
    maybe_strip_module_prefix:  Strip the "module." prefix from
        DataParallel-wrapped checkpoints. Idempotent.

Why a single utility module
---------------------------
Stage 2a/2b/3/4 all need the same two helpers. Duplicating them in
4 places risks drift. Keeping them here means a single fix here
propagates to all stages.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor


def normalize_fmri_batch(fmri: Tensor) -> Tensor:
    """
    Normalize fMRI batch to a consistent 6D shape for the temporal encoder.

    Args:
        fmri: A tensor of shape
            - 5D [B, 1, D, H, W]    : legacy time-averaged path
            - 6D [B, 1, D, H, W, T] : Plan C preserve_temporal_dim=True path

    Returns:
        6D tensor [B, 1, D, H, W, T_target]. If 5D, T_target=1 (legacy becomes
        a single timepoint, equivalent to the time-mean image).
    """
    if fmri.dim() == 5:
        return fmri.unsqueeze(-1)
    if fmri.dim() == 6:
        return fmri
    raise ValueError(
        f"fMRI must be 5D or 6D; got {fmri.dim()}D with shape {tuple(fmri.shape)}"
    )


def maybe_strip_module_prefix(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Strip a leading 'module.' prefix from all keys in a state dict.

    This is needed when loading a checkpoint saved under DataParallel
    (which prefixes every key with 'module.') into a non-wrapped model,
    or vice versa. The check is symmetric: if there are NO 'module.'
    keys in the checkpoint, the input is returned unchanged.

    Args:
        state_dict: Model state_dict to normalize.

    Returns:
        New dict with consistent prefix (or original if no prefix found).
    """
    if not state_dict:
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict)
    if not has_module:
        return state_dict
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def shape_filtered_load_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, Tensor],
    strict: bool = False,
    verbose: bool = True,
) -> tuple:
    """
    Load a state dict into a model, filtering out keys with shape mismatches.

    Use this instead of model.load_state_dict() when loading a Stage 1
    checkpoint into a downstream model that may have slightly different
    submodules (e.g. the fMRITemporalEncoder is present in Stage 1 but
    not in a Stage 2a classifier head). Strict=False makes the load
    non-throwing, but shape mismatches still produce noisy UserWarnings
    that obscure real bugs.

    This helper:
        1. Strips DataParallel 'module.' prefix if needed.
        2. Compares each tensor shape against the model's state_dict.
        3. Loads only keys that match in BOTH name and shape.
        4. Returns (n_loaded, n_skipped, n_missing) for reporting.

    Args:
        model: Target model (already constructed with the right modules).
        state_dict: Source state_dict (may be a checkpoint['model_state_dict']).
        strict: If True, raise on any unhandled mismatch. Default False.
        verbose: If True, print a one-line summary. Default True.

    Returns:
        Tuple of (n_loaded, n_skipped, n_missing).
    """
    sd = maybe_strip_module_prefix(state_dict)
    model_sd = model.state_dict()
    has_module = any(k.startswith("module.") for k in model_sd)
    if has_module:
        # Model is DataParallel-wrapped. Re-prefix the sd before comparing.
        sd_to_check = {f"module.{k}": v for k, v in sd.items()}
        target_sd = model_sd
    else:
        sd_to_check = sd
        target_sd = model_sd

    filtered = {}
    n_skipped_shape = 0
    for k, v in sd_to_check.items():
        if k in target_sd and target_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            n_skipped_shape += 1

    # Track which expected keys are still missing
    loaded_keys = set(filtered.keys())
    n_missing = sum(1 for k in target_sd if k not in loaded_keys)

    model.load_state_dict(filtered, strict=strict)

    n_loaded = len(filtered)
    if verbose:
        print(
            f"[stage23_compat] Loaded {n_loaded} tensors, "
            f"skipped {n_skipped_shape} (shape mismatch or absent in target), "
            f"{n_missing} target keys left at default."
        )
    return n_loaded, n_skipped_shape, n_missing


# ---------------------------------------------------------------------------
# Modality-toggle helpers (used by Stage 1-5 scripts for consistency)
# ---------------------------------------------------------------------------
def resolve_optional_modalities(args) -> list:
    """
    Compute the final list of optional modalities from CLI / YAML flags.

    This is the canonical implementation shared by Stage 1, 2, 3, 4, 5.
    Rules (in priority order):
      1. If --t1_only (CLI) or t1_only: true (YAML): return [] (T1-only).
      2. Otherwise, return the subset of {fmri, asl, qsm, flair} whose
         --use_X is True (and not --no_X).

    Args:
        args: parsed argparse namespace (with use_fmri / use_asl / use_qsm /
              use_flair / t1_only / no_fmri / no_asl / no_qsm / no_flair attrs).

    Returns:
        list of optional modality names, in the canonical order.
    """
    t1_only = bool(getattr(args, "t1_only", False))
    if t1_only:
        return []

    toggles = {
        "fmri":  (bool(getattr(args, "use_fmri", True))
                  and not bool(getattr(args, "no_fmri", False))),
        "asl":   (bool(getattr(args, "use_asl", True))
                  and not bool(getattr(args, "no_asl", False))),
        "qsm":   (bool(getattr(args, "use_qsm", True))
                  and not bool(getattr(args, "no_qsm", False))),
        "flair": (bool(getattr(args, "use_flair", True))
                  and not bool(getattr(args, "no_flair", False))),
    }
    return [m for m, on in toggles.items() if on]


def resolve_use_demographic(args) -> bool:
    """
    Resolve use_demographic flag from CLI / YAML.

    Returns True if demographic conditioning is on (default).
    """
    return bool(getattr(args, "use_demographic", True)) and not bool(
        getattr(args, "no_demographic", False)
    )


# ---------------------------------------------------------------------------
# CLI helper: add modality-toggle flags to any argparse parser
# ---------------------------------------------------------------------------
def add_modality_args(parser) -> None:
    """
    Add the 5 modality-toggle + demographic flags to an argparse parser.

    Use this in every stage script (1-5) so the CLI is consistent and a
    T1-only Stage 1 checkpoint loads cleanly into Stage 2/3/4/5.
    """
    parser.add_argument("--use_fmri", action="store_true", default=True,
                        help="Use fMRI modality (default ON)")
    parser.add_argument("--no_fmri", action="store_true", default=False,
                        help="Disable fMRI modality")
    parser.add_argument("--use_asl", action="store_true", default=True,
                        help="Use ASL modality (default ON)")
    parser.add_argument("--no_asl", action="store_true", default=False,
                        help="Disable ASL modality")
    parser.add_argument("--use_qsm", action="store_true", default=True,
                        help="Use QSM modality (default ON)")
    parser.add_argument("--no_qsm", action="store_true", default=False,
                        help="Disable QSM modality")
    parser.add_argument("--use_flair", action="store_true", default=True,
                        help="Use FLAIR modality (default ON)")
    parser.add_argument("--no_flair", action="store_true", default=False,
                        help="Disable FLAIR modality")
    parser.add_argument("--t1_only", action="store_true", default=False,
                        help="T1-only mode: disable all 4 optional modalities")
    parser.add_argument("--use_demographic", action="store_true", default=True,
                        help="Add age + sex conditioning to latent (default ON)")
    parser.add_argument("--no_demographic", action="store_true", default=False,
                        help="Disable demographic conditioning")
