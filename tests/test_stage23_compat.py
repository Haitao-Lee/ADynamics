"""
Smoke test for utils/stage23_compat.py.

Validates the two helpers shared by Stage 2-5 trainers:
  1. normalize_fmri_batch: 5D/6D fMRI shape handling.
  2. maybe_strip_module_prefix: DataParallel 'module.' prefix handling.
  3. shape_filtered_load_state_dict: ckpt load with shape-aware filtering.

Run: python tests/test_stage23_compat.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from utils.stage23_compat import (
    normalize_fmri_batch,
    maybe_strip_module_prefix,
    shape_filtered_load_state_dict,
)


# ---------------------------------------------------------------------------
def test_normalize_fmri_batch_5d_to_6d() -> None:
    """Legacy 3D-like fMRI [B,1,D,H,W] -> 6D [B,1,D,H,W,1]."""
    fmri5 = torch.randn(2, 1, 64, 64, 34)
    fmri6 = normalize_fmri_batch(fmri5)
    assert fmri6.shape == (2, 1, 64, 64, 34, 1), f"got {fmri6.shape}"
    # Round-trip: 6D in -> 6D out
    fmri6b = normalize_fmri_batch(fmri6)
    assert fmri6b.shape == (2, 1, 64, 64, 34, 1)
    print("  [OK] normalize 5D -> 6D and round-trip")


def test_normalize_fmri_batch_6d_preserved() -> None:
    """Plan C 6D fMRI [B,1,D,H,W,T] is a no-op."""
    fmri = torch.randn(2, 1, 64, 64, 34, 220)
    out = normalize_fmri_batch(fmri)
    assert out.shape == (2, 1, 64, 64, 34, 220)
    assert torch.equal(out, fmri)
    print("  [OK] normalize 6D is no-op")


def test_normalize_fmri_batch_3d_error() -> None:
    """Rank < 5 or 7 > 6 should raise."""
    try:
        normalize_fmri_batch(torch.randn(2, 1, 64, 64))
        assert False, "should have raised"
    except ValueError:
        pass
    print("  [OK] normalize raises on rank 4")


# ---------------------------------------------------------------------------
def test_maybe_strip_module_prefix_with_prefix() -> None:
    """Strips 'module.' prefix when present."""
    sd = {"module.encoder.weight": torch.zeros(3), "module.bias": torch.zeros(3)}
    out = maybe_strip_module_prefix(sd)
    assert "module.encoder.weight" not in out
    assert "encoder.weight" in out
    assert "bias" in out
    print("  [OK] strip prefix when present")


def test_maybe_strip_module_prefix_no_prefix() -> None:
    """No-op when no prefix."""
    sd = {"encoder.weight": torch.zeros(3), "bias": torch.zeros(3)}
    out = maybe_strip_module_prefix(sd)
    assert out is sd or out == sd
    assert "encoder.weight" in out
    print("  [OK] no-op when no prefix")


# ---------------------------------------------------------------------------
class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.heads = nn.ModuleDict({"a": nn.Linear(4, 2)})


def test_shape_filtered_load_drops_mismatched() -> None:
    """Keys with wrong shape should be dropped silently."""
    model = _TinyModel()
    sd = {
        "fc.weight": torch.zeros(4, 4),  # OK
        "fc.bias": torch.zeros(99),       # shape mismatch
        "heads.a.weight": torch.zeros(2, 4),  # OK
        "missing_key": torch.zeros(1),    # not in model
    }
    n_loaded, n_skipped, n_missing = shape_filtered_load_state_dict(
        model, sd, strict=False, verbose=False,
    )
    assert n_loaded == 2, f"loaded {n_loaded}"
    # skipped = shape mismatch (1) + key not in model (1)
    assert n_skipped == 2, f"skipped {n_skipped}"
    # missing = fc.bias + heads.a.bias are not in sd -> 2
    assert n_missing == 2, f"missing {n_missing}"
    print(f"  [OK] shape_filtered: loaded=2 skipped=2 missing=2")


def test_shape_filtered_load_strips_module() -> None:
    """DataParallel-prefixed keys should be loaded into non-prefixed model."""
    model = _TinyModel()
    sd = {
        "module.fc.weight": torch.zeros(4, 4),
        "module.fc.bias": torch.zeros(4),
        "module.heads.a.weight": torch.zeros(2, 4),
    }
    n_loaded, _, _ = shape_filtered_load_state_dict(
        model, sd, strict=False, verbose=False,
    )
    assert n_loaded == 3, f"loaded {n_loaded}"
    print(f"  [OK] shape_filtered strips module. prefix")


def test_shape_filtered_load_no_match() -> None:
    """All-mismatched sd loads nothing."""
    model = _TinyModel()
    sd = {"fc.weight": torch.zeros(99, 99), "fc.bias": torch.zeros(99)}
    n_loaded, n_skipped, n_missing = shape_filtered_load_state_dict(
        model, sd, strict=False, verbose=False,
    )
    assert n_loaded == 0
    assert n_skipped == 2
    # model has 4 keys (fc.weight/bias, heads.a.weight/bias), none in loaded -> 4 missing
    assert n_missing == 4
    print(f"  [OK] all-mismatched: loaded=0 skipped=2 missing=4")


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("utils/stage23_compat.py smoke tests")
    print("=" * 60)
    tests = [
        ("normalize 5D->6D", test_normalize_fmri_batch_5d_to_6d),
        ("normalize 6D preserved", test_normalize_fmri_batch_6d_preserved),
        ("normalize rank error", test_normalize_fmri_batch_3d_error),
        ("strip prefix", test_maybe_strip_module_prefix_with_prefix),
        ("no-strip no-op", test_maybe_strip_module_prefix_no_prefix),
        ("shape_filtered drops mismatched", test_shape_filtered_load_drops_mismatched),
        ("shape_filtered strips module.", test_shape_filtered_load_strips_module),
        ("shape_filtered no match", test_shape_filtered_load_no_match),
    ]
    for name, fn in tests:
        print(f"\n[Test] {name}")
        fn()
    print("\n" + "=" * 60)
    print(f"All {len(tests)} tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
