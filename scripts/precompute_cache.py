"""precompute_cache.py
Pre-compute transforms for every sample. Uses numpy only (no MONAI).
Processes one sample at a time to avoid memory issues.

Usage:
    python scripts/precompute_cache.py
    python scripts/precompute_cache.py --cache_dir E:/LHT_workspace/AD/ADynamics/npy_cache

Output:
    <cache_dir>/precomputed/chunk_XXXXX.pt  (one file per 50 samples)
    <cache_dir>/precomputed/index.json      (manifest of all chunks)
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom as scipy_zoom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core_data.transforms import MULTI_MODAL_SPATIAL_SIZES


def load_npy(path: str, cache_dir: str) -> np.ndarray:
    """Load from npy cache."""
    norm = os.path.normpath(path).replace(os.sep, "/")
    h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
    cp = os.path.join(cache_dir, h + ".npy")
    if os.path.exists(cp):
        return np.load(cp).astype(np.float32)
    import nibabel as nib
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


def resize_3d(arr: np.ndarray, target: tuple) -> np.ndarray:
    if arr.shape == target:
        return arr
    factors = [t / s for t, s in zip(target, arr.shape)]
    return scipy_zoom(arr, factors, order=1, mode='constant', cval=0.0).astype(np.float32)


def normalize_intensity(arr: np.ndarray) -> np.ndarray:
    mask = arr > 0
    if mask.sum() == 0:
        return arr
    m, s = arr[mask].mean(), arr[mask].std()
    if s < 1e-8:
        return arr - m
    arr = arr.copy()
    arr[mask] = (arr[mask] - m) / s
    return arr


def process_sample(item: dict, spatial_sizes: dict, cache_dir: str, t_target: int = 200) -> dict:
    """Process one sample, return dict of tensors or None on error."""
    paths = item.get("paths", item)
    entry = {}

    # T1
    t1_path = paths.get("t1")
    if not t1_path or not os.path.exists(t1_path):
        return None
    t1 = load_npy(t1_path, cache_dir)
    t1 = resize_3d(t1, spatial_sizes["t1"])
    t1 = normalize_intensity(t1)
    entry["t1"] = torch.from_numpy(t1).unsqueeze(0).half()
    del t1

    # Optional modalities
    available = []
    for mod in ["fmri", "asl", "qsm", "flair"]:
        p = paths.get(mod)
        if not p or not os.path.exists(p):
            entry[mod] = None
            continue
        try:
            arr = load_npy(p, cache_dir)
            if mod == "fmri" and arr.ndim == 4:
                d, h, w = spatial_sizes["fmri"]
                # Process fMRI: resize each timepoint, pad/trim time
                T = arr.shape[3]
                T_out = min(T, t_target)
                resized = np.zeros((d, h, w, T_out), dtype=np.float32)
                for t in range(T_out):
                    resized[:, :, :, t] = resize_3d(arr[:, :, :, t], (d, h, w))
                del arr
                # Z-score each volume
                for t in range(resized.shape[3]):
                    vol = resized[:, :, :, t]
                    mask = vol > 0
                    if mask.sum() > 0:
                        m, s = vol[mask].mean(), vol[mask].std()
                        if s > 1e-8:
                            resized[:, :, :, t][mask] = (vol[mask] - m) / s
                # Pad time if needed
                if T < t_target:
                    pad = np.zeros((d, h, w, t_target - T), dtype=np.float32)
                    resized = np.concatenate([resized, pad], axis=3)
                entry[mod] = torch.from_numpy(resized).half()
                del resized
                available.append(mod)
            elif arr.ndim == 3:
                arr = resize_3d(arr, spatial_sizes[mod])
                arr = normalize_intensity(arr)
                entry[mod] = torch.from_numpy(arr).unsqueeze(0).half()
                del arr
                available.append(mod)
            else:
                entry[mod] = None
        except Exception:
            entry[mod] = None

    entry["label"] = item.get("label", 0)
    entry["patient_id"] = item.get("patient_id", "")
    entry["available_modalities"] = available

    # Demographics
    raw_age = item.get("age", item.get("paths", {}).get("age"))
    entry["age"] = float(raw_age) if raw_age not in (None, "") else None
    raw_sex = item.get("sex", item.get("paths", {}).get("sex"))
    entry["sex"] = int(raw_sex) if raw_sex not in (None, "") else None

    return entry


def main():
    parser = argparse.ArgumentParser(description="Precompute multi-modal dataset cache")
    parser.add_argument("--cache_dir", type=str, default="./npy_cache",
                        help="Cache directory (default: ./npy_cache)")
    parser.add_argument("--manifest", type=str, default="./core_data/dataset_manifest_merged_v2.json",
                        help="Path to dataset manifest JSON")
    parser.add_argument("--fmri_t_target", type=int, default=60,
                        help="fMRI time dimension target")
    args = parser.parse_args()

    json_path = args.manifest
    cache_dir = args.cache_dir
    output_dir = os.path.join(cache_dir, "precomputed")
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path) as f:
        manifest = json.load(f)
    items = manifest if isinstance(manifest, list) else manifest.get("samples", list(manifest.values()))
    print(f"Manifest: {len(items)} samples")
    print(f"Cache dir: {cache_dir}")

    spatial_sizes = dict(MULTI_MODAL_SPATIAL_SIZES)
    print(f"Spatial sizes: {spatial_sizes}")

    t_target = args.fmri_t_target
    chunk_size = 50
    chunks = []
    t0 = time.time()
    ok = 0
    errors = 0

    for start in range(0, len(items), chunk_size):
        end = min(start + chunk_size, len(items))
        chunk_path = os.path.join(output_dir, f"chunk_{start:05d}.pt")

        # Skip if chunk already exists (resume support)
        if os.path.exists(chunk_path):
            existing = torch.load(chunk_path, map_location="cpu", weights_only=False)
            ok += len(existing)
            chunks.append(chunk_path)
            del existing
            continue

        chunk = {}
        for i in range(start, end):
            try:
                result = process_sample(items[i], spatial_sizes, cache_dir, t_target=t_target)
                if result is not None:
                    chunk[i] = result
                    ok += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  [WARN] sample {i}: {e}")

        # Save chunk
        torch.save(chunk, chunk_path, _use_new_zipfile_serialization=True)
        chunks.append(chunk_path)

        elapsed = time.time() - t0
        rate = (ok + errors) / elapsed
        remaining = len(items) - ok - errors
        eta = remaining / rate / 60 if rate > 0 else 0
        print(f"  [{ok+errors}/{len(items)}] ok={ok} err={errors} {rate:.1f}/s ETA {eta:.0f}min")

        del chunk
        gc.collect()

    # Save index
    index = {"chunks": chunks, "total": len(items), "ok": ok, "errors": errors,
             "spatial_sizes": {k: list(v) for k, v in spatial_sizes.items()}}
    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone in {time.time()-t0:.0f}s. {ok} OK, {errors} errors.")
    print(f"Chunks: {len(chunks)} files in {output_dir}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
