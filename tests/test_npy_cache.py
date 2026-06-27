"""Quick test that the new _load_npy_cached returns the same data as
the old nibabel path. Picks a sample that has T1 + fMRI + 2 optional
modalities, reads each via both paths, and asserts equality.

Also benchmarks: how fast is .npy vs .nii.gz on a warm cache?

Does NOT touch GPU, does NOT start training. Safe to run alongside
the prebuild workers.
"""
import os, sys, time
sys.path.insert(0, r'E:\LHT_workspace\AD\ADynamics')
os.chdir(r'E:\LHT_workspace\AD\ADynamics')

import numpy as np

# Pick a sample from the manifest
import json
with open(r'E:\LHT_workspace\AD\ADynamics\core_data\dataset_manifest_merged_v2.json') as f:
    manifest = json.load(f)
items = manifest if isinstance(manifest, list) else manifest.get('samples', list(manifest.values()))

# Find a sample with T1 + fMRI + ASL (so we test 4D + 3D)
for item in items[:200]:
    paths = item.get('paths', item)
    t1 = paths.get('t1')
    fmri = paths.get('fmri')
    asl = paths.get('asl')
    if t1 and fmri and asl and os.path.exists(t1) and os.path.exists(fmri) and os.path.exists(asl):
        # Pick this one
        break
print(f't1   : {t1}')
print(f'fmri : {fmri}')
print(f'asl  : {asl}')

# Method 1: nibabel (old)
import nibabel as nib

def load_nibabel(path):
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32)

# Method 2: .npy cache (new, via dataset helper)
from core_data.dataset import MultiModalDataset

# Build a tiny dummy dataset just to access _load_npy_cached
ds = MultiModalDataset(
    data_list=[{'t1': t1, 'fmri': fmri, 'asl': asl, 'label': 0, 'patient_id': 'test'}],
    transform=None,
    use_npy_cache=True,
)

print('\n--- Verifying T1 ---')
t_nib = load_nibabel(t1)
t_npy = ds._load_npy_cached(t1)
print(f'  nibabel shape={t_nib.shape} dtype={t_nib.dtype}')
print(f'  .npy    shape={t_npy.shape} dtype={t_npy.dtype}')
assert t_nib.shape == t_npy.shape, f'shape mismatch: {t_nib.shape} vs {t_npy.shape}'
diff = np.abs(t_nib - t_npy).max()
print(f'  max abs diff: {diff:.2e}  (should be ~0, tiny float rounding OK)')
assert diff < 1e-3, f'data mismatch: {diff}'
print('  PASS')

print('\n--- Verifying fMRI (4D) ---')
f_nib = load_nibabel(fmri)
f_npy = ds._load_npy_cached(fmri)
print(f'  nibabel shape={f_nib.shape} dtype={f_nib.dtype}')
print(f'  .npy    shape={f_npy.shape} dtype={f_npy.dtype}')
assert f_nib.shape == f_npy.shape
diff = np.abs(f_nib - f_npy).max()
print(f'  max abs diff: {diff:.2e}')
assert diff < 1e-3
print('  PASS')

print('\n--- Verifying ASL (3D) ---')
a_nib = load_nibabel(asl)
a_npy = ds._load_npy_cached(asl)
print(f'  nibabel shape={a_nib.shape} dtype={a_nib.dtype}')
print(f'  .npy    shape={a_npy.shape} dtype={a_npy.dtype}')
assert a_nib.shape == a_npy.shape
diff = np.abs(a_nib - a_npy).max()
print(f'  max abs diff: {diff:.2e}')
assert diff < 1e-3
print('  PASS')

# Benchmark: 10 reads of each
print('\n--- Benchmark (10 reads each) ---')
N = 10
t0 = time.time()
for _ in range(N):
    _ = load_nibabel(t1)
nib_t1_ms = (time.time() - t0) / N * 1000

t0 = time.time()
for _ in range(N):
    _ = ds._load_npy_cached(t1)
npy_t1_ms = (time.time() - t0) / N * 1000
print(f'  T1:    nibabel={nib_t1_ms:.0f} ms  vs  .npy={npy_t1_ms:.1f} ms  ({nib_t1_ms/max(npy_t1_ms,0.1):.1f}x speedup)')

t0 = time.time()
for _ in range(N):
    _ = load_nibabel(fmri)
nib_fmri_ms = (time.time() - t0) / N * 1000

t0 = time.time()
for _ in range(N):
    _ = ds._load_npy_cached(fmri)
npy_fmri_ms = (time.time() - t0) / N * 1000
print(f'  fMRI:  nibabel={nib_fmri_ms:.0f} ms  vs  .npy={npy_fmri_ms:.1f} ms  ({nib_fmri_ms/max(npy_fmri_ms,0.1):.1f}x speedup)')

t0 = time.time()
for _ in range(N):
    _ = load_nibabel(asl)
nib_asl_ms = (time.time() - t0) / N * 1000

t0 = time.time()
for _ in range(N):
    _ = ds._load_npy_cached(asl)
npy_asl_ms = (time.time() - t0) / N * 1000
print(f'  ASL:   nibabel={nib_asl_ms:.0f} ms  vs  .npy={npy_asl_ms:.1f} ms  ({nib_asl_ms/max(npy_asl_ms,0.1):.1f}x speedup)')

print('\nALL CHECKS PASSED')
