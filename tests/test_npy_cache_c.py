"""Quick test: trainer finds the .npy on C: via sha1(path), not
alongside the .nii.gz. Verifies data identity vs nibabel.
"""
import os, sys, time, json, hashlib
sys.path.insert(0, r'E:\LHT_workspace\AD\ADynamics')
os.chdir(r'E:\LHT_workspace\AD\ADynamics')

# Make sure the .npy is NOT alongside the .nii.gz (we just moved them
# all to C:). The test should still find them via the new npy_cache_dir.
import glob
siblings = [f for f in glob.glob(r'E:\LHT_workspace\AD\processed_data\**\*.npy', recursive=True) if '.tmp' not in f]
print(f'sibling .npy on E: {len(siblings)}  (should be 0 after the move)')

# Find the hash for a sample
import hashlib
def h(p):
    # Match MultiModalDataset._load_npy_cached: normalize then hash.
    norm = os.path.normpath(p).replace(os.sep, "/")
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]

# Load manifest
with open(r'E:\LHT_workspace\AD\ADynamics\core_data\dataset_manifest_merged_v2.json') as f:
    manifest = json.load(f)
items = manifest if isinstance(manifest, list) else manifest.get('samples', list(manifest.values()))

# Find a sample with T1 + fMRI + ASL
sample = None
for item in items[:200]:
    paths = item.get('paths', item)
    t1 = paths.get('t1')
    fmri = paths.get('fmri')
    asl = paths.get('asl')
    if t1 and fmri and asl and os.path.exists(t1) and os.path.exists(fmri) and os.path.exists(asl):
        sample = {'t1': t1, 'fmri': fmri, 'asl': asl}
        break
t1, fmri, asl = sample['t1'], sample['fmri'], sample['asl']
print(f't1   hash = {h(t1)}  expected at C:/ADynamics_npy_cache/{h(t1)}.npy')
print(f'fmri hash = {h(fmri)}  expected at C:/ADynamics_npy_cache/{h(fmri)}.npy')
print(f'asl  hash = {h(asl)}  expected at C:/ADynamics_npy_cache/{h(asl)}.npy')

# Verify each cache file exists on C:
for name, p in [('t1', t1), ('fmri', fmri), ('asl', asl)]:
    cp = f'C:/ADynamics_npy_cache/{h(p)}.npy'
    exists = os.path.exists(cp)
    sz = os.path.getsize(cp) if exists else 0
    print(f'  {name}: {cp}  exists={exists}  size={sz/1024**2:.1f} MB')
    assert exists, f'cache file missing: {cp}'

# Now test the dataset helper
from core_data.dataset import MultiModalDataset
ds = MultiModalDataset(
    data_list=[{'t1': t1, 'fmri': fmri, 'asl': asl, 'label': 0, 'patient_id': 'test'}],
    transform=None,
    use_npy_cache=True,
    npy_cache_dir='C:/ADynamics_npy_cache',
)
print('\n--- Verifying T1 via npy_cache_dir ---')
arr = ds._load_npy_cached(t1)
print(f'  shape={arr.shape} dtype={arr.dtype}')
# Compare to nibabel
import nibabel as nib, numpy as np
ref = np.asarray(nib.load(t1).dataobj, dtype=np.float32)
diff = np.abs(ref - arr).max()
print(f'  max diff vs nibabel: {diff:.2e}')
assert diff < 1e-3
print('  PASS')

print('\n--- Verifying fMRI (4D) ---')
arr = ds._load_npy_cached(fmri)
ref = np.asarray(nib.load(fmri).dataobj, dtype=np.float32)
print(f'  shape={arr.shape}')
diff = np.abs(ref - arr).max()
print(f'  max diff vs nibabel: {diff:.2e}')
assert diff < 1e-3
print('  PASS')

print('\n--- Verifying ASL ---')
arr = ds._load_npy_cached(asl)
ref = np.asarray(nib.load(asl).dataobj, dtype=np.float32)
print(f'  shape={arr.shape}')
diff = np.abs(ref - arr).max()
print(f'  max diff vs nibabel: {diff:.2e}')
assert diff < 1e-3
print('  PASS')

# Benchmark
N = 10
print(f'\n--- Benchmark ({N} reads each) ---')
for name, p in [('T1', t1), ('fMRI', fmri), ('ASL', asl)]:
    t0 = time.time()
    for _ in range(N):
        _ = np.asarray(nib.load(p).dataobj, dtype=np.float32)
    nib_ms = (time.time() - t0) / N * 1000
    t0 = time.time()
    for _ in range(N):
        _ = ds._load_npy_cached(p)
    npy_ms = (time.time() - t0) / N * 1000
    print(f'  {name}: nibabel={nib_ms:.0f}ms  vs  npy(C:/)={npy_ms:.1f}ms  ({nib_ms/max(npy_ms,0.1):.1f}x)')

print('\nALL CHECKS PASSED')
