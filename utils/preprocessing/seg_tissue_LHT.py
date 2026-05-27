import os
import json
from pathlib import Path
from typing import Dict
import SimpleITK as sitk
import numpy as np

def _strip_niigz(name: str) -> str:
    if name.endswith(".nii.gz"): return name[:-7]
    if name.endswith(".nii"): return name[:-4]
    return name

def _compute_metrics_from_pve(pve_csf_path: Path, pve_gm_path: Path, pve_wm_path: Path) -> Dict:
    pve_csf_img = sitk.ReadImage(str(pve_csf_path))
    pve_gm_img  = sitk.ReadImage(str(pve_gm_path))
    pve_wm_img  = sitk.ReadImage(str(pve_wm_path))

    csf = sitk.GetArrayFromImage(pve_csf_img).astype(np.float64)
    gm  = sitk.GetArrayFromImage(pve_gm_img ).astype(np.float64)
    wm  = sitk.GetArrayFromImage(pve_wm_img ).astype(np.float64)

    sx, sy, sz = pve_csf_img.GetSpacing()
    voxvol_mm3 = float(sx * sy * sz)

    GM_mm3  = float(gm.sum()  * voxvol_mm3)
    WM_mm3  = float(wm.sum()  * voxvol_mm3)
    CSF_mm3 = float(csf.sum() * voxvol_mm3)
    ICV_mm3 = GM_mm3 + WM_mm3 + CSF_mm3

    GM_mL, WM_mL, CSF_mL, ICV_mL = [v/1000.0 for v in (GM_mm3, WM_mm3, CSF_mm3, ICV_mm3)]
    def frac(x): return float(x/ICV_mm3) if ICV_mm3 > 0 else float("nan")
    
    pve_sum = csf + gm + wm
    brain_mask = pve_sum > 0.01
    mean_abs_dev = float(np.mean(np.abs(pve_sum[brain_mask] - 1.0))) if brain_mask.any() else float("nan")

    return {
        "voxel_size_mm": [float(sx), float(sy), float(sz)],
        "voxel_volume_mm3": voxvol_mm3,
        "volume_mm3": {"GM": GM_mm3, "WM": WM_mm3, "CSF": CSF_mm3, "ICV": ICV_mm3},
        "volume_mL":  {"GM": GM_mL, "WM": WM_mL, "CSF": CSF_mL, "ICV": ICV_mL},
        "fractions":  {"GM": frac(GM_mm3), "WM": frac(WM_mm3), "CSF": frac(CSF_mm3)},
        "brain_voxels": int(brain_mask.sum()),
        "pve_sum_mean_abs_deviation": mean_abs_dev
    }

def segment_tissue(in_dir: str, out_dir: str, **kwargs) -> Dict[str, Dict[str, str]]:
    """
    Python-native Tissue Segmentation using K-Means Clustering.
    Replaces FSL FAST for generating GM/WM/CSF masks & metrics.
    """
    in_root  = Path(in_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(in_root.glob("*.nii.gz"))
    if not files:
        print(f"[segment_tissue] No *.nii.gz found in {in_root}.")
        return {}

    results: Dict[str, Dict[str, str]] = {}

    for f in files:
        case = _strip_niigz(f.name)
        case_dir = out_root / case
        case_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = case_dir / case

        print(f"[segment_tissue] Processing KMeans clustering for: {case}")
        
        try:
            # 1. Read Image & Extract Brain Mask
            img = sitk.ReadImage(str(f))
            arr = sitk.GetArrayFromImage(img)
            brain_mask = arr > 1e-3  # Ignore background
            brain_pixels = arr[brain_mask]
            
            if len(brain_pixels) == 0:
                print(f"[segment_tissue][ERROR] Empty brain mask for {case}.")
                continue

            # 2. Fast 1D K-Means (K=3 for CSF, GM, WM)
            k = 3
            # Initialize centers roughly based on intensity percentiles
            centers = np.percentile(brain_pixels, [16, 50, 84])
            for _ in range(30):
                dists = np.abs(brain_pixels[:, None] - centers[None, :])
                labels = np.argmin(dists, axis=1)
                new_centers = np.array([
                    brain_pixels[labels == i].mean() if (labels == i).any() else centers[i] 
                    for i in range(k)
                ])
                if np.allclose(centers, new_centers): break
                centers = new_centers

            # Sort centers: Darkest = CSF(1), Mid = GM(2), Brightest = WM(3)
            sort_idx = np.argsort(centers)
            label_map = {sort_idx[0]: 1, sort_idx[1]: 2, sort_idx[2]: 3}
            
            # 3. Map back to 3D Array
            seg_arr = np.zeros_like(arr, dtype=np.uint8)
            brain_labels = np.zeros_like(brain_pixels, dtype=np.uint8)
            for i in range(k):
                brain_labels[labels == i] = label_map[i]
            seg_arr[brain_mask] = brain_labels

            # Save Main Segmentation (0=BG, 1=CSF, 2=GM, 3=WM)
            seg_img = sitk.GetImageFromArray(seg_arr)
            seg_img.CopyInformation(img)
            seg_path = out_prefix.with_name(out_prefix.name + "_seg.nii.gz")
            sitk.WriteImage(seg_img, str(seg_path))

            # 4. Generate Pseudo-PVEs (Hard masks cast to float32 for metric compability)
            pve_paths = []
            for tissue_val, suffix in zip([1, 2, 3], ["_pve_0.nii.gz", "_pve_1.nii.gz", "_pve_2.nii.gz"]):
                pve_arr = (seg_arr == tissue_val).astype(np.float32)
                pve_img = sitk.GetImageFromArray(pve_arr)
                pve_img.CopyInformation(img)
                pve_path = out_prefix.with_name(out_prefix.name + suffix)
                sitk.WriteImage(pve_img, str(pve_path))
                pve_paths.append(pve_path)

            # Record Outputs
            results[case] = {
                "seg": str(seg_path),
                "pve_csf": str(pve_paths[0]),
                "pve_gm": str(pve_paths[1]),
                "pve_wm": str(pve_paths[2])
            }

            # 5. Compute and Save Metrics
            metrics = _compute_metrics_from_pve(pve_paths[0], pve_paths[1], pve_paths[2])
            metrics["note"] = "Computed using Native KMeans (SimpleITK) instead of FSL FAST"
            
            with open(case_dir / "metrics.json", "w") as fjson:
                json.dump(metrics, fjson, indent=2)

        except Exception as e:
            print(f"[segment_tissue][ERROR] Failed on {case}: {e}")
            continue

        print(f"[segment_tissue] Done: {case}")

    return results