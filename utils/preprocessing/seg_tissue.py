import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict
import SimpleITK as sitk
import numpy as np

def _strip_niigz(name: str) -> str:
    """Remove .nii.gz or .nii suffix, return case name."""
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name

def _resolve_fast_cmd(user_cmd: str | None = None) -> str:
    """Find the fast executable path. Priority: user-specified > PATH > $FSLDIR inference."""
    if user_cmd:
        return user_cmd
    p = shutil.which("fast")
    if p:
        return p
    fsl_dir = os.environ.get("FSLDIR")
    # Common locations: /home/.../fsl/bin/fast or /home/.../fsl/share/fsl/bin/fast
    candidates = []
    if fsl_dir:
        candidates += [
            str(Path(fsl_dir) / "bin" / "fast"),
            str(Path(fsl_dir) / "share" / "fsl" / "bin" / "fast"),
        ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        "Cannot find FSL FAST executable. Please ensure: (1) `which fast` finds it in terminal; "
        "(2) Pass absolute path via fast_cmd parameter, e.g., '/home/syx/fsl/share/fsl/bin/fast'; "
        "(3) Or set FSLDIR environment variable and add $FSLDIR/bin to PATH."
    )

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

def segment_tissue(in_dir: str, out_dir: str,
                   fast_cmd: str | None = None,
                   image_type: int = 1,  # T1
                   num_classes: int = 3,
                   save_bias: bool = True) -> Dict[str, Dict[str, str]]:
    """
    Batch FSL FAST tissue segmentation:
      - Each case gets its own subfolder out_dir/ASD_C_P/
      - Computes and writes metrics.json (PVE volume/ICV/volume fractions)
    """
    in_root  = Path(in_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(in_root.glob("*.nii.gz"))
    if not files:
        print(f"[segment_tissue] No *.nii.gz found in {in_root}.")
        return {}

    fast_bin = _resolve_fast_cmd(fast_cmd)

    results: Dict[str, Dict[str, str]] = {}
    base_env = os.environ.copy()
    base_env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")

    fsl_dir = "/home/syx/fsl"
    base_env["FSLDIR"] = fsl_dir
    base_env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    # Ensure fsl/bin is at the front of PATH
    base_env["PATH"] = f"{fsl_dir}/bin:" + base_env.get("PATH","")
    # Critical: add fsl/lib to avoid conda's libstdc++ taking priority
    base_env["LD_LIBRARY_PATH"] = f"{fsl_dir}/lib:" + base_env.get("LD_LIBRARY_PATH","")
    # Avoid locale triggering parsing issues
    base_env["LC_ALL"] = "C"
    base_env["LANG"]   = "C"

    for f in files:
        case = _strip_niigz(f.name)                 # Correct case name: ASD_1_1
        case_dir = out_root / case
        case_dir.mkdir(parents=True, exist_ok=True)

        out_prefix = case_dir / case                # Prefix: .../ASD_1_1/ASD_1_1
        cmd = [fast_bin, "-t", str(image_type), "-n", str(num_classes), "-o", str(out_prefix)]
        if save_bias:
            cmd.insert(3, "-B")

        print(f"[segment_tissue] RUN: {' '.join(cmd)}  INPUT={f}")
        try:
            subprocess.run(
                cmd + [str(f)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True, text=True, env=base_env
            )
        except subprocess.CalledProcessError as e:
            print(f"[segment_tissue][ERROR] {case} FAST failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
            continue
        except FileNotFoundError:
            print(f"[segment_tissue][ERROR] FAST executable not found: {fast_bin}")
            continue

        seg_path     = out_prefix.with_name(out_prefix.name + "_seg.nii.gz")
        pve0_path    = out_prefix.with_name(out_prefix.name + "_pve_0.nii.gz")  # CSF
        pve1_path    = out_prefix.with_name(out_prefix.name + "_pve_1.nii.gz")  # GM
        pve2_path    = out_prefix.with_name(out_prefix.name + "_pve_2.nii.gz")  # WM
        bias_path    = out_prefix.with_name(out_prefix.name + "_bias.nii.gz")
        restore_path = out_prefix.with_name(out_prefix.name + "_restore.nii.gz")

        out_map = {"seg": str(seg_path)}
        if pve0_path.exists(): out_map["pve_csf"] = str(pve0_path)
        if pve1_path.exists(): out_map["pve_gm"]  = str(pve1_path)
        if pve2_path.exists(): out_map["pve_wm"]  = str(pve2_path)
        if save_bias and bias_path.exists():       out_map["bias"] = str(bias_path)
        if save_bias and restore_path.exists():    out_map["restore"] = str(restore_path)
        results[case] = out_map

        # Write metrics.json
        metrics_path = case_dir / "metrics.json"
        try:
            if all(p.exists() for p in (pve0_path, pve1_path, pve2_path)):
                metrics = _compute_metrics_from_pve(pve0_path, pve1_path, pve2_path)
            else:
                # Fallback: hard label estimation
                img = sitk.ReadImage(str(seg_path))
                seg = sitk.GetArrayFromImage(img).astype(np.int16)
                sx, sy, sz = img.GetSpacing()
                voxvol_mm3 = float(sx*sy*sz)
                GM_mm3  = float((seg == 1).sum() * voxvol_mm3)
                WM_mm3  = float((seg == 2).sum() * voxvol_mm3)
                CSF_mm3 = float((seg == 0).sum() * voxvol_mm3)
                ICV_mm3 = GM_mm3 + WM_mm3 + CSF_mm3
                GM_mL, WM_mL, CSF_mL, ICV_mL = [v/1000.0 for v in (GM_mm3, WM_mm3, CSF_mm3, ICV_mm3)]
                def frac(x): return float(x/ICV_mm3) if ICV_mm3 > 0 else float("nan")
                metrics = {
                    "voxel_size_mm": [float(sx), float(sy), float(sz)],
                    "voxel_volume_mm3": voxvol_mm3,
                    "volume_mm3": {"GM": GM_mm3, "WM": WM_mm3, "CSF": CSF_mm3, "ICV": ICV_mm3},
                    "volume_mL":  {"GM": GM_mL, "WM": WM_mL, "CSF": CSF_mL, "ICV": ICV_mL},
                    "fractions":  {"GM": frac(GM_mm3), "WM": frac(WM_mm3), "CSF": frac(CSF_mm3)},
                    "note": "PVE unavailable, using hard label estimation"
                }
            with open(metrics_path, "w") as fjson:
                json.dump(metrics, fjson, indent=2)
        except Exception as e:
            print(f"[segment_tissue][WARN] {case} metrics computation failed: {e}")

        print(f"[segment_tissue] Done: {case} -> {case_dir}")

    return results
