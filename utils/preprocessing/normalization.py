import os
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk

def _robust_zscore(img: sitk.Image, mask_arr: np.ndarray, p_low=0.5, p_high=99.5, eps=1e-8):
    """在给定 mask 内做分位裁剪后的 Z-score 归一化。"""
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    vals = arr[mask_arr]
    lo, hi = np.percentile(vals, [p_low, p_high])
    arr_clip = np.clip(arr, lo, hi)
    mu  = float(arr_clip[mask_arr].mean())
    sd  = float(arr_clip[mask_arr].std() + eps)
    arr_z = (arr_clip - mu) / sd

    out = sitk.GetImageFromArray(arr_z.astype(np.float32))
    out.CopyInformation(img)

    stats = {
        "p_low": p_low, "p_high": p_high,
        "lo": float(lo), "hi": float(hi),
        "mu": mu, "sd": sd,
    }
    return out, stats

def normalization_zscore(in_dir: str, out_dir: str, *,
                         p_low: float = 0.5, p_high: float = 99.5,
                         use_otsu_fallback: bool = True,
                         min_brain_voxels: int = 1000) -> None:
    """
    对注册后的无颅骨脑图像做稳健 Z-score 归一化（0.5–99.5% 分位裁剪）。
    输入目录结构（示例）:
      in_dir/
        ASD_1_3/coarse/registered_brain.nii.gz
        HC_2_15/coarse/registered_brain.nii.gz
        ...

    输出：
      out_dir/ASD_1_3.nii.gz
      out_dir/ASD_1_3.stats.json
      ...

    参数:
      p_low, p_high: 分位裁剪阈值
      use_otsu_fallback: 非零体素过少时是否用 Otsu 阈值作为兜底脑内掩码
      min_brain_voxels: 判定有效脑体素的最小数量
    """
    in_root  = Path(in_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 遍历受试者目录（ASD_C_P / HC_C_P）
    for subj_dir in sorted([p for p in in_root.iterdir() if p.is_dir()]):
        name = subj_dir.name  # e.g., ASD_1_23
        in_img_path = subj_dir / "coarse" / "registered_brain.nii.gz"
        if not in_img_path.is_file():
            print(f"[跳过] 找不到影像：{in_img_path}")
            continue

        try:
            img = sitk.ReadImage(str(in_img_path))
            arr = sitk.GetArrayFromImage(img)

            # 默认脑内 mask：非零体素
            mask = (arr != 0)

            # 兜底：非零体素太少则用 Otsu
            if mask.sum() < min_brain_voxels and use_otsu_fallback:
                otsu = sitk.OtsuThreshold(img, 0, 1)
                mask = sitk.GetArrayFromImage(otsu) > 0

            if mask.sum() < min_brain_voxels:
                print(f"[跳过] {name}: 有效脑内体素过少（{mask.sum()}），可能是坏数据。")
                continue

            out_img, stats = _robust_zscore(img, mask, p_low=p_low, p_high=p_high)

            out_img_path   = out_root / f"{name}.nii.gz"
            out_stats_path = out_root / f"{name}.stats.json"
            sitk.WriteImage(out_img, str(out_img_path))
            with open(out_stats_path, "w") as f:
                json.dump(stats, f, indent=2)

            print(f"[完成] {name} -> {out_img_path.name}")

        except Exception as e:
            print(f"[错误] {name}: {e}")
