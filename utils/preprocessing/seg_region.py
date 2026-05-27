import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import SimpleITK as sitk

# template file name 
TAGS = {
    "4p5to8p5":  "AAL3v1_in_MNIPed_4p5to8p5_aal_in_template.nii.gz",
    "7to11":     "AAL3v1_in_MNIPed_7to11_aal_in_template.nii.gz",
    "7.5to13p5": "AAL3v1_in_MNIPed_7.5to13p5_aal_in_template.nii.gz",
    "10to14":    "AAL3v1_in_MNIPed_10to14_aal_in_template.nii.gz",
    "13to18p5":  "AAL3v1_in_MNIPed_13to18p5_aal_in_template.nii.gz",
    "4p5to18p5":  "AAL3v1_in_MNIPed_4p5to18p5_aal_in_template.nii.gz",
}
# rough registration: middle age of each template
MIDS = {
    "13to18p5":  6.5,
    "13to18p5":     9.0,
    "13to18p5": 10.5,
    "13to18p5":    12.0,
    "13to18p5":  15.75,
    "13to18p5":  25
}

def _strip_niigz(name: str) -> str:
    """去掉 NIfTI 文件的后缀名"""
    if name.endswith(".nii.gz"): return name[:-7]
    if name.endswith(".nii"):    return name[:-4]
    return name

def _read_image(img_path: Path) -> sitk.Image:
    """读取 NIfTI 图像并返回 SimpleITK 图像对象"""
    if not img_path.exists():
        raise FileNotFoundError(f"文件 {img_path} 不存在。")
    return sitk.ReadImage(str(img_path))

def _resample(moving: sitk.Image, reference: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    """重采样图像到参考空间"""
    return sitk.Resample(
        moving, reference, transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID()
    )

def _resample_mask(mask: sitk.Image, reference: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    """重采样二值掩膜到参考空间"""
    return sitk.Resample(mask, reference, transform,
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

def _write_image(img: sitk.Image, path: Path):
    """写入 NIfTI 图像到磁盘"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), True)

def _write_transform(tx: sitk.Transform, path: Path):
    """保存配准变换（刚体或仿射）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(tx, str(path))

def _prepare_image_for_registration(img: sitk.Image) -> sitk.Image:
    """确保图像为 3D，并转换为 sitkFloat32 类型"""
    # 如果是 4D 图像，提取第一维度
    if img.GetDimension() == 4:
        img = sitk.Extract(img, [img.GetSize()[0], img.GetSize()[1], img.GetSize()[2], 0], [0, 0, 0, 0])
    
    # 强制转换为 sitkFloat32 类型
    img = sitk.Cast(img, sitk.sitkFloat32)
    return img

def _registration_rigid(
    fixed: sitk.Image, moving: sitk.Image
) -> Tuple[sitk.Transform, sitk.Transform]:
    """刚体配准：返回刚体变换和合成变换"""
    # 刚体初始化
    fixed = _prepare_image_for_registration(fixed)
    moving = _prepare_image_for_registration(moving)
    init_rigid = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    # 刚体优化
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-3, 200, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(init_rigid, inPlace=False)

    rigid = reg.Execute(fixed, moving)
    comp = sitk.CompositeTransform(3)
    comp.AddTransform(rigid)

    return rigid, comp

def segment_region(img_dir: str,
                   out_dir: str,
                   template_path: str,
                   meta_path: str = None,
                   type: str = "aal") -> Dict[str, str]:
    """
    对所有标准化图像进行脑区分割，并根据年龄选择对应的 AAL 模板进行配准。
    """
    # 读取 meta 文件
    df = pd.read_excel(meta_path)
    if not {"name", "age"}.issubset(df.columns):
        raise ValueError("meta_3DT1.xlsx 需要包含列：'name', 'age'")

    # 获取所有标准化图像路径
    in_path = Path(img_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 遍历所有标准化图像
    for _, row in df.iterrows():
        name = str(row["name"]).strip()        
        age = float(row["age"]) if not isinstance(row["age"], str) else float(5.0)
        
        # 获取图像路径
        moving_img_path = in_path / f"{name}.nii.gz"

        if not moving_img_path.exists():
            print(f"[WARN] 跳过：找不到图像 {moving_img_path}")
            continue

        print(f"\n=== {name} | age={age} ===")

        # ---------- 选择粗配准模板（方案B：最近中点） ----------
        tag_key = min(MIDS.keys(), key=lambda k: abs(age - MIDS[k]))  # 选择距离年龄最近的模板
        if age == 'nan':
            tag_key = "13to18p5" 
        aal_template_path = Path(template_path) / TAGS[tag_key]

        if not aal_template_path.exists():
            print(f"[WARN] 模板缺失 {aal_template_path}")
            continue

        # 读取 AAL 模板图像
        aal_template = sitk.ReadImage(str(aal_template_path))

        # 读取患者标准化图像
        moving = sitk.ReadImage(str(moving_img_path))

        # 刚体配准：将 AAL 模板与患者图像配准
        rigid_tx, composite_tx = _registration_rigid(fixed=aal_template, moving=moving)

        # 重采样 AAL 模板到患者图像空间
        aal_in_patient = _resample_mask(aal_template, moving, composite_tx)

        # 保存配准后的 AAL 模板
        output_file = out_path / f"{name}_aal_in_patient.nii.gz"
        _write_image(aal_in_patient, output_file)

        # 保存配准变换
        transform_file = out_path / f"{name}_rigid.tfm"
        _write_transform(rigid_tx, transform_file)

        print(f"AAL 模板已成功迁移到患者图像空间：{output_file}")

    return {"status": "completed"}
