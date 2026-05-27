"""
    to register the input images to the template, include 2 steps:
    1. rough: register the different age type images to different age template
    2. fine:  register the rough registered images to the final template
"""
import os
from pathlib import Path
from typing import Tuple, Optional, Dict
import pandas as pd
import SimpleITK as sitk
import numpy as np


# template file name 
TAGS = {
    "4p5to8p5":  "tpl-MNIPed_4p5to8p5_res-1",
    "7to11":     "tpl-MNIPed_7to11_res-1",
    "7.5to13p5": "tpl-MNIPed_7.5to13p5_res-1",
    "10to14":    "tpl-MNIPed_10to14_res-1",
    "13to18p5":  "tpl-MNIPed_13to18p5_res-1",
    "all":       "tpl-MNIPed_4p5to18p5_res-1",  # 精配准目标（统一空间）
}
# rough registration: middle age of each template
MIDS = {
    "all":  6.5,
    "all":     9.0,
    "all": 10.5,
    "all":    12.0,
    "all":  15.75,
    "all": 25  # 用于统一模板下的分析
}


# ===================== 工具函数 =====================
def _tpl_paths(template_dir: Path, tag_key: str) -> Tuple[Path, Optional[Path]]:
    """
    返回模板T1w与mask路径 (t1w_path, mask_path or None)
    """
    base = TAGS[tag_key]
    t1 = template_dir / f"{base}_T1w.nii.gz"
    msk = template_dir / f"{base}_desc-brain_mask.nii.gz"
    return t1, (msk if msk.exists() else None)


def _pick_template_nearest_mid(age: float) -> str:
    """方案B：按中点最近选择 tag_key"""
    return min(MIDS.keys(), key=lambda k: abs(age - MIDS[k]))


def _read_image(img_path: Path) -> sitk.Image:
    if not img_path.exists():
        raise FileNotFoundError(str(img_path))
    return sitk.ReadImage(str(img_path))


def _write_image(img: sitk.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), True)


def _write_transform(tx: sitk.Transform, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(tx, str(path))


def _resample(moving: sitk.Image, reference: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    return sitk.Resample(
        moving, reference, transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    
def _resample_mask(mask: sitk.Image, reference: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    return sitk.Resample(mask, reference, transform,
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

def _squeeze_to_scalar_3d(img: sitk.Image) -> sitk.Image:
    # 矢量像素 → 取第0通道；4D且最后一维=1 → Extract为3D
    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorIndexSelectionCast(img, 0)
    if img.GetDimension() == 4:
        sx, sy, sz, st = img.GetSize()
        if st != 1:
            raise ValueError(f"Image is 4D with t={st} != 1，无法挤压为3D。")
        img = sitk.Extract(img, [sx, sy, sz, 0], [0, 0, 0, 0])
    return img

def _prepare_image_for_metric(img: sitk.Image) -> sitk.Image:
    img = _squeeze_to_scalar_3d(img)
    if img.GetPixelID() != sitk.sitkFloat32:
        img = sitk.Cast(img, sitk.sitkFloat32)
    return img

def _prepare_mask(mask: sitk.Image, dilate_mm: float = 0.0) -> sitk.Image:
    """把各种“掩膜/抠空图”统一成 0/1 的二值 UInt8 掩膜；可选膨胀（物理单位）。"""
    mask = _squeeze_to_scalar_3d(mask)

    # 读取数值范围，自动决定阈值：
    stats = sitk.StatisticsImageFilter(); stats.Execute(mask)
    vmin, vmax = float(stats.GetMinimum()), float(stats.GetMaximum())

    # 若像素在 [0,1] 内，多半是 prob/0-1 掩膜，用 0.5；否则用 >0 兼容“抠空强度图”
    thr = 0.5 if (0.0 <= vmin and vmax <= 1.0) else 0.0
    mask = sitk.Cast(mask > thr, sitk.sitkUInt8)

    # 可选：轻微闭运算去小孔洞
    # mask = sitk.BinaryMorphologicalClosing(mask, (1,1,1))

    # 可选：按毫米膨胀
    if dilate_mm > 0:
        # 由 mm 转换为各向素半径（向上取整至少 1）
        sp = mask.GetSpacing()
        rad = tuple(max(1, int(np.ceil(dilate_mm/s))) for s in sp)
        mask = sitk.BinaryDilate(mask, rad)

    return mask


def _center_of_image(img: sitk.Image):
    size = np.array(img.GetSize(), dtype=float)
    spacing = np.array(img.GetSpacing(), dtype=float)
    origin = np.array(img.GetOrigin(), dtype=float)
    R = np.array(img.GetDirection(), dtype=float).reshape(3,3)
    c_idx = (size - 1.0) * spacing * 0.5
    return tuple(origin + R.dot(c_idx))

def _registration_rigid(
    fixed: sitk.Image, moving: sitk.Image,
    fixed_mask: Optional[sitk.Image] = None,
    moving_mask: Optional[sitk.Image] = None,
    init_mode: str = "MOMENTS",  # 可选 "GEOMETRY" / "MOMENTS"
) -> tuple[sitk.Transform, sitk.Transform]:
    """
    仅刚体配准。
    返回：(rigid, composite)  其中 composite = rigid
    """
    # --- 刚体初始化 ---
    mode = (sitk.CenteredTransformInitializerFilter.GEOMETRY
            if init_mode.upper()=="GEOMETRY"
            else sitk.CenteredTransformInitializerFilter.MOMENTS)
    try:
        init_rigid = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(), mode
        )
    except Exception as e:
        print(f"[WARN] CTI失败：{e}\n[WARN] 改用几何居中初始化。")
        init_rigid = sitk.Euler3DTransform()
        init_rigid.SetIdentity()
        fx_c = np.array(_center_of_image(fixed))
        mv_c = np.array(_center_of_image(moving))
        init_rigid.SetCenter(tuple(fx_c))
        init_rigid.SetTranslation(tuple(fx_c - mv_c))

    # --- 刚体优化 ---
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    if fixed_mask:  reg.SetMetricFixedMask(fixed_mask)
    if moving_mask: reg.SetMetricMovingMask(moving_mask)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-3, 200, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    try:
        reg.SetInitialTransform(init_rigid, inPlace=False)
    except TypeError:
        reg.SetInitialTransform(init_rigid, False)

    rigid = reg.Execute(fixed, moving)

    # 组合变换：就等于刚体本身
    comp = sitk.CompositeTransform(3)
    comp.AddTransform(rigid)
    return rigid, comp



def _registration_affine(
    fixed: sitk.Image, moving: sitk.Image,
    fixed_mask: Optional[sitk.Image] = None,
    moving_mask: Optional[sitk.Image] = None,
) -> tuple[sitk.Transform, sitk.Transform, sitk.Transform]:
    """
    Coarse registration: Rigid + Affine
    Returns: (rigid, affine, composite)
    """
    # 1. Rigid Initialization (GEOMETRY to avoid neck artifacts)
    try:
        init_rigid = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
    except Exception as e:
        print(f"[WARN] CTI failed: {e}. Fallback to manual geometry centering.")
        init_rigid = sitk.Euler3DTransform()
        init_rigid.SetIdentity()
        fx_c = np.array(_center_of_image(fixed))
        mv_c = np.array(_center_of_image(moving))
        init_rigid.SetCenter(fx_c)
        init_rigid.SetTranslation(tuple(mv_c - fx_c))

    # 2. Rigid Registration (Masks disabled for initial overlap)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-3, 200, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    try:
        reg.SetInitialTransform(init_rigid, inPlace=False)
    except TypeError:
        reg.SetInitialTransform(init_rigid, False)
        
    rigid = reg.Execute(fixed, moving)

    # 3. Affine Registration (Fixed mask ONLY, reduced learning rate)
    affine_init = sitk.AffineTransform(3)
    reg2 = sitk.ImageRegistrationMethod()
    reg2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    
    if fixed_mask:  
        reg2.SetMetricFixedMask(fixed_mask)
    # moving_mask is intentionally ignored to prevent overlap errors
    
    reg2.SetInterpolator(sitk.sitkLinear)
    reg2.SetOptimizerAsRegularStepGradientDescent(0.5, 5e-4, 300, relaxationFactor=0.5)
    reg2.SetOptimizerScalesFromPhysicalShift()
    reg2.SetShrinkFactorsPerLevel([4, 2, 1])
    reg2.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg2.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    reg2.SetMovingInitialTransform(rigid)
    try:
        reg2.SetInitialTransform(affine_init, inPlace=False)
    except TypeError:
        reg2.SetInitialTransform(affine_init, False)
        
    affine = reg2.Execute(fixed, moving)

    # 4. Composite Transform
    comp = sitk.CompositeTransform(3)
    comp.AddTransform(rigid)
    comp.AddTransform(affine)
    return rigid, affine, comp


def _registration_bspline(
    fixed: sitk.Image, moving: sitk.Image,
    fixed_mask: Optional[sitk.Image] = None,
    moving_mask: Optional[sitk.Image] = None,
    grid_spacing_mm: float = 40.0,
) -> sitk.Transform:
    """
    精配准：B-spline（Mattes MI，多分辨率）。
    兼容不同 SimpleITK 版本的优化器接口：
      1) 优先 LBFGSB (新API，含 defaultStepLength/lineSearchAccuracy)
      2) 退化到 LBFGSB (老API，少量参数)
      3) 再退到 GradientDescent（万一 LBFGSB 不可用）
    """
    # 初始化 B-spline 网格
    fixed_size = fixed.GetSize()
    fixed_spacing = fixed.GetSpacing()
    mesh_size = [
        max(1, int(round((fixed_size[i]*fixed_spacing[i]) / grid_spacing_mm)))
        for i in range(3)
    ]
    tx = sitk.BSplineTransformInitializer(fixed, mesh_size, order=3)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    if fixed_mask:  reg.SetMetricFixedMask(fixed_mask)
    if moving_mask: reg.SetMetricMovingMask(moving_mask)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetShrinkFactorsPerLevel([3, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # 逐级尝试优化器配置
    ok = False
    try:
        # 新版 SimpleITK（若支持）
        reg.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=200,
            maximumNumberOfCorrections=5,
            # 新版可能支持下面两个，如果你版本不支持会抛 TypeError
            defaultStepLength=1.0,
            lineSearchAccuracy=0.9,
        )
        ok = True
    except TypeError:
        try:
            # 老版 LBFGSB（参数更少）
            reg.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=1e-5,
                numberOfIterations=200,
                maximumNumberOfCorrections=5,
            )
            ok = True
        except Exception:
            pass

    if not ok:
        # 最后退路：普通梯度下降
        reg.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=250,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
        reg.SetOptimizerScalesFromPhysicalShift()

    reg.SetInitialTransform(tx, inPlace=False)
    bspline = reg.Execute(fixed, moving)
    return bspline



# ===================== 主流程 =====================
def template_registration(in_path: Path, out_path: Path, mask_path: Path, meta_path: Path, template_path: Path, do_fine: bool = False):
    """
    两步配准：
    1) 粗配准：按方案B（最近中点）选择 cohort 模板，做 刚体+仿射；
    2) 精配准（可选）：把粗结果再对齐到 4.5–18.5 模板，做 B-spline 非线性。
    
    [NEW]: Includes Resume/Checkpointing functionality to skip already processed stages.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    mask_path = Path(mask_path)
    template_path = Path(template_path)

    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(meta_path)
    if not {"name", "age"}.issubset(df.columns):
        raise ValueError("meta_3DT1.xlsx requires columns: 'name', 'age'")

    tpl_all_t1, tpl_all_msk = _tpl_paths(template_path, "all")
    if not tpl_all_t1.exists():
        raise FileNotFoundError(f"Unified template not found: {tpl_all_t1}")

    for _, row in df.iterrows():
        name = str(row["name"]).strip()        
        age = float(row["age"]) if not isinstance(row["age"], str) else float(5.0)
        
        moving_img_path = in_path / f"{name}_den_n4.nii.gz"
        moving_msk_path = mask_path / f"{name}_den_n4.nii.gz"  
        
        if not moving_img_path.exists():
            print(f"[WARN] Skip: Cannot find image {moving_img_path}")
            continue
            
        use_mask = moving_msk_path.exists()
        if not use_mask:
            print(f"[WARN] Mask missing, will register without mask: {moving_msk_path}")

        # =====================================================================
        # [Checkpointing Logic] Check existing outputs to resume safely
        # =====================================================================
        out_coarse_dir = out_path / name / "coarse"
        out_fine_dir = out_path / name / "fine"

        coarse_done = (out_coarse_dir / "registered_brain.nii.gz").exists() and (out_coarse_dir / "affine.tfm").exists()
        fine_done = (out_fine_dir / "composite.h5").exists() and (out_fine_dir / "registered.nii.gz").exists()

        if do_fine and fine_done:
            print(f"\n[SKIP] === {name} | Coarse & Fine registration already completed. ===")
            continue
        elif not do_fine and coarse_done:
            print(f"\n[SKIP] === {name} | Coarse registration already completed. ===")
            continue
        # =====================================================================

        print(f"\n=== Processing {name} | age={age} ===")

        # Pick template
        tag_key = _pick_template_nearest_mid(age)
        tpl_t1, tpl_msk = _tpl_paths(template_path, tag_key)
        if not tpl_t1.exists():
            print(f"[WARN] Template {tpl_t1} missing. Fallback to unified template.")
            tpl_t1, tpl_msk = tpl_all_t1, tpl_all_msk

        # Load images
        fixed_coarse   = _prepare_image_for_metric(_read_image(tpl_t1))
        fixed_coarse_m = _prepare_mask(_read_image(tpl_msk)) if (use_mask and tpl_msk and tpl_msk.exists()) else None

        moving         = _prepare_image_for_metric(_read_image(moving_img_path))
        moving_m       = _prepare_mask(_read_image(moving_msk_path)) if use_mask else None

        # ---------- Stage 1: Coarse Registration ----------
        coarse_comp = sitk.CompositeTransform(3)
        
        if not coarse_done:
            print(f"[Coarse] Registering to template={tpl_t1.name} (tag={tag_key})")
            rigid_tx, affine_tx, coarse_comp_generated = _registration_affine(
                fixed=fixed_coarse, moving=moving,
                fixed_mask=fixed_coarse_m, moving_mask=moving_m
            )
            
            out_coarse_dir.mkdir(parents=True, exist_ok=True)
            _write_transform(rigid_tx,  out_coarse_dir / "rigid.tfm")
            _write_transform(affine_tx, out_coarse_dir / "affine.tfm")        
            
            coarse_comp.AddTransform(rigid_tx)
            coarse_comp.AddTransform(affine_tx)
            moved_coarse = _resample(moving, fixed_coarse, coarse_comp)
            
            _write_image(moved_coarse, out_coarse_dir / "registered.nii.gz")
            
            if use_mask and moving_m is not None:
                moved_mask_coarse = _resample_mask(moving_m, fixed_coarse, coarse_comp)  
                _write_image(moved_mask_coarse, out_coarse_dir / "registered_mask.nii.gz")
                brain = sitk.Mask(moved_coarse, moved_mask_coarse)
                _write_image(brain, out_coarse_dir / "registered_brain.nii.gz")
                
            print(f"[Coarse] Done and saved for {name}.")
        else:
            # If coarse is done but fine isn't, fast-load the transforms
            print(f"[Coarse] Found existing coarse results. Loading transforms directly...")
            rigid_tx = sitk.ReadTransform(str(out_coarse_dir / "rigid.tfm"))
            affine_tx = sitk.ReadTransform(str(out_coarse_dir / "affine.tfm"))
            coarse_comp.AddTransform(rigid_tx)
            coarse_comp.AddTransform(affine_tx)

        if not do_fine:
            continue

        # ---------- Stage 2: Fine Registration (B-spline) ----------
        fixed_fine   = _prepare_image_for_metric(_read_image(tpl_all_t1))
        fixed_fine_m = _prepare_mask(_read_image(tpl_all_msk)) if (tpl_all_msk and tpl_all_msk.exists()) else None

        moved_for_bspline = _resample(moving, fixed_fine, coarse_comp)

        print(f"[Fine] Non-linear registration to unified template={tpl_all_t1.name}")
        bspline = _registration_bspline(
            fixed=fixed_fine, moving=moved_for_bspline,
            fixed_mask=fixed_fine_m, moving_mask=None,  
            grid_spacing_mm=40.0
        )

        composite = sitk.CompositeTransform(3)
        composite.AddTransform(bspline)
        composite.AddTransform(coarse_comp)

        moved_fine = _resample(moving, fixed_fine, composite)

        out_fine_dir.mkdir(parents=True, exist_ok=True)
        _write_image(moved_fine, out_fine_dir / "registered.nii.gz")
        _write_transform(bspline,   out_fine_dir / "bspline.tfm")
        _write_transform(bspline,   out_fine_dir / "bspline.h5")
        _write_transform(coarse_comp, out_fine_dir / "coarse.h5")     
        _write_transform(composite, out_fine_dir / "composite.h5")

        print(f"[Done] Coarse + Fine completed for {name}.")

    