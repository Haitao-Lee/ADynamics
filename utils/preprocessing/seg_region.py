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
    """Remove NIfTI file suffix"""
    if name.endswith(".nii.gz"): return name[:-7]
    if name.endswith(".nii"):    return name[:-4]
    return name

def _read_image(img_path: Path) -> sitk.Image:
    """Read NIfTI image and return SimpleITK image object"""
    if not img_path.exists():
        raise FileNotFoundError(f"File {img_path} does not exist.")
    return sitk.ReadImage(str(img_path))

def _resample(moving: sitk.Image, reference: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    """Resample image to reference space"""
    return sitk.Resample(
        moving, reference, transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID()
    )

def _resample_mask(mask: sitk.Image, reference: sitk.Image, transform: sitk.Transform) -> sitk.Image:
    """Resample binary mask to reference space"""
    return sitk.Resample(mask, reference, transform,
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

def _write_image(img: sitk.Image, path: Path):
    """Write NIfTI image to disk"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), True)

def _write_transform(tx: sitk.Transform, path: Path):
    """Save registration transform (rigid or affine)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(tx, str(path))

def _prepare_image_for_registration(img: sitk.Image) -> sitk.Image:
    """Ensure image is 3D and convert to sitkFloat32 type"""
    # If 4D image, extract first dimension
    if img.GetDimension() == 4:
        img = sitk.Extract(img, [img.GetSize()[0], img.GetSize()[1], img.GetSize()[2], 0], [0, 0, 0, 0])

    # Force convert to sitkFloat32 type
    img = sitk.Cast(img, sitk.sitkFloat32)
    return img

def _registration_rigid(
    fixed: sitk.Image, moving: sitk.Image
) -> Tuple[sitk.Transform, sitk.Transform]:
    """Rigid registration: returns rigid transform and composite transform"""
    # Rigid initialization
    fixed = _prepare_image_for_registration(fixed)
    moving = _prepare_image_for_registration(moving)
    init_rigid = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    # Rigid optimization
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
    Perform brain region segmentation on all normalized images, selecting the corresponding
    AAL template based on age for registration.
    """
    # Read meta file
    df = pd.read_excel(meta_path)
    if not {"name", "age"}.issubset(df.columns):
        raise ValueError("meta_3DT1.xlsx requires columns: 'name', 'age'")

    # Get all normalized image paths
    in_path = Path(img_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Iterate all normalized images
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        age = float(row["age"]) if not isinstance(row["age"], str) else float(5.0)

        # Get image path
        moving_img_path = in_path / f"{name}.nii.gz"

        if not moving_img_path.exists():
            print(f"[WARN] Skip: Cannot find image {moving_img_path}")
            continue

        print(f"\n=== {name} | age={age} ===")

        # ---------- Select coarse registration template (Scheme B: nearest midpoint) ----------
        tag_key = min(MIDS.keys(), key=lambda k: abs(age - MIDS[k]))  # Select template nearest to age
        if age == 'nan':
            tag_key = "13to18p5"
        aal_template_path = Path(template_path) / TAGS[tag_key]

        if not aal_template_path.exists():
            print(f"[WARN] Template missing {aal_template_path}")
            continue

        # Read AAL template image
        aal_template = sitk.ReadImage(str(aal_template_path))

        # Read patient normalized image
        moving = sitk.ReadImage(str(moving_img_path))

        # Rigid registration: register AAL template to patient image
        rigid_tx, composite_tx = _registration_rigid(fixed=aal_template, moving=moving)

        # Resample AAL template to patient image space
        aal_in_patient = _resample_mask(aal_template, moving, composite_tx)

        # Save registered AAL template
        output_file = out_path / f"{name}_aal_in_patient.nii.gz"
        _write_image(aal_in_patient, output_file)

        # Save registration transform
        transform_file = out_path / f"{name}_rigid.tfm"
        _write_transform(rigid_tx, transform_file)

        print(f"AAL template successfully transformed to patient space: {output_file}")

    return {"status": "completed"}
