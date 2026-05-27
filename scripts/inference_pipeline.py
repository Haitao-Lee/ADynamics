"""
ADynamics End-to-End Inference Pipeline with Preprocessing.

Combines preprocessing pipeline and trained modules into a complete disease progression modeling system:

Preprocessing (optional, toggle with --preprocess):
    1. Denoise T1 images using ANTsPy
    2. N4 bias field correction
    3. Registration to template space

Inference:
    1. Load preprocessed NC patient T1 MRI and patient age
    2. VAE Encoder: Extract initial latent z0 (shape: [1, C, 16, 16, 12] for HD)
    3. CFM Euler Integration: Evolve z0 -> z_final using learned velocity field
    4. Deformation Generator: Generate 3D displacement field from z_final
    5. Spatial Transformer: Apply warp to original MRI
    6. Save results for 3D Slicer QC

Usage:
    # With preprocessing (raw T1 -> preprocessed -> inference)
    python scripts/inference_pipeline.py \
        --input path/to/raw_T1.nii.gz \
        --preprocess \
        --age 70 \
        --template_dir E:/LHT_workspace/AD/T1/preprocess/tools \
        --vae_checkpoint checkpoints/stage1_vae/vae_best.pt \
        --cfm_checkpoint checkpoints/stage3_cfm/cfm_best.pt \
        --deform_checkpoint checkpoints/stage4_deform/deform_best.pt \
        --output_dir ./inference_results

    # Without preprocessing (already preprocessed registered_brain.nii.gz)
    python scripts/inference_pipeline.py \
        --input path/to/registered_brain.nii.gz \
        --skip_preprocessing \
        --age 70 \
        --vae_checkpoint checkpoints/stage1_vae/vae_best.pt \
        --cfm_checkpoint checkpoints/stage3_cfm/cfm_best.pt \
        --deform_checkpoint checkpoints/stage4_deform/deform_best.pt \
        --output_dir ./inference_results
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# Add local preprocessing utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils" / "preprocessing"))

from core_data.transforms import get_val_transforms
from monai.transforms import LoadImaged
from models.spatial_transform import (
    DeformationGenerator,
    SpatialTransformer,
)
from models.vector_field import VelocityFieldNet
from models.vae3d import MultiModalVAE3D
from utils.io_utils import save_tensor_to_nifti


# HD configuration
HD_SPATIAL_SIZE = (256, 256, 192)
HD_LATENT_SPATIAL = (16, 16, 12)


# ===================== Preprocessing Functions =====================

def _run_denoise(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Path:
    """
    Step 1: Denoise T1 image using ANTsPy.

    Args:
        input_path: Path to raw T1 nii.gz
        output_dir: Output directory for denoised image

    Returns:
        Path to denoised image
    """
    from utils.denoise import denoise_single_t1_antspy

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem.replace(".nii", "")
    output_path = output_dir / f"{stem}_den.nii.gz"

    if output_path.exists():
        print(f"[Denoise] Skip (exists): {output_path}")
        return output_path

    print(f"[Denoise] Processing: {input_path.name}")
    denoise_single_t1_antspy(
        in_nii=str(input_path),
        out_nii=str(output_path),
        brain_mask_nii=None,
        noise_model="Rician",
        verbose=True,
    )
    return output_path


def _run_hd_bet(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    device: str = "cuda:0",
) -> Path:
    """
    Step 2: Generate brainmask using HD-BET.

    Args:
        input_path: Path to input nii.gz (denoised or bias-corrected)
        output_dir: Output directory for brainmask
        device: GPU device for HD-BET

    Returns:
        Path to brainmask
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem.replace(".nii", "")
    output_path = output_dir / f"{stem}.nii.gz"

    if output_path.exists():
        print(f"[HD-BET] Skip (exists): {output_path}")
        return output_path

    print(f"[HD-BET] Processing: {input_path.name}")

    cmd = [
        "hd-bet",
        "-i", str(input_path),
        "-o", str(output_dir / stem),
        "-device", device,
    ]
    import subprocess
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"HD-BET failed: {result.stderr}")

    # HD-BET outputs {stem}_mask.nii.gz, rename to match expected naming
    mask_from_hd = output_dir / f"{stem}_mask.nii.gz"
    if mask_from_hd.exists() and not output_path.exists():
        shutil.move(str(mask_from_hd), str(output_path))

    return output_path


def _run_n4_bias_correction(
    input_path: Union[str, Path],
    mask_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Path:
    """
    Step 3: N4 bias field correction.

    Args:
        input_path: Path to denoised image
        mask_path: Path to brainmask
        output_dir: Output directory for bias-corrected image

    Returns:
        Path to bias-corrected image
    """
    from utils.n4_bias_correction import n4_bias_correction

    input_path = Path(input_path)
    mask_path = Path(mask_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem.replace(".nii", "")
    output_path = output_dir / f"{stem}_n4.nii.gz"

    if output_path.exists():
        print(f"[N4] Skip (exists): {output_path}")
        return output_path

    print(f"[N4] Processing: {input_path.name}")
    n4_bias_correction(
        in_nii=str(input_path),
        out_nii=str(output_path),
        mask_nii=str(mask_path) if mask_path.exists() else None,
        shrink_factor=2,
        bspline_fitting_distance=200.0,
        verbose=True,
    )
    return output_path


def _run_registration(
    input_path: Union[str, Path],
    mask_path: Union[str, Path],
    template_dir: Union[str, Path],
    output_dir: Union[str, Path],
    age: float,
) -> Tuple[Path, Path]:
    """
    Step 4: Registration to template space.

    Args:
        input_path: Path to N4 bias-corrected image
        mask_path: Path to brainmask
        template_dir: Path to template directory
        output_dir: Output directory for registered results
        age: Patient age for template selection

    Returns:
        Tuple of (registered_brain_path, registered_image_path)
    """
    from utils.registration import template_registration
    import pandas as pd

    input_path = Path(input_path)
    mask_path = Path(mask_path)
    template_dir = Path(template_dir)
    output_dir = Path(output_dir)

    stem = input_path.stem.replace(".nii", "")

    # Create temporary metadata file for single-subject registration
    temp_dir = output_dir / "temp_meta"
    temp_dir.mkdir(parents=True, exist_ok=True)
    meta_path = temp_dir / "meta_temp.xlsx"
    df = pd.DataFrame({"name": [stem], "age": [age]})
    df.to_excel(meta_path, index=False)

    # Setup input structure as expected by template_registration
    temp_input = temp_dir / "input"
    temp_input.mkdir(parents=True, exist_ok=True)

    # Copy input and mask with expected naming: {name}_den_n4.nii.gz
    stem = input_path.stem.replace(".nii", "").replace("_n4", "")
    import shutil
    shutil.copy(str(input_path), str(temp_input / f"{stem}_den_n4.nii.gz"))
    if mask_path.exists():
        shutil.copy(str(mask_path), str(temp_input / f"{stem}_den_n4.nii.gz"))

    # Registration outputs go to output_dir / name / coarse / registered_brain.nii.gz
    registered_brain = output_dir / stem / "coarse" / "registered_brain.nii.gz"
    registered_img = output_dir / stem / "coarse" / "registered.nii.gz"

    if registered_brain.exists():
        print(f"[Registration] Skip (exists): {registered_brain}")
        # Clean up temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        return registered_brain, registered_img

    print(f"[Registration] Processing: {input_path.name} (age={age})")

    try:
        template_registration(
            in_path=temp_input,
            out_path=output_dir,
            mask_path=temp_input,
            meta_path=meta_path,
            template_path=template_dir,
            do_fine=False,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return registered_brain, registered_img


def run_full_preprocessing(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    template_dir: Union[str, Path],
    age: float,
    device: str = "cuda:0",
    skip_denoise: bool = False,
    skip_hdbet: bool = False,
) -> Tuple[Path, Path]:
    """
    Run the full preprocessing pipeline: Denoise -> N4 -> Registration.

    Args:
        input_path: Path to raw T1 nii.gz
        output_dir: Output base directory for all preprocessing steps
        template_dir: Path to template directory (contains registration_template/)
        age: Patient age for template selection
        device: GPU device for HD-BET
        skip_denoise: Skip denoising step (if already done)
        skip_hdbet: Skip HD-BET step (if mask already exists)

    Returns:
        Tuple of (registered_brain_path, registered_image_path)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    template_dir = Path(template_dir)

    print("\n" + "=" * 60)
    print("Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Denoise
    denoised_dir = output_dir / "denoised"
    if skip_denoise:
        # Assume input is already denoised, just use it directly
        denoised_path = input_path
        print(f"[Denoise] Skipped, using input directly")
    else:
        denoised_path = _run_denoise(input_path, denoised_dir)

    # Step 2: HD-BET for brainmask (needed for N4)
    rough_mask_dir = output_dir / "hd-bet_brainmask_rough"
    if skip_hdbet:
        # Assume mask already exists, find it
        stem = input_path.stem.replace(".nii", "").replace("_den", "")
        rough_mask_path = rough_mask_dir / f"{stem}.nii.gz"
        if not rough_mask_path.exists():
            # Try common naming patterns
            for pattern in ["*_mask.nii.gz", "*_brain_mask.nii.gz"]:
                matches = list(rough_mask_dir.glob(pattern))
                if matches:
                    rough_mask_path = matches[0]
                    break
        print(f"[HD-BET] Skipped, using existing mask: {rough_mask_path}")
    else:
        rough_mask_path = _run_hd_bet(denoised_path, rough_mask_dir, device)

    # Step 3: N4 bias correction
    n4_dir = output_dir / "n4_bias_corrected"
    n4_path = _run_n4_bias_correction(denoised_path, rough_mask_path, n4_dir)

    # Step 4: Registration
    registered_brain, registered_img = _run_registration(
        input_path=n4_path,
        mask_path=rough_mask_path,
        template_dir=template_dir,
        output_dir=output_dir / "registered",
        age=age,
    )

    print("\n" + "=" * 60)
    print(f"Preprocessing Complete!")
    print(f"  Registered brain: {registered_brain}")
    print("=" * 60 + "\n")

    return registered_brain, registered_img


# ===================== Inference Pipeline =====================

class EvolvePipeline:
    """
    End-to-end AD progression evolution pipeline.

    Combines:
        - VAE Encoder (frozen)
        - CFM Velocity Field (trained)
        - Deformation Generator (trained)
        - Spatial Transformer

    All forward passes use torch.no_grad() for memory efficiency.
    """

    def __init__(
        self,
        vae: ADynamicsVAE3D,
        vector_field: VelocityFieldNet,
        deform_generator: DeformationGenerator,
        device: Union[str, torch.device] = "cuda",
        spatial_size: Tuple[int, int, int] = HD_SPATIAL_SIZE,
    ) -> None:
        """
        Initialize the evolution pipeline.

        Args:
            vae: Trained VAE model for encoding
            vector_field: Trained CFM velocity field network
            deform_generator: Trained deformation field generator
            device: Device to run inference on
            spatial_size: Spatial dimensions of input MRI
        """
        self.device = torch.device(device)
        self.spatial_size = spatial_size

        self.vae = vae.to(self.device)
        self.vae.eval()

        self.vector_field = vector_field.to(self.device)
        self.vector_field.eval()

        self.deform_generator = deform_generator.to(self.device)
        self.deform_generator.eval()

        self.stn = SpatialTransformer(mode="bilinear", padding_mode="border")
        self.transform = get_val_transforms(spatial_size=spatial_size)

    @torch.no_grad()
    def encode(self, mri: Tensor) -> Tensor:
        """
        Encode MRI to latent representation.

        Args:
            mri: MRI tensor of shape [1, 1, D, H, W]

        Returns:
            Latent tensor of shape [1, latent_channels, 16, 16, 12]
        """
        # MultiModalVAE3D.encode returns concat features, not mu
        # Use forward pass to get mu through fusion projection
        x_dict = {"t1": mri}
        z_concat = self.vae.encode(x_dict)
        mu = self.vae.fusion_proj(z_concat)
        return mu

    @torch.no_grad()
    def integrate_ode(
        self,
        z0: Tensor,
        c: Optional[Tensor] = None,
        age: Optional[Tensor] = None,
        sex: Optional[Tensor] = None,
        steps: int = 20,
        use_demographics: bool = False,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Euler integration of velocity field from t=0 to t=1.

        Args:
            z0: Initial latent [1, C, 16, 16, 12]
            c: Optional clinical conditions [1, num_conditions]
            age: Optional normalized ages [1, 1] for demographics conditioning
            sex: Optional binary sexes [1, 1] for demographics conditioning
            steps: Number of integration steps
            use_demographics: If True, use age/sex instead of c

        Returns:
            Tuple of (z_final, trajectory) where trajectory is list of z_t
        """
        z_t = z0.clone()
        dt = 1.0 / steps
        trajectory = [z_t.clone()]

        for i in tqdm(range(steps), desc="ODE Integration", leave=False):
            t = torch.tensor([i * dt], device=self.device, dtype=z_t.dtype)
            if use_demographics:
                v_t = self.vector_field(z_t, t, c=None, age=age, sex=sex)
            else:
                v_t = self.vector_field(z_t, t, c)
            z_t = z_t + v_t * dt
            trajectory.append(z_t.clone())

            if i % 10 == 0:
                torch.cuda.empty_cache()

        return z_t, trajectory

    @torch.no_grad()
    def generate_deformation(
        self,
        z_final: Tensor,
    ) -> Tensor:
        """
        Generate deformation field from evolved latent.

        Args:
            z_final: Evolved latent [1, C, 16, 16, 12]

        Returns:
            Deformation field [1, 3, D, H, W]
        """
        flow = self.deform_generator(z_final)
        return flow

    @torch.no_grad()
    def apply_warp(
        self,
        mri: Tensor,
        flow: Tensor,
    ) -> Tensor:
        """
        Apply deformation to MRI image.

        Args:
            mri: Original MRI [1, 1, D, H, W]
            flow: Deformation field [1, 3, D, H, W]

        Returns:
            Warped MRI [1, 1, D, H, W]
        """
        warped = self.stn(mri, flow)
        return warped

    def evolve(
        self,
        mri: Tensor,
        c: Optional[Tensor] = None,
        age: Optional[Tensor] = None,
        sex: Optional[Tensor] = None,
        ode_steps: int = 20,
        use_demographics: bool = False,
    ) -> Dict[str, Any]:
        """
        Full evolution pipeline with torch.no_grad() for memory efficiency.

        Args:
            mri: Input MRI [1, 1, D, H, W]
            c: Optional clinical conditions [1, num_conditions]
            age: Optional normalized ages [1, 1] for demographics conditioning
            sex: Optional binary sexes [1, 1] for demographics conditioning
            ode_steps: Number of ODE integration steps
            use_demographics: If True, use age/sex instead of c

        Returns:
            Dictionary containing evolved_mri, deformation_field, z_final, trajectory, z0
        """
        z0 = self.encode(mri)
        z_final, trajectory = self.integrate_ode(
            z0, c, age, sex, steps=ode_steps, use_demographics=use_demographics
        )
        flow = self.generate_deformation(z_final)
        evolved_mri = self.apply_warp(mri, flow)

        return {
            "evolved_mri": evolved_mri,
            "deformation_field": flow,
            "z_final": z_final,
            "trajectory": trajectory,
            "z0": z0,
        }

    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str,
        patient_id: str = "patient",
        affine: Optional[np.ndarray] = None,
        spacing: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Save evolution results to NIfTI files.

        Extracts real spacing from affine matrix when available.
        Uses permute_to_xyz=False to match spatial_transform conventions.

        Files saved:
            - {patient_id}_original.nii.gz: Original input MRI
            - {patient_id}_evolved.nii.gz: Evolved MRI (AD-like)
            - {patient_id}_flow_D.nii.gz: Displacement in D dimension
            - {patient_id}_flow_H.nii.gz: Displacement in H dimension
            - {patient_id}_flow_W.nii.gz: Displacement in W dimension
            - {patient_id}_trajectory.npz: Latent trajectory for analysis

        Args:
            results: Results dictionary from evolve()
            output_dir: Directory to save results
            patient_id: Identifier for the patient
            affine: 4x4 affine matrix for NIfTI (if None, creates identity)
            spacing: Physical voxel spacing (dx, dy, dz). If None, extracted from affine
        """
        os.makedirs(output_dir, exist_ok=True)

        if spacing is None and affine is not None:
            spacing = (
                float(np.sqrt(np.sum(affine[:3, 0] ** 2))),
                float(np.sqrt(np.sum(affine[:3, 1] ** 2))),
                float(np.sqrt(np.sum(affine[:3, 2] ** 2))),
            )

        if affine is None:
            if spacing is None:
                spacing = (1.0, 1.0, 1.0)
            affine = np.eye(4, dtype=np.float64)
            affine[0, 0] = spacing[0]
            affine[1, 1] = spacing[1]
            affine[2, 2] = spacing[2]

        if "original_mri" in results:
            original = results["original_mri"]
            if isinstance(original, Tensor):
                original = original.cpu().numpy()
            save_tensor_to_nifti(
                original,
                affine,
                os.path.join(output_dir, f"{patient_id}_original.nii.gz"),
                permute_to_xyz=False,
            )

        evolved_mri = results["evolved_mri"]
        if isinstance(evolved_mri, Tensor):
            evolved_mri = evolved_mri.cpu().numpy()
        save_tensor_to_nifti(
            evolved_mri,
            affine,
            os.path.join(output_dir, f"{patient_id}_evolved.nii.gz"),
            permute_to_xyz=False,
        )

        flow = results["deformation_field"]
        if isinstance(flow, Tensor):
            flow = flow.cpu().numpy()

        for i, dim_name in enumerate(["D", "H", "W"]):
            flow_component = flow[0, i]
            save_tensor_to_nifti(
                torch.from_numpy(flow_component),
                affine,
                os.path.join(output_dir, f"{patient_id}_flow_{dim_name}.nii.gz"),
                permute_to_xyz=False,
            )

        trajectory = results["trajectory"]
        trajectory_array = np.stack([t.cpu().numpy() for t in trajectory], axis=0)
        np.savez(
            os.path.join(output_dir, f"{patient_id}_trajectory.npz"),
            trajectory=trajectory_array,
            z0=results["z0"].cpu().numpy(),
            z_final=results["z_final"].cpu().numpy(),
        )

        print(f"\nResults saved to {output_dir}:")
        print(f"  - {patient_id}_original.nii.gz")
        print(f"  - {patient_id}_evolved.nii.gz")
        print(f"  - {patient_id}_flow_D.nii.gz")
        print(f"  - {patient_id}_flow_H.nii.gz")
        print(f"  - {patient_id}_flow_W.nii.gz")
        print(f"  - {patient_id}_trajectory.npz")


def load_mri(
    filepath: str,
    spatial_size: Tuple[int, int, int] = HD_SPATIAL_SIZE,
    transform=None,
) -> Tuple[Tensor, np.ndarray]:
    """
    Load and preprocess MRI file using MONAI for proper metadata preservation.

    Args:
        filepath: Path to NIfTI file
        spatial_size: Target spatial size for preprocessing
        transform: Optional MONAI transforms

    Returns:
        Tuple of (preprocessed_tensor, affine, image_meta_dict)
    """
    loader = LoadImaged(reader="NibabelReader", image_only=False)
    loaded = loader({"image": filepath})

    image = loaded["image"]
    image_meta_dict = loaded["image_meta_dict"]
    affine = image_meta_dict.get("affine", np.eye(4))

    if transform is not None:
        image = transform({"image": image})["image"]

    return image, affine, image_meta_dict


def main():
    parser = argparse.ArgumentParser(
        description="ADynamics End-to-End Inference Pipeline with Preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input T1 MRI NIfTI file (raw or preprocessed registered_brain.nii.gz)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--patient_id",
        type=str,
        default="patient",
        help="Patient identifier for output files",
    )

    # Preprocessing options
    preprocess_group = parser.add_argument_group("Preprocessing (optional)")
    preprocess_group.add_argument(
        "--preprocess",
        action="store_true",
        help="Run full preprocessing pipeline on raw T1 (denoise -> N4 -> registration)",
    )
    preprocess_group.add_argument(
        "--skip_preprocessing",
        action="store_true",
        help="Skip preprocessing (input is already preprocessed registered_brain.nii.gz)",
    )
    preprocess_group.add_argument(
        "--template_dir",
        type=str,
        default="ADynamics/utils/templates",
        help="Path to template directory (needed for --preprocess)",
    )
    preprocess_group.add_argument(
        "--preprocess_output_dir",
        type=str,
        default=None,
        help="Output directory for preprocessing (default: <output_dir>/preprocessed)",
    )
    preprocess_group.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU device for HD-BET",
    )

    # Demographics
    parser.add_argument(
        "--age",
        type=float,
        default=None,
        help="Patient age (required for preprocessing template selection and inference)",
    )
    parser.add_argument(
        "--sex",
        type=int,
        default=None,
        help="Patient sex: 0=female, 1=male (optional, for demographics-aware inference)",
    )

    # Model checkpoints
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="checkpoints/stage1_vae/vae_best.pt",
        help="Path to VAE checkpoint",
    )
    parser.add_argument(
        "--cfm_checkpoint",
        type=str,
        default="checkpoints/stage3_cfm/cfm_best.pt",
        help="Path to CFM velocity field checkpoint",
    )
    parser.add_argument(
        "--deform_checkpoint",
        type=str,
        default="checkpoints/stage4_deform/deform_best.pt",
        help="Path to deformation generator checkpoint",
    )

    # Model architecture (must match checkpoints)
    parser.add_argument(
        "--spatial_size",
        type=int,
        nargs=3,
        default=[256, 256, 192],
        help="Spatial size for preprocessing (HD: 256 256 192)",
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
        default=64,
        help="Number of latent channels (must match checkpoint)",
    )
    parser.add_argument(
        "--use_demographics",
        action="store_true",
        help="Use demographics (age/sex) conditioning instead of clinical conditions",
    )

    # Inference options
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=20,
        help="Number of ODE integration steps",
    )
    parser.add_argument(
        "--inference_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )

    args = parser.parse_args()

    # Validate preprocessing flags
    if args.preprocess and args.skip_preprocessing:
        raise ValueError("Cannot use both --preprocess and --skip_preprocessing")

    spatial_size = tuple(args.spatial_size)
    latent_channels = args.latent_channels
    device = torch.device(args.inference_device)

    print("\n" + "=" * 60)
    print("ADynamics End-to-End Inference Pipeline")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Input: {args.input}")

    # ===================== Preprocessing =====================
    input_path = Path(args.input)

    if args.preprocess:
        if args.age is None:
            raise ValueError("--age is required when using --preprocess")

        preprocess_output_dir = Path(args.preprocess_output_dir) if args.preprocess_output_dir else Path(args.output_dir) / "preprocessed"

        registered_brain, registered_img = run_full_preprocessing(
            input_path=input_path,
            output_dir=preprocess_output_dir,
            template_dir=Path(args.template_dir),
            age=args.age,
            device=args.device,
        )

        # Use the registered brain as input for inference
        input_path = registered_brain
        print(f"[Inference] Using preprocessed input: {input_path}")

    elif not args.skip_preprocessing:
        # Check if input is already preprocessed (registered_brain.nii.gz)
        # If it looks like a preprocessed file (contains registered), skip preprocessing
        if "registered" not in str(input_path) and not input_path.name.endswith("_brain.nii.gz"):
            print("\n[WARNING] Input does not appear to be preprocessed registered brain.")
            print("  Consider using --preprocess if this is raw T1 data.")
            print("  Or use --skip_preprocessing if input is already preprocessed registered_brain.nii.gz")

    # ===================== Load Models =====================
    print("\n[Model] Initializing models...")

    # Use MultiModalVAE3D to match training pipeline
    # Note: Training uses MultiModalVAE3D with optional modalities
    vae = MultiModalVAE3D(
        spatial_size=spatial_size,
        in_channels=1,
        latent_channels=latent_channels,
        base_channels=32,
        num_classes=4,
        dropout_rate=0.0,  # No dropout in inference
        decoder_depth=4,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
    )

    vector_field = VelocityFieldNet(
        latent_channels=latent_channels,
        latent_spatial=HD_LATENT_SPATIAL,
        time_embed_dim=128,
        cond_embed_dim=64,
        num_conditions=1,
        use_demographics=args.use_demographics,
    )

    deform_generator = DeformationGenerator(
        latent_channels=latent_channels,
        latent_spatial=HD_LATENT_SPATIAL,
        output_spatial=spatial_size,
        base_channels=16,
    )

    # Load checkpoints
    if os.path.exists(args.vae_checkpoint):
        print(f"[Model] Loading VAE from {args.vae_checkpoint}")
        state_dict = torch.load(args.vae_checkpoint, map_location=device)
        vae.load_state_dict(state_dict["model_state_dict"])

    if os.path.exists(args.cfm_checkpoint):
        print(f"[Model] Loading CFM from {args.cfm_checkpoint}")
        state_dict = torch.load(args.cfm_checkpoint, map_location=device)
        vector_field.load_state_dict(state_dict["model_state_dict"])

    if os.path.exists(args.deform_checkpoint):
        print(f"[Model] Loading Deformation Generator from {args.deform_checkpoint}")
        state_dict = torch.load(args.deform_checkpoint, map_location=device)
        deform_generator.load_state_dict(state_dict["model_state_dict"])

    # Create pipeline
    pipeline = EvolvePipeline(
        vae=vae,
        vector_field=vector_field,
        deform_generator=deform_generator,
        device=device,
        spatial_size=spatial_size,
    )

    # ===================== Load MRI =====================
    print(f"\n[Data] Loading MRI from {input_path}")
    transform = get_val_transforms(spatial_size=spatial_size)
    mri, affine, image_meta_dict = load_mri(str(input_path), spatial_size, transform)

    mri = mri.to(device)
    print(f"[Data] Input MRI shape: {tuple(mri.shape)}")

    # ===================== Prepare Condition =====================
    c = None
    age_tensor = None
    sex_tensor = None

    if args.use_demographics:
        if args.age is not None:
            age_normalized = args.age / 100.0
            age_tensor = torch.tensor([[age_normalized]], dtype=torch.float32).to(device)
            print(f"[Condition] Age: {args.age} -> normalized: {age_normalized}")
        if args.sex is not None:
            sex_tensor = torch.tensor([[args.sex]], dtype=torch.float32).to(device)
            print(f"[Condition] Sex: {args.sex} (0=female, 1=male)")
    else:
        if args.age is not None:
            age_normalized = args.age / 100.0
            c = torch.tensor([[age_normalized]], dtype=torch.float32).to(device)
            print(f"[Condition] Age: {args.age} -> normalized: {age_normalized}")

    # ===================== Run Inference =====================
    print("\n[Inference] Running evolution pipeline...")
    print("  1. Encoding MRI to latent...")
    print("  2. CFM Euler integration (z0 -> z_final)...")
    print("  3. Generating deformation field...")
    print("  4. Applying spatial warp...")

    results = pipeline.evolve(
        mri,
        c=c,
        age=age_tensor,
        sex=sex_tensor,
        ode_steps=args.ode_steps,
        use_demographics=args.use_demographics,
    )

    print("\n[Inference] Evolution complete!")
    print(f"  Initial latent z0 shape: {tuple(results['z0'].shape)}")
    print(f"  Final latent z_final shape: {tuple(results['z_final'].shape)}")
    print(f"  Trajectory length: {len(results['trajectory'])}")

    # Save results
    results["original_mri"] = mri
    pipeline.save_results(
        results,
        output_dir=args.output_dir,
        patient_id=args.patient_id,
        affine=affine,
    )

    # Compute statistics
    flow = results["deformation_field"]
    flow_np = flow.cpu().numpy()
    print(f"\nDeformation field statistics:")
    print(f"  Mean displacement: D={flow_np[0,0].mean():.3f}, H={flow_np[0,1].mean():.3f}, W={flow_np[0,2].mean():.3f}")
    print(f"  Std displacement:  D={flow_np[0,0].std():.3f}, H={flow_np[0,1].std():.3f}, W={flow_np[0,2].std():.3f}")
    print(f"  Max displacement:  D={flow_np[0,0].max():.3f}, H={flow_np[0,1].max():.3f}, W={flow_np[0,2].max():.3f}")

    print("\n" + "=" * 60)
    print("Inference complete! Results saved for 3D Slicer QC.")
    print("=" * 60)


if __name__ == "__main__":
    main()
