"""
ADynamics End-to-End Inference Pipeline.

This is the工程结晶 (engineering masterpiece) that combines all trained
modules into a complete disease progression modeling system.

Pipeline Flow:
    1. Load new NC patient T1 MRI and patient age
    2. VAE Encoder: Extract initial latent z0
    3. CFM Euler Integration: Evolve z0 -> z_final using learned velocity field
    4. Deformation Generator: Generate 3D displacement field from z_final
    5. Spatial Transformer: Apply warp to original MRI
    6. Save results for 3D Slicer QC

Usage:
    python scripts/inference_pipeline.py \
        --input path/to/NC_T1.nii.gz \
        --age 70 \
        --vae_checkpoint checkpoints/stage1_vae/vae_best.pt \
        --cfm_checkpoint checkpoints/stage3_cfm/cfm_best.pt \
        --deform_checkpoint checkpoints/stage4_deform/deform_best.pt \
        --output_dir ./inference_results
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data.transforms import get_val_transforms
from models.spatial_transform import (
    DeformationGenerator,
    SpatialTransformer,
    create_identity_flow,
)
from models.vector_field import VelocityFieldNet
from models.vae3d import ADynamicsVAE3D
from utils.io_utils import save_tensor_to_nifti


class EvolvePipeline:
    """
    End-to-end AD progression evolution pipeline.

    Combines:
        - VAE Encoder (frozen)
        - CFM Velocity Field (trained)
        - Deformation Generator (trained)
        - Spatial Transformer

    To evolve an NC brain MRI through disease stages to AD.
    """

    def __init__(
        self,
        vae: ADynamicsVAE3D,
        vector_field: VelocityFieldNet,
        deform_generator: DeformationGenerator,
        device: Union[str, torch.device] = "cuda",
        spatial_size: Tuple[int, int, int] = (128, 128, 128),
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

        # Move models to device and set to eval mode
        self.vae = vae.to(self.device)
        self.vae.eval()

        self.vector_field = vector_field.to(self.device)
        self.vector_field.eval()

        self.deform_generator = deform_generator.to(self.device)
        self.deform_generator.eval()

        # Initialize spatial transformer
        self.stn = SpatialTransformer(mode="bilinear", padding_mode="border")

        # Preprocessing transforms
        self.transform = get_val_transforms(spatial_size=spatial_size)

    @torch.no_grad()
    def encode(self, mri: Tensor) -> Tensor:
        """
        Encode MRI to latent representation.

        Args:
            mri: MRI tensor of shape [1, 1, D, H, W]

        Returns:
            Latent tensor of shape [1, latent_channels, D', H', W']
        """
        mu, _ = self.vae.encode(mri)
        return mu

    @torch.no_grad()
    def integrate_ode(
        self,
        z0: Tensor,
        c: Optional[Tensor] = None,
        steps: int = 20,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Euler integration of velocity field from t=0 to t=1.

        Solves: dz/dt = v(z, t) starting from z0

        Args:
            z0: Initial latent [1, C, D', H', W']
            c: Optional clinical conditions [1, num_conditions]
            steps: Number of integration steps

        Returns:
            Tuple of (z_final, trajectory) where trajectory is list of z_t
        """
        z_t = z0.clone()
        dt = 1.0 / steps
        trajectory = [z_t.clone()]

        for i in range(steps):
            t = torch.tensor([i * dt], device=self.device)
            v_t = self.vector_field(z_t, t, c)
            z_t = z_t + v_t * dt
            trajectory.append(z_t.clone())

        return z_t, trajectory

    @torch.no_grad()
    def generate_deformation(
        self,
        z_final: Tensor,
    ) -> Tensor:
        """
        Generate deformation field from evolved latent.

        Args:
            z_final: Evolved latent [1, C, D', H', W']

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

    @torch.no_grad()
    def evolve(
        self,
        mri: Tensor,
        c: Optional[Tensor] = None,
        ode_steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Full evolution pipeline.

        Args:
            mri: Input MRI [1, 1, D, H, W]
            c: Optional clinical conditions [1, num_conditions]
            ode_steps: Number of ODE integration steps

        Returns:
            Dictionary containing:
                - evolved_mri: Final evolved MRI
                - deformation_field: 3D displacement field
                - z_final: Final latent state
                - trajectory: List of intermediate latent states
                - z0: Initial latent state
        """
        # Step 1: Encode to latent
        z0 = self.encode(mri)

        # Step 2: Integrate ODE in latent space
        z_final, trajectory = self.integrate_ode(z0, c, steps=ode_steps)

        # Step 3: Generate deformation field
        flow = self.generate_deformation(z_final)

        # Step 4: Apply warp to original image
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
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """
        Save evolution results to NIfTI files.

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
            voxel_size: Voxel spacing in mm
        """
        os.makedirs(output_dir, exist_ok=True)

        # Create affine if not provided
        if affine is None:
            affine = np.eye(4, dtype=np.float64)
            affine[0, 0] = voxel_size[0]
            affine[1, 1] = voxel_size[1]
            affine[2, 2] = voxel_size[2]

        # Save original MRI
        if "original_mri" in results:
            original = results["original_mri"]
            if isinstance(original, Tensor):
                original = original.cpu().numpy()
            save_tensor_to_nifti(
                original,
                affine,
                os.path.join(output_dir, f"{patient_id}_original.nii.gz"),
                voxel_size=voxel_size,
            )

        # Save evolved MRI
        evolved_mri = results["evolved_mri"]
        if isinstance(evolved_mri, Tensor):
            evolved_mri = evolved_mri.cpu().numpy()
        save_tensor_to_nifti(
            evolved_mri,
            affine,
            os.path.join(output_dir, f"{patient_id}_evolved.nii.gz"),
            voxel_size=voxel_size,
        )

        # Save deformation field components
        flow = results["deformation_field"]
        if isinstance(flow, Tensor):
            flow = flow.cpu().numpy()

        # Save each component separately (for 3D Slicer visualization)
        for i, dim_name in enumerate(["D", "H", "W"]):
            flow_component = flow[0, i]  # shape: [D, H, W]
            save_tensor_to_nifti(
                torch.from_numpy(flow_component),
                affine,
                os.path.join(output_dir, f"{patient_id}_flow_{dim_name}.nii.gz"),
                voxel_size=voxel_size,
            )

        # Save trajectory as numpy array for analysis
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
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
    transform=None,
) -> Tuple[Tensor, np.ndarray]:
    """
    Load and preprocess MRI file.

    Args:
        filepath: Path to NIfTI file
        spatial_size: Target spatial size for preprocessing
        transform: Optional MONAI transforms

    Returns:
        Tuple of (preprocessed_tensor, original_affine)
    """
    # Load NIfTI
    nii = nib.load(filepath)
    affine = nii.affine
    data = nii.get_fdata().astype(np.float32)
    # shape: [D, H, W]

    # Convert to tensor and add dimensions
    mri = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    # shape: [1, 1, D, H, W]

    # Apply transforms
    if transform is not None:
        mri = transform({"image": mri})["image"]

    return mri, affine


def create_dummy_mri(
    spatial_size: Tuple[int, int, int] = (128, 128, 128),
) -> Tensor:
    """
    Create a dummy MRI for testing.

    Args:
        spatial_size: Spatial dimensions

    Returns:
        Dummy MRI tensor [1, 1, D, H, W]
    """
    D, H, W = spatial_size

    # Create brain-like ellipsoid
    x = np.linspace(-1, 1, D)
    y = np.linspace(-1, 1, H)
    z = np.linspace(-1, 1, W)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    # Brain shape
    ellipsoid = (xx**2 + yy**2 * 1.2 + zz**2 * 0.8) <= 0.7
    brain = ellipsoid.astype(np.float32)

    # Add some structure (white matter, gray matter)
    wm_mask = (xx**2 + yy**2 * 1.2 + zz**2 * 0.8) <= 0.4
    gm_mask = ellipsoid & ~wm_mask

    brain[wm_mask] = 0.9
    brain[gm_mask] = 0.6
    brain[~ellipsoid] = 0.1

    # Add some noise
    brain = brain + np.random.randn(D, H, W).astype(np.float32) * 0.05
    brain = np.clip(brain, 0, 1)

    mri = torch.from_numpy(brain).unsqueeze(0).unsqueeze(0)
    # shape: [1, 1, D, H, W]

    return mri


def main():
    parser = argparse.ArgumentParser(
        description="ADynamics End-to-End Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input NC T1 MRI NIfTI file",
    )
    parser.add_argument(
        "--age",
        type=float,
        default=None,
        help="Patient age (normalized to [0, 1] internally)",
    )
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
    parser.add_argument(
        "--spatial_size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Spatial size for preprocessing",
    )
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=20,
        help="Number of ODE integration steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--use_dummy",
        action="store_true",
        help="Use dummy MRI instead of real file",
    )

    args = parser.parse_args()

    spatial_size = tuple(args.spatial_size)
    device = torch.device(args.device)

    print("\n" + "=" * 60)
    print("ADynamics End-to-End Inference Pipeline")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Spatial size: {spatial_size}")
    print(f"ODE steps: {args.ode_steps}")

    # Initialize models with architecture matching checkpoints
    latent_channels = 64  # Must match training config

    vae = ADynamicsVAE3D(
        spatial_size=spatial_size,
        in_channels=1,
        latent_channels=latent_channels,
    )

    vector_field = VelocityFieldNet(
        latent_channels=latent_channels,
        latent_spatial=(8, 8, 8),
        time_embed_dim=128,
        cond_embed_dim=64,
        num_conditions=1,
    )

    deform_generator = DeformationGenerator(
        latent_channels=latent_channels,
        latent_spatial=(8, 8, 8),
        output_spatial=spatial_size,
    )

    # Load checkpoints
    if os.path.exists(args.vae_checkpoint):
        print(f"\nLoading VAE from {args.vae_checkpoint}")
        state_dict = torch.load(args.vae_checkpoint, map_location=device)
        vae.load_state_dict(state_dict["model_state_dict"])

    if os.path.exists(args.cfm_checkpoint):
        print(f"Loading CFM from {args.cfm_checkpoint}")
        state_dict = torch.load(args.cfm_checkpoint, map_location=device)
        vector_field.load_state_dict(state_dict["model_state_dict"])

    if os.path.exists(args.deform_checkpoint):
        print(f"Loading Deformation Generator from {args.deform_checkpoint}")
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

    # Load or create MRI
    if args.use_dummy:
        print("\nUsing dummy MRI for inference")
        mri = create_dummy_mri(spatial_size)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = np.diag([1.0, 1.0, 1.0])
    elif args.input:
        print(f"\nLoading MRI from {args.input}")
        transform = get_val_transforms(spatial_size=spatial_size)
        mri, affine = load_mri(args.input, spatial_size, transform)
    else:
        raise ValueError("Must provide --input or --use_dummy")

    # Prepare input
    mri = mri.to(device)
    print(f"Input MRI shape: {tuple(mri.shape)}")

    # Prepare condition (normalized age)
    c = None
    if args.age is not None:
        age_normalized = args.age / 100.0  # Normalize to [0, 1]
        c = torch.tensor([[age_normalized]], dtype=torch.float32).to(device)
        print(f"Condition: age = {args.age} -> normalized = {age_normalized}")

    # Run evolution
    print("\nRunning evolution pipeline...")
    print("  1. Encoding MRI to latent...")
    print("  2. CFM Euler integration (z0 -> z_final)...")
    print("  3. Generating deformation field...")
    print("  4. Applying spatial warp...")

    results = pipeline.evolve(mri, c=c, ode_steps=args.ode_steps)

    print("\n  Evolution complete!")
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
        voxel_size=(1.0, 1.0, 1.0),
    )

    # Compute some statistics
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
