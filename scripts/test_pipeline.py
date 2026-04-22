"""
ADynamics Stage 1 Pipeline Integration Test

This script validates that all Stage 1 components are correctly integrated:
- Data transforms (MONAI)
- Dataset and DataLoader
- 3D VAE model
- Training loop with backward pass

Run this script to verify the pipeline before training:
    python scripts/test_pipeline.py
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data.dataset import create_dummy_dataset, get_dataloader
from core_data.transforms import get_train_transforms, get_val_transforms
from models.vae3d import ADynamicsVAE3D
from utils.io_utils import save_tensor_to_nifti


def test_transforms(spatial_size: tuple = (128, 128, 128)) -> bool:
    """
    Test MONAI transforms produce correct output shape.

    Args:
        spatial_size: Expected output spatial size

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("TEST 1: Data Transforms")
    print("=" * 60)

    # Create dummy data
    temp_dir = tempfile.mkdtemp(prefix="adynamics_test_")
    dummy_data_list = create_dummy_dataset(
        spatial_size=spatial_size,
        num_samples=2,
    )

    # Get transforms
    train_transforms = get_train_transforms(spatial_size=spatial_size)

    # Apply transforms to first sample
    sample = train_transforms(dummy_data_list[0])
    image = sample["image"]

    expected_shape = (1, *spatial_size)
    actual_shape = tuple(image.shape)

    print(f"  Input dummy file: {dummy_data_list[0]['image']}")
    print(f"  Expected shape: {expected_shape}")
    print(f"  Actual shape:   {actual_shape}")

    if actual_shape == expected_shape:
        print("  ✅ Transforms test PASSED")
        return True
    else:
        print("  ❌ Transforms test FAILED")
        return False


def test_vae_forward_pass(
    spatial_size: tuple = (128, 128, 128),
    in_channels: int = 1,
    latent_channels: int = 64,
) -> bool:
    """
    Test VAE forward pass produces expected shapes.

    Args:
        spatial_size: Input spatial size
        in_channels: Number of input channels
        latent_channels: Latent space channels

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("TEST 2: VAE Forward Pass")
    print("=" * 60)

    # Create VAE model
    model = ADynamicsVAE3D(
        spatial_size=spatial_size,
        in_channels=in_channels,
        latent_channels=latent_channels,
    )
    model.eval()

    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(
        batch_size,
        in_channels,
        *spatial_size,
    )
    print(f"  Input shape: {tuple(dummy_input.shape)}")

    # Forward pass without gradient
    with torch.no_grad():
        recon, mu, logvar = model(dummy_input)

    print(f"  Reconstruction shape: {tuple(recon.shape)}")
    print(f"  Mu shape:            {tuple(mu.shape)}")
    print(f"  Logvar shape:        {tuple(logvar.shape)}")

    # Verify shapes
    checks_passed = True

    if recon.shape != dummy_input.shape:
        print(f"  ❌ Reconstruction shape mismatch: expected {dummy_input.shape}, got {recon.shape}")
        checks_passed = False
    else:
        print("  ✅ Reconstruction shape correct")

    if mu.shape != logvar.shape:
        print(f"  ❌ Mu and Logvar shape mismatch")
        checks_passed = False
    else:
        print("  ✅ Mu and Logvar shapes match")

    # Verify latent dimensions
    expected_latent_spatial = tuple(s // (2**4) for s in spatial_size)  # 4 downsamples
    if mu.shape[2:] != expected_latent_spatial:
        print(f"  ❌ Latent spatial shape mismatch: expected {expected_latent_spatial}, got {mu.shape[2:]}")
        checks_passed = False
    else:
        print(f"  ✅ Latent spatial shape correct: {mu.shape[2:]}")

    if checks_passed:
        print("  ✅ VAE forward pass test PASSED")
    else:
        print("  ❌ VAE forward pass test FAILED")

    return checks_passed


def test_vae_training_step(
    spatial_size: tuple = (128, 128, 128),
    in_channels: int = 1,
    latent_channels: int = 64,
) -> bool:
    """
    Test VAE training step with backward pass.

    Args:
        spatial_size: Input spatial size
        in_channels: Number of input channels
        latent_channels: Latent space channels

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("TEST 3: VAE Training Step (Backward Pass)")
    print("=" * 60)

    # Create VAE model
    model = ADynamicsVAE3D(
        spatial_size=spatial_size,
        in_channels=in_channels,
        latent_channels=latent_channels,
    )
    model.train()

    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(
        batch_size,
        in_channels,
        *spatial_size,
    )
    print(f"  Input shape: {tuple(dummy_input.shape)}")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Forward pass
    recon, mu, logvar = model(dummy_input)

    # Compute loss
    recon_loss = torch.nn.functional.l1_loss(recon, dummy_input)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    num_latent_elements = mu.numel()
    kl_loss_normalized = kl_loss / num_latent_elements
    kl_weight = 0.0001
    loss = recon_loss + kl_weight * kl_loss_normalized

    print(f"  Reconstruction loss: {recon_loss.item():.6f}")
    print(f"  KL loss (normalized): {kl_loss_normalized.item():.6f}")
    print(f"  Total loss: {loss.item():.6f}")

    # Check for NaN
    if torch.isnan(loss):
        print("  ❌ Loss is NaN!")
        return False

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"  ❌ Gradients contain NaN for {name}")
                return False
            has_gradients = True

    if not has_gradients:
        print("  ❌ No gradients computed")
        return False

    # Optimizer step
    optimizer.step()

    print("  ✅ Backward pass successful")
    print("  ✅ Gradients computed correctly")
    print("  ✅ Optimizer step successful")
    print("  ✅ VAE training step test PASSED")
    return True


def test_nifti_io(spatial_size: tuple = (128, 128, 128)) -> bool:
    """
    Test NIfTI save/load roundtrip.

    Args:
        spatial_size: Spatial size for test tensor

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("TEST 4: NIfTI I/O Roundtrip")
    print("=" * 60)

    # Create dummy tensor
    tensor = torch.rand(*spatial_size)
    print(f"  Tensor shape: {tuple(tensor.shape)}")

    # Create identity affine
    affine = np.eye(4, dtype=np.float64)

    # Save to temporary file
    temp_dir = tempfile.mkdtemp(prefix="adynamics_nifti_test_")
    temp_path = os.path.join(temp_dir, "test_output.nii.gz")

    save_tensor_to_nifti(tensor, affine, temp_path)
    print(f"  Saved to: {temp_path}")

    # Load back
    loaded_nii = nib.load(temp_path)
    loaded_data = loaded_nii.get_fdata()

    print(f"  Loaded shape: {tuple(loaded_data.shape)}")
    print(f"  Loaded affine:\n{loaded_nii.affine}")

    # Verify
    if loaded_data.shape != tensor.shape:
        print("  ❌ Shape mismatch after roundtrip")
        return False

    # Check data is similar (accounting for float32 conversion)
    if not np.allclose(loaded_data, tensor.numpy(), rtol=1e-5):
        print("  ❌ Data mismatch after roundtrip")
        return False

    print("  ✅ NIfTI I/O test PASSED")
    return True


def test_dataloader(spatial_size: tuple = (128, 128, 128)) -> bool:
    """
    Test full dataloader pipeline.

    Args:
        spatial_size: Expected output spatial size

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("TEST 5: DataLoader Pipeline")
    print("=" * 60)

    # Create dummy dataset
    dummy_data_list = create_dummy_dataset(
        spatial_size=spatial_size,
        num_samples=4,
    )

    print(f"  Created {len(dummy_data_list)} dummy samples")

    # Get transforms
    train_transforms = get_train_transforms(spatial_size=spatial_size)
    val_transforms = get_val_transforms(spatial_size=spatial_size)

    # Create dataloaders
    train_loader, val_loader = get_dataloader(
        data_list=dummy_data_list,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        batch_size=2,
        num_workers=0,  # Use 0 for testing
        val_split=0.5,  # 50% for val
        shuffle=True,
        seed=42,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Test iterating
    for i, batch in enumerate(train_loader):
        images = batch["image"]
        labels = batch["label"]
        print(f"  Batch {i}: image shape {tuple(images.shape)}, label shape {tuple(labels.shape)}")

        if images.shape != (2, 1, *spatial_size):
            print(f"  ❌ Batch shape mismatch")
            return False

    print("  ✅ DataLoader pipeline test PASSED")
    return True


def main() -> None:
    """
    Run all Stage 1 pipeline tests.
    """
    print("\n" + "#" * 60)
    print("# ADynamics Stage 1 Pipeline Integration Test")
    print("#" * 60)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    spatial_size = (128, 128, 128)
    in_channels = 1
    latent_channels = 64

    # Track results
    results = []

    # Run tests
    results.append(("Transforms", test_transforms(spatial_size)))
    results.append(("VAE Forward Pass", test_vae_forward_pass(spatial_size, in_channels, latent_channels)))
    results.append(("VAE Training Step", test_vae_training_step(spatial_size, in_channels, latent_channels)))
    results.append(("NIfTI I/O", test_nifti_io(spatial_size)))
    results.append(("DataLoader", test_dataloader(spatial_size)))

    # Summary
    print("\n" + "#" * 60)
    print("# Test Summary")
    print("#" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ADynamics Stage 1 Pipeline is fully integrated and tested!")
    else:
        print("❌ Some tests failed. Please review the output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
