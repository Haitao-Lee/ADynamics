"""
ADynamics Stage 1 Pipeline Integration Test

This script validates that all Stage 1 components are correctly integrated:
- Data transforms (MONAI) with HD resolution (256x256x192)
- Dataset and DataLoader with stratified splitting
- 3D VAE model for HD inputs
- Training loop with AMP and backward pass

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from core_data.dataset import create_dummy_dataset, get_train_val_test_dataloaders, cleanup_dummy_dataset
from core_data.transforms import get_train_transforms, get_val_transforms
from engine.losses import total_vae_loss
from models.vae3d import ADynamicsVAE3D
from utils.io_utils import save_tensor_to_nifti


HD_SPATIAL_SIZE = (256, 256, 192)
HD_LATENT_SPATIAL = (16, 16, 12)


def test_transforms(spatial_size: tuple = HD_SPATIAL_SIZE) -> bool:
    """
    Test MONAI transforms produce correct output shape for HD.

    Args:
        spatial_size: Expected output spatial size (default: HD_SPATIAL_SIZE)

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("TEST 1: Data Transforms (HD)")
    print("=" * 60)

    dummy_data_list = create_dummy_dataset(
        spatial_size=spatial_size,
        num_samples=2,
    )

    try:
        train_transforms = get_train_transforms(spatial_size=spatial_size)

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
    finally:
        cleanup_dummy_dataset(dummy_data_list)


def test_vae_forward_pass(
    spatial_size: tuple = HD_SPATIAL_SIZE,
    in_channels: int = 1,
    latent_channels: int = 64,
) -> bool:
    """
    Test VAE forward pass with memory tracking for HD.

    Args:
        spatial_size: Input spatial size (HD: 256x256x192)
        in_channels: Number of input channels
        latent_channels: Latent space channels

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("TEST 2: VAE Forward Pass (HD)")
    print("=" * 60)

    model = ADynamicsVAE3D(
        spatial_size=spatial_size,
        in_channels=in_channels,
        latent_channels=latent_channels,
        base_channels=32,
    )
    model.eval()

    batch_size = 2
    dummy_input = torch.randn(batch_size, in_channels, *spatial_size)
    print(f"  Input shape: {tuple(dummy_input.shape)}")

    # Memory tracking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU memory before forward: {mem_before:.2f} MB")

    with torch.no_grad():
        recon, mu, logvar = model(dummy_input)

    print(f"  Reconstruction shape: {tuple(recon.shape)}")
    print(f"  Mu shape:            {tuple(mu.shape)}")
    print(f"  Logvar shape:        {tuple(logvar.shape)}")

    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1024**2
        print(f"  GPU memory after forward: {mem_after:.2f} MB")
        print(f"  GPU memory delta: {mem_after - mem_before:.2f} MB")

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

    expected_latent_spatial = HD_LATENT_SPATIAL
    if mu.shape[2:] != expected_latent_spatial:
        print(f"  ❌ Latent spatial shape mismatch: expected {expected_latent_spatial}, got {mu.shape[2:]}")
        checks_passed = False
    else:
        print(f"  ✅ Latent spatial shape correct: {mu.shape[2:]}")

    # Cleanup
    del model, dummy_input, recon, mu, logvar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if checks_passed:
        print("  ✅ VAE forward pass test PASSED")
    else:
        print("  ❌ VAE forward pass test FAILED")

    return checks_passed


def test_vae_training_step(
    spatial_size: tuple = HD_SPATIAL_SIZE,
    in_channels: int = 1,
    latent_channels: int = 64,
) -> bool:
    """
    Test VAE training step with AMP and backward pass.

    Uses total_vae_loss from engine.losses for consistency.

    Args:
        spatial_size: Input spatial size
        in_channels: Number of input channels
        latent_channels: Latent space channels

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("TEST 3: VAE Training Step with AMP (HD)")
    print("=" * 60)

    from torch.amp import autocast, GradScaler

    model = ADynamicsVAE3D(
        spatial_size=spatial_size,
        in_channels=in_channels,
        latent_channels=latent_channels,
        base_channels=32,
    )
    model.train()

    batch_size = 2
    dummy_input = torch.randn(batch_size, in_channels, *spatial_size)
    print(f"  Input shape: {tuple(dummy_input.shape)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler('cuda')
    config = {"kl_weight": 0.0001, "recon_loss_type": "l1"}

    # Forward pass with AMP
    with autocast('cuda'):
        recon, mu, logvar = model(dummy_input)
        loss = total_vae_loss(
            recon,
            dummy_input,
            mu,
            logvar,
            kl_weight=config["kl_weight"],
            recon_loss_type=config["recon_loss_type"],
        )

    print(f"  Total loss: {loss.item():.6f}")

    if torch.isnan(loss):
        print("  ❌ Loss is NaN!")
        return False

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

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

    print("  ✅ Backward pass successful with AMP")
    print("  ✅ Gradients computed correctly")
    print("  ✅ Optimizer step successful")
    print("  ✅ VAE training step test PASSED")

    # Cleanup
    del model, dummy_input, optimizer, scaler, loss, recon, mu, logvar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def test_nifti_io(spatial_size: tuple = HD_SPATIAL_SIZE) -> bool:
    """
    Test NIfTI save/load roundtrip with physical axis validation.

    Creates a non-symmetric tensor that can detect axis flips:
    - D dimension (depth): first 10 layers = 1.0, rest = 0.0
    - H dimension (height): first 20 rows = 0.5, rest = 0.0
    - W dimension (width): first 30 columns = 0.25, rest = 0.0

    This ensures that if axes are permuted during save/load, the test will fail.

    Args:
        spatial_size: Spatial size for test tensor

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("TEST 4: NIfTI I/O Roundtrip with Axis Validation (HD)")
    print("=" * 60)

    D, H, W = spatial_size

    # Create non-symmetric tensor to detect axis flips
    tensor = torch.zeros(D, H, W, dtype=torch.float32)

    # D dimension (depth): first 10 layers = 1.0
    tensor[:10, :, :] = 1.0

    # H dimension (height): first 20 rows = 0.5 (only in layers 10-20 to differentiate)
    tensor[10:20, :20, :] = 0.5

    # W dimension (width): first 30 columns = 0.25 (only in rows 20-30 to differentiate)
    tensor[20:30, 20:30, :30] = 0.25

    print(f"  Tensor shape: {tuple(tensor.shape)}")
    print(f"  D[:10,:,:].max() = {tensor[:10,:,:].max().item():.2f} (should be 1.0)")
    print(f"  D[10:20,:20,:].max() = {tensor[10:20,:20,:].max().item():.2f} (should be 0.5)")
    print(f"  D[20:30,20:30,:30].max() = {tensor[20:30,20:30,:30].max().item():.2f} (should be 0.25)")

    affine = np.eye(4, dtype=np.float64)

    temp_dir = tempfile.mkdtemp(prefix="adynamics_nifti_test_")
    temp_path = os.path.join(temp_dir, "test_output.nii.gz")

    try:
        save_tensor_to_nifti(tensor, affine, temp_path)
        print(f"  Saved to: {temp_path}")

        loaded_nii = nib.load(temp_path)
        loaded_data = loaded_nii.get_fdata()

        print(f"  Loaded shape: {tuple(loaded_data.shape)}")

        # Shape validation
        if loaded_data.shape != (D, H, W):
            print(f"  ❌ Shape mismatch: expected {(D, H, W)}, got {tuple(loaded_data.shape)}")
            return False

        # Physical axis validation: check that the "1.0" signal is in the correct D location
        # If axes were flipped, this would fail
        loaded_tensor = torch.from_numpy(loaded_data.astype(np.float32))

        # Check D dimension marker (should be at layer 0-9)
        d_marker_max = loaded_tensor[:10, :, :].max()
        h_marker_max = loaded_tensor[10:20, :20, :].max()
        w_marker_max = loaded_tensor[20:30, 20:30, :30].max()

        print(f"  Loaded D[:10,:,:].max() = {d_marker_max.item():.2f} (should be 1.0)")
        print(f"  Loaded D[10:20,:20,:].max() = {h_marker_max.item():.2f} (should be 0.5)")
        print(f"  Loaded D[20:30,20:30,:30].max() = {w_marker_max.item():.2f} (should be 0.25)")

        # Axis swap detection: if axes are flipped, these values will be wrong
        if abs(d_marker_max.item() - 1.0) > 0.01:
            print("  ❌ D-axis marker corrupted - possible axis swap!")
            return False
        if abs(h_marker_max.item() - 0.5) > 0.01:
            print("  ❌ H-axis marker corrupted - possible axis swap!")
            return False
        if abs(w_marker_max.item() - 0.25) > 0.01:
            print("  ❌ W-axis marker corrupted - possible axis swap!")
            return False

        # Data integrity check
        if not np.allclose(loaded_data, tensor.numpy(), rtol=1e-5):
            print("  ❌ Data mismatch after roundtrip")
            return False

        print("  ✅ NIfTI I/O test PASSED with axis validation")
        return True

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


def test_dataloader(spatial_size: tuple = HD_SPATIAL_SIZE) -> bool:
    """
    Test full dataloader pipeline with stratified splitting.

    Args:
        spatial_size: Expected output spatial size

    Returns:
        True if test passes
    """
    print("\n" + "=" * 60)
    print("TEST 5: DataLoader Pipeline with Stratified Split")
    print("=" * 60)

    dummy_data_list = create_dummy_dataset(
        spatial_size=spatial_size,
        num_samples=8,
    )

    try:
        print(f"  Created {len(dummy_data_list)} dummy samples")

        train_transforms = get_train_transforms(spatial_size=spatial_size)
        val_transforms = get_val_transforms(spatial_size=spatial_size)

        train_loader, val_loader, test_loader = get_train_val_test_dataloaders(
            data_list=dummy_data_list,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=val_transforms,
            batch_size=2,
            num_workers=0,
            train_split=0.7,
            val_split=0.15,
            shuffle=True,
            seed=42,
            use_cache=False,
        )

        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader) if test_loader else 0}")

        for i, batch in enumerate(train_loader):
            images = batch["image"]
            labels = batch["label"]
            print(f"  Batch {i}: image shape {tuple(images.shape)}, label shape {tuple(labels.shape)}")

            if images.shape != (2, 1, *spatial_size):
                print(f"  ❌ Batch shape mismatch")
                return False

        print("  ✅ DataLoader pipeline test PASSED")
        return True
    finally:
        cleanup_dummy_dataset(dummy_data_list)


def main() -> None:
    """
    Run all Stage 1 pipeline tests with HD configuration.
    """
    print("\n" + "#" * 60)
    print("# ADynamics Stage 1 Pipeline Integration Test (HD)")
    print("#" * 60)
    print(f"# Configuration: spatial_size={HD_SPATIAL_SIZE}, latent_spatial={HD_LATENT_SPATIAL}")

    torch.manual_seed(42)
    np.random.seed(42)

    spatial_size = HD_SPATIAL_SIZE
    in_channels = 1
    latent_channels = 64

    results = []

    results.append(("Transforms", test_transforms(spatial_size)))
    results.append(("VAE Forward Pass", test_vae_forward_pass(spatial_size, in_channels, latent_channels)))
    results.append(("VAE Training Step", test_vae_training_step(spatial_size, in_channels, latent_channels)))
    results.append(("NIfTI I/O", test_nifti_io(spatial_size)))
    results.append(("DataLoader", test_dataloader(spatial_size)))

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
        print("✅ ADynamics Stage 1 Pipeline (HD) is fully integrated and tested!")
    else:
        print("❌ Some tests failed. Please review the output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()