"""
NIfTI I/O utilities for ADynamics.

Provides functions for reading and writing NIfTI files with proper
affine handling to ensure correct orientation in 3D Slicer and other viewers.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


def load_nifti(
    filepath: Union[str, Path],
    dtype: Optional[np.dtype] = None,
) -> Tuple[NDArray[np.float32], NDArray[np.float64]]:
    """
    Load a NIfTI file and return the data array and affine matrix.

    Args:
        filepath: Path to the NIfTI file (.nii or .nii.gz)
        dtype: Optional numpy dtype to convert the data to. If None, uses float32.

    Returns:
        Tuple of (data, affine) where:
            - data: 3D or 4D numpy array of the image data
            - affine: 4x4 affine transformation matrix

    Raises:
        FileNotFoundError: If the file does not exist
        nibabel.loadsave.ImageFileError: If the file cannot be loaded
    """
    filepath = str(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NIfTI file not found: {filepath}")

    # Load NIfTI image
    nii_img = nib.load(filepath)

    # Get data as numpy array
    data = nii_img.get_fdata(dtype=dtype if dtype is not None else np.float32)

    # Get affine matrix
    affine = nii_img.affine

    return data, affine


def save_tensor_to_nifti(
    tensor: Tensor,
    affine: Union[NDArray[np.float64], Tensor],
    filename: Union[str, Path],
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Save a PyTorch tensor as a NIfTI file.

    Converts a PyTorch tensor [D, H, W] or [B, 1, D, H, W] to a NIfTI file
    with proper orientation. The resulting file can be opened in 3D Slicer
    with correct spatial orientation.

    Args:
        tensor: PyTorch Tensor of shape [D, H, W] or [B, 1, D, H, W]
            Values should be in range [0, 1] for proper visualization
        affine: 4x4 affine transformation matrix (numpy array or tensor).
            If None, identity matrix is used with optional voxel_size.
        filename: Output path for the NIfTI file (.nii or .nii.gz)
        voxel_size: Optional tuple of (dx, dy, dz) voxel sizes in mm.
            Used to construct identity affine if affine is None.

    Raises:
        ValueError: If tensor has invalid shape
        RuntimeError: If save operation fails

    Example:
        >>> # Save a single 3D tensor
        >>> tensor = torch.rand(128, 128, 128)
        >>> affine = np.eye(4)
        >>> save_tensor_to_nifti(tensor, affine, "output.nii.gz")

        >>> # Save a batch tensor (takes first sample)
        >>> tensor = torch.rand(2, 1, 128, 128, 128)
        >>> save_tensor_to_nifti(tensor, None, "output.nii.gz", voxel_size=(1.0, 1.0, 1.0))
    """
    # Handle filename
    filename = str(filename)
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

    # Convert tensor to numpy
    if isinstance(tensor, Tensor):
        # Detach from computation graph and move to CPU
        tensor = tensor.detach().cpu()

        # Handle batch dimension
        if tensor.dim() == 5:
            # Batch dimension: [B, 1, D, H, W] -> take first sample
            tensor = tensor[0, 0]  # shape: [D, H, W]
        elif tensor.dim() == 4:
            # Assume [1, D, H, W] -> remove channel dim
            tensor = tensor[0]  # shape: [D, H, W]
        elif tensor.dim() != 3:
            raise ValueError(
                f"Tensor must be 3D [D, H, W], 4D [1, D, H, W], or 5D [B, 1, D, H, W], "
                f"got {tensor.dim()}D with shape {tensor.shape}"
            )

        # Convert to numpy
        data = tensor.numpy()
    else:
        data = tensor

    # Ensure data is contiguous and float32
    data = np.ascontiguousarray(data, dtype=np.float32)

    # Handle affine matrix
    if affine is None:
        # Create identity affine with optional voxel size
        if voxel_size is None:
            voxel_size = (1.0, 1.0, 1.0)
        affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])
    elif isinstance(affine, Tensor):
        affine = affine.cpu().numpy()

    # Ensure affine is the correct shape
    if affine.shape != (4, 4):
        raise ValueError(f"Affine matrix must be 4x4, got shape {affine.shape}")

    # Create NIfTI image
    # Using RAS orientation (Right-Anterior-Superior) which is standard for 3D Slicer
    nii_img = nib.Nifti1Image(data, affine)

    # Save to file
    try:
        nib.save(nii_img, filename)
    except Exception as e:
        raise RuntimeError(f"Failed to save NIfTI file to {filename}: {e}") from e


def tensor_to_nifti_data(
    tensor: Tensor,
) -> NDArray[np.float32]:
    """
    Convert a PyTorch tensor to NIfTI-compatible numpy array.

    Args:
        tensor: PyTorch Tensor of shape [D, H, W] or [B, 1, D, H, W]

    Returns:
        3D numpy array of shape [D, H, W]
    """
    if isinstance(tensor, Tensor):
        tensor = tensor.detach().cpu()

        if tensor.dim() == 5:
            tensor = tensor[0, 0]
        elif tensor.dim() == 4:
            tensor = tensor[0]

        data = tensor.numpy()
    else:
        data = tensor

    return np.ascontiguousarray(data, dtype=np.float32)


def create_identity_affine(
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> NDArray[np.float64]:
    """
    Create a 4x4 identity-like affine matrix with specified voxel sizes and origin.

    This creates an affine matrix in RAS+ orientation suitable for 3D Slicer.

    Args:
        voxel_size: Voxel dimensions in mm (dx, dy, dz)
        origin: Origin position in mm (x, y, z)

    Returns:
        4x4 affine matrix as numpy array
    """
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = voxel_size[0]  # x voxel size
    affine[1, 1] = voxel_size[1]  # y voxel size
    affine[2, 2] = voxel_size[2]  # z voxel size
    affine[0, 3] = origin[0]
    affine[1, 3] = origin[1]
    affine[2, 3] = origin[2]
    return affine


def resample_nifti(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolation: str = "linear",
) -> None:
    """
    Resample a NIfTI file to target voxel spacing.

    Args:
        input_path: Path to input NIfTI file
        output_path: Path to output NIfTI file
        target_spacing: Target voxel spacing in mm (dx, dy, dz)
        interpolation: Interpolation method ("linear", "nearest", "cubic")
    """
    import scipy.ndimage as ndimage

    # Load image
    nii = nib.load(str(input_path))
    data = nii.get_fdata()
    affine = nii.affine

    # Get current spacing from affine
    current_spacing = np.abs(np.diag(affine[:3, :3]))

    # Calculate zoom factors
    zoom = current_spacing / np.array(target_spacing)

    # Resample
    order = {"linear": 1, "nearest": 0, "cubic": 3}.get(interpolation, 1)
    resampled_data = ndimage.zoom(data, zoom, order=order)

    # Create new affine with target spacing
    new_affine = affine.copy()
    new_affine[:3, :3] = np.diag(target_spacing)

    # Save
    resampled_nii = nib.Nifti1Image(resampled_data, new_affine)
    nib.save(resampled_nii, str(output_path))
