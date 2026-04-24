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


def _format_tensor_for_nifti(
    tensor: Tensor,
    permute_to_xyz: bool = False,
) -> Tuple[NDArray[np.float32], bool]:
    """
    Convert a PyTorch tensor to NIfTI-compatible numpy array with proper channel handling.

    Handles multi-channel deformation fields correctly by preserving all channels
    and transposing to NIfTI's [D, H, W, C] convention for multi-channel data.

    Args:
        tensor: PyTorch Tensor of shape [D, H, W], [C, D, H, W], [1, D, H, W],
                [B, 1, D, H, W], or [B, C, D, H, W]
        permute_to_xyz: If True, permute axes from [D, H, W] to [X, Y, Z] for
                        physical coordinate alignment

    Returns:
        Tuple of (data_array, is_multichannel) where:
            - data_array: numpy array in NIfTI format ([D, H, W] or [D, H, W, C])
            - is_multichannel: True if original had multiple channels

    Raises:
        ValueError: If tensor has invalid dimensions
    """
    if isinstance(tensor, Tensor):
        tensor = tensor.detach().cpu()

    original_shape = tensor.shape

    if tensor.dim() == 5:
        # Batch dimension: [B, C, D, H, W] -> [C, D, H, W] (take first sample)
        tensor = tensor[0]
    elif tensor.dim() == 4:
        # Could be [1, D, H, W] or [C, D, H, W]
        pass
    elif tensor.dim() == 3:
        # Single channel 3D: [D, H, W]
        pass
    else:
        raise ValueError(
            f"Tensor must be 3D-5D, got {tensor.dim()}D with shape {original_shape}"
        )

    is_multichannel = tensor.shape[0] > 1

    if is_multichannel:
        # Multi-channel data (e.g., 3-channel deformation field [C, D, H, W])
        # NIfTI convention: channels last -> [D, H, W, C]
        channels = tensor.shape[0]
        data = tensor.numpy()
        data = np.transpose(data, (1, 2, 3, 0))  # [C, D, H, W] -> [D, H, W, C]
    else:
        # Single channel: squeeze channel dimension -> [D, H, W]
        data = tensor.squeeze(0).numpy()

    # Optionally permute DHW -> XYZ for physical coordinate alignment
    if permute_to_xyz:
        if is_multichannel:
            # [D, H, W, C] -> [X, Y, Z, C] where X=W, Y=H, Z=D
            data = np.transpose(data, (2, 1, 0, 3))
        else:
            # [D, H, W] -> [X, Y, Z] where X=W, Y=H, Z=D
            data = np.transpose(data, (2, 1, 0))

    return np.ascontiguousarray(data, dtype=np.float32), is_multichannel


def save_tensor_to_nifti(
    tensor: Tensor,
    affine: Union[NDArray[np.float64], Tensor, None],
    filename: Union[str, Path],
    voxel_size: Optional[Tuple[float, float, float]] = None,
    permute_to_xyz: bool = False,
) -> None:
    """
    Save a PyTorch tensor as a NIfTI file.

    Converts a PyTorch tensor to a NIfTI file with proper orientation.
    Handles multi-channel deformation fields correctly by preserving all channels.
    The resulting file can be opened in 3D Slicer with correct spatial orientation.

    Args:
        tensor: PyTorch Tensor of shape [D, H, W], [C, D, H, W], [1, D, H, W],
                [B, 1, D, H, W], or [B, C, D, H, W]
        affine: 4x4 affine transformation matrix (numpy array or tensor).
                If None, identity matrix is used with optional voxel_size.
        filename: Output path for the NIfTI file (.nii or .nii.gz)
        voxel_size: Optional tuple of (dx, dy, dz) voxel sizes in mm.
                    Used to construct identity affine if affine is None.
        permute_to_xyz: If True, permute axes from [D, H, W] to [X, Y, Z]
                        for physical coordinate alignment in 3D Slicer.

    Raises:
        ValueError: If tensor has invalid shape or affine is invalid
        RuntimeError: If save operation fails

    Example:
        >>> # Save a single 3D tensor
        >>> tensor = torch.rand(128, 128, 128)
        >>> affine = np.eye(4)
        >>> save_tensor_to_nifti(tensor, affine, "output.nii.gz")

        >>> # Save a 3-channel deformation field
        >>> tensor = torch.rand(3, 128, 128, 128)  # [C, D, H, W]
        >>> save_tensor_to_nifti(tensor, affine, "deform.nii.gz")

        >>> # Save with XYZ permutation
        >>> tensor = torch.rand(128, 128, 128)
        >>> save_tensor_to_nifti(tensor, None, "output.nii.gz", permute_to_xyz=True)
    """
    filename = str(filename)
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

    # Use the helper to properly format tensor for NIfTI
    data, is_multichannel = _format_tensor_for_nifti(tensor, permute_to_xyz=permute_to_xyz)

    # Handle affine matrix
    if affine is None:
        if voxel_size is None:
            voxel_size = (1.0, 1.0, 1.0)
        affine = np.diag([voxel_size[0], voxel_size[1], voxel_size[2], 1.0])
    elif isinstance(affine, Tensor):
        affine = affine.cpu().numpy()

    if affine.shape != (4, 4):
        raise ValueError(f"Affine matrix must be 4x4, got shape {affine.shape}")

    # Create NIfTI image with RAS orientation
    nii_img = nib.Nifti1Image(data, affine)

    try:
        nib.save(nii_img, filename)
    except Exception as e:
        raise RuntimeError(f"Failed to save NIfTI file to {filename}: {e}") from e


def tensor_to_nifti_data(
    tensor: Tensor,
    permute_to_xyz: bool = False,
) -> NDArray[np.float32]:
    """
    Convert a PyTorch tensor to NIfTI-compatible numpy array.

    Args:
        tensor: PyTorch Tensor of shape [D, H, W], [C, D, H, W], [1, D, H, W],
                [B, 1, D, H, W], or [B, C, D, H, W]
        permute_to_xyz: If True, permute axes from [D, H, W] to [X, Y, Z]

    Returns:
        Numpy array in NIfTI format ([D, H, W] or [D, H, W, C])
    """
    data, _ = _format_tensor_for_nifti(tensor, permute_to_xyz=permute_to_xyz)
    return data


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
    Resample a NIfTI file to target voxel spacing while preserving rotation/shear.

    Computes the new affine matrix by scaling the original column vectors
    proportionally, rather than replacing with a diagonal matrix. This
    preserves any rotation or shear components in the original affine.

    Args:
        input_path: Path to input NIfTI file
        output_path: Path to output NIfTI file
        target_spacing: Target voxel spacing in mm (dx, dy, dz)
        interpolation: Interpolation method ("linear", "nearest", "cubic")

    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If target_spacing is invalid
        RuntimeError: If resampling operation fails
    """
    import scipy.ndimage as ndimage

    input_path = str(input_path)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input NIfTI file not found: {input_path}")

    # Load image
    nii = nib.load(input_path)
    data = nii.get_fdata()
    affine = nii.affine

    # Compute current spacing from the column vectors of the affine matrix
    # Each column vector's magnitude represents the voxel size in that direction
    current_spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

    # Calculate zoom factors for resampling
    zoom = current_spacing / np.array(target_spacing)

    # Resample data
    order = {"linear": 1, "nearest": 0, "cubic": 3}.get(interpolation, 1)
    resampled_data = ndimage.zoom(data, zoom, order=order)

    # Compute scale factor for each axis
    scale_factor = np.array(target_spacing) / current_spacing

    # Create new affine by scaling original column vectors proportionally
    # This preserves any rotation or shear in the original affine
    new_affine = affine.copy()
    new_affine[:3, :3] = affine[:3, :3] * scale_factor

    # Save resampled image
    resampled_nii = nib.Nifti1Image(resampled_data, new_affine)
    try:
        nib.save(resampled_nii, str(output_path))
    except Exception as e:
        raise RuntimeError(f"Failed to save resampled NIfTI to {output_path}: {e}") from e