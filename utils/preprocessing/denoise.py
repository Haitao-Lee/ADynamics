import os
# [CRITICAL FIX 1]: Must limit threads BEFORE importing ants!
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import ants
from typing import Optional, List

def denoise_single_t1_antspy(
    in_nii: str,
    out_nii: str,
    brain_mask_nii: Optional[str] = None,
    noise_model: str = "Rician",    # "Rician" | "Gaussian"
    verbose: bool = True
):
    """
    Denoise a single 3D T1 MRI image.
    """
    # Read image and [CRITICAL FIX 2]: Force clone to float type to prevent crashes
    img = ants.image_read(in_nii).clone("float")

    # Read brain mask if available, also force to float (matrix multiplication requires consistent types)
    mask = None
    if brain_mask_nii is not None and os.path.exists(brain_mask_nii):
        mask = ants.image_read(brain_mask_nii).clone("float")

    # if verbose:
        # print(f"0[OK] Ready to denoise: {in_nii}")

    # Denoise (ANTsPy uses non-local means/patch-based approach internally, Rician for MRI)
    den = ants.denoise_image(
        image=img,
        mask=mask,
        noise_model=noise_model
    )

    # if verbose:
        # print(f"1[OK] Denoised successfully in memory")

    # Save image
    ants.image_write(den, out_nii)

    if verbose:
        print(f"[OK] Saved to: {out_nii}")


def batch_denoise_dir_antspy(
    in_dir: str,
    out_dir: str,
    mask_dir: Optional[str] = None,
    suffix: str = "_den.nii.gz",
    noise_model: str = "Rician"
):
    """
    Batch denoise all .nii / .nii.gz files in a directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".nii.gz"):
            continue

        # print(f"\n--- Processing file: {name} ---")
        in_nii = os.path.join(in_dir, name)
        stem = name.replace(".nii.gz", "")
        out_nii = os.path.join(out_dir, f"{stem}{suffix}")

        # Skip if already denoised (resume logic)
        if os.path.exists(out_nii):
            print(f"Skipping (already exists): {out_nii}")
            continue

        mask_nii = None
        if mask_dir:
            cand1 = os.path.join(mask_dir, f"{stem}_mask.nii.gz")
            cand2 = os.path.join(mask_dir, f"{stem}_brain_mask.nii.gz")
            cand3 = os.path.join(mask_dir, f"{stem}.mask.nii.gz")
            for c in (cand1, cand2, cand3):
                if os.path.exists(c):
                    mask_nii = c
                    break

        denoise_single_t1_antspy(
            in_nii=in_nii,
            out_nii=out_nii,
            brain_mask_nii=mask_nii,
            noise_model=noise_model
        )
