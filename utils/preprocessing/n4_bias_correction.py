from pathlib import Path
from typing import Optional, Sequence, Union, Iterable
import ants, traceback

def _stem_no_ext(p: Path) -> str:
    s = p.name
    if s.endswith(".nii.gz"): return s[:-7]
    if s.endswith(".nii"):    return s[:-4]
    return p.stem

def n4_bias_correction(
    in_nii: Union[str, Path],
    out_nii: Union[str, Path],
    mask_nii: Optional[Union[str, Path]] = None,
    *,
    shrink_factor: int = 2,
    bspline_fitting_distance: float = 200.0,   # 外部接口不变
    iters: Sequence[int] = (50, 50, 30, 20),
    tol: float = 1e-7,
    verbose: bool = True,
) -> None:
    in_nii  = Path(in_nii)
    out_nii = Path(out_nii)
    mask_nii = Path(mask_nii) if mask_nii is not None else None

    if verbose:
        print(f"[N4] input: {in_nii}")
        print(f"[N4] mask : {mask_nii if (mask_nii and mask_nii.exists()) else 'auto(get_mask)'}")

    img = ants.image_read(str(in_nii))
    mask = ants.image_read(str(mask_nii)) if (mask_nii and mask_nii.exists()) else ants.get_mask(img)

    # 兼容不同 ANTsPy 版本的参数名
    kwargs = dict(
        image=img,
        mask=mask,
        shrink_factor=shrink_factor,
        convergence={"iters": list(iters), "tol": tol},
    )
    try:
        n4 = ants.n4_bias_field_correction(**kwargs, spline_param=bspline_fitting_distance)
    except TypeError:
        n4 = ants.n4_bias_field_correction(**kwargs, bspline_fitting_distance=bspline_fitting_distance)

    out_nii.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(n4, str(out_nii))
    if verbose:
        print(f"[N4] output: {out_nii}")

def _iter_nii_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob("*.nii")
        yield from root.rglob("*.nii.gz")
    else:
        for p in root.iterdir():
            if p.is_file() and (str(p).endswith(".nii") or str(p).endswith(".nii.gz")):
                yield p

def n4_bias_correction_batch(
    in_dir: Union[str, Path],
    out_dir: Union[str, Path],
    mask_dir: Optional[Union[str, Path]] = None,
    *,
    suffix: str = "_n4.nii.gz",
    shrink_factor: int = 2,
    bspline_fitting_distance: float = 200.0,
    iters: Sequence[int] = (50, 50, 30, 20),
    tol: float = 1e-7,
    overwrite: bool = False,
    recursive: bool = False,
    keep_tree: bool = False,
    verbose: bool = True,
) -> None:
    in_dir  = Path(in_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(mask_dir) if mask_dir is not None else None

    nii_list = list(_iter_nii_files(in_dir, recursive))
    if verbose:
        print(f"[N4-BATCH]Found {len(nii_list)} files under {in_dir} (recursive={recursive}).")

    for src in nii_list:
        stem = _stem_no_ext(src)
        # 输出路径
        if keep_tree:
            rel  = src.relative_to(in_dir).parent
            out_subdir = out_dir / rel
            out_subdir.mkdir(parents=True, exist_ok=True)
            dst = out_subdir / f"{stem}{suffix}"
        else:
            dst = out_dir / f"{stem}{suffix}"

        if dst.exists() and not overwrite:
            if verbose: print(f"[Skip] exists: {dst}")
            continue

        # 新逻辑：mask_dir 下寻找“同名文件”
        mask_nii = None
        if mask_dir:
            same_name = mask_dir / src.name
            if same_name.exists():
                mask_nii = same_name
            else:
                if verbose:
                    print(f"[Warn] mask not found for {src.name} under {mask_dir}; will use auto(get_mask).")

        try:
            n4_bias_correction(
                in_nii=src,
                out_nii=dst,
                mask_nii=mask_nii,
                shrink_factor=shrink_factor,
                bspline_fitting_distance=bspline_fitting_distance,
                iters=iters,
                tol=tol,
                verbose=verbose
            )
        except Exception as e:
            print(f"[ERR] {src} -> {dst}: {e}")
            traceback.print_exc()
