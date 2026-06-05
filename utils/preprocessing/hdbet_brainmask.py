"""HD-BET brain extraction wrapper for ADynamics preprocessing.

Thin Python wrapper around the hd-bet CLI (https://github.com/MIC-DKFZ/HD-BET).
Handles both single-file and batch (directory) inputs and supports CPU/GPU.

Note: hd-bet must be installed separately and on PATH. See install_env.ps1
for the recommended installation command.
"""

from pathlib import Path
import shutil
import subprocess
from typing import Optional


def run_hd_bet(
    in_path: Path,
    out_path: Path,
    device: str = "cuda:0",
    extra_args: Optional[list] = None,
) -> None:
    """
    Invoke the hd-bet CLI on a single NIfTI file or a directory of files.

    Args:
        in_path:  Input NIfTI file, or a directory containing multiple NIfTI files
                  (will be processed in batch).
        out_path: Output NIfTI path, or output directory. The directory is
                  created with parents=True if it does not exist.
        device:   'cuda:0' (default) or 'cpu'.
        extra_args: Optional list of extra arguments to pass to hd-bet,
                    e.g. ['-tta', '-mode', 'fast'].

    Raises:
        RuntimeError: if hd-bet is not installed or not on PATH.
        subprocess.CalledProcessError: if hd-bet exits with a non-zero status.
    """
    if shutil.which("hd-bet") is None:
        raise RuntimeError(
            "hd-bet executable not found on PATH. "
            "Install it first: see install_env.ps1 or visit "
            "https://github.com/MIC-DKFZ/HD-BET for instructions."
        )

    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "hd-bet",
        "-i", str(in_path),
        "-o", str(out_path),
        "-device", device,
    ]
    if extra_args:
        cmd.extend(extra_args)

    subprocess.run(cmd, check=True)

