from pathlib import Path
import shutil
import subprocess
from typing import Optional
from utils.denoise import batch_denoise_dir_antspy
from utils.n4_bias_correction import n4_bias_correction_batch


def run_hd_bet(
    in_path: Path,
    out_path: Path,
    device: str = "cuda:0",
    extra_args: Optional[list] = None,
) -> None: 
    """
    以 Python 调用 hd-bet CLI，可作用于“文件或目录”。
    - in_path/out_path: 既可为单个 NIfTI 文件，也可为目录（目录时批量处理）
    - device: 例如 'cuda:0' 或 'cpu'
    - extra_args: 传递给 hd-bet 的其他参数列表（如 ['-tta', '-mode', 'fast']）
    """
    if shutil.which("hd-bet") is None:
        raise RuntimeError("未找到 'hd-bet' 可执行程序，请确认已正确安装并在 PATH 中。")

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
        cmd += list(map(str, extra_args))

    print(f"[hd-bet] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(f"hd-bet 运行失败（返回码 {res.returncode}）。详见上方输出。")