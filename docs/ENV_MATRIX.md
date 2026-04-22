# ADynamics Environment Matrix

**Document Version:** 1.0.0
**Last Updated:** 2026-04-22
**Target Hardware:** NVIDIA RTX 3090 (24GB) x2 / Similar GPUs with CUDA 12.x
**Python Version:** 3.11
**Conda Environment:** `ADynamics`

---

## Hardware Detection Summary

| Property | Value |
|----------|-------|
| GPU Model | NVIDIA GeForce RTX 3090 |
| VRAM | 24GB per GPU |
| Compute Capability | 8.6 (Ampere) |
| CUDA Support | 12.x (12.1, 12.4) |
| Driver Check | `nvidia-smi --query-gpu=name,memory.total --format=csv` |

---

## Environment Matrix Table

### Category 1: Core Framework

| 库名称 | 推荐版本 | 安装命令 (精确到版本号和源) | 作用说明 |
|--------|----------|---------------------------|----------|
| `pytorch` | 2.5.1 | `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y` | PyTorch核心框架，深度学习训练与推理引擎 |
| `torchvision` | 0.20.1 | (同上，与pytorch一起安装) | 计算机视觉工具库，图像预处理 |
| `torchaudio` | 2.5.1 | (同上，与pytorch一起安装) | 音频处理工具（本项目间接依赖） |
| `pytorch-cuda` | 12.1 | (同上) | CUDA 12.1 支持，让PyTorch在RTX 3090上使用GPU加速 |

**备选方案 (CPU Only):**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### Category 2: Medical Imaging

| 库名称 | 推荐版本 | 安装命令 (精确到版本号和源) | 作用说明 |
|--------|----------|---------------------------|----------|
| `monai` | 1.4.0 | `pip install monai==1.4.0` | 医学影像深度学习框架，提供预处理transforms和预训练模型 |
| `nibabel` | 5.3.x | `pip install "nibabel>=5.0.0,<6.0.0"` | NIfTI格式读写，支持3D医学影像I/O |
| `SimpleITK` | 2.3.x | `pip install SimpleITK>=2.2.0` | 医学图像分割与配准，支持多格式转换 |

### Category 3: Math & ODE

| 库名称 | 推荐版本 | 安装命令 (精确到版本号和源) | 作用说明 |
|--------|----------|---------------------------|----------|
| `torchdiffeq` | 0.2.5 | `pip install torchdiffeq>=0.2.5` | ODE求解器，用于CFM流匹配的Euler积分和RK4积分 |

### Category 4: Data & Utils

| 库名称 | 推荐版本 | 安装命令 (精确到版本号和源) | 作用说明 |
|--------|----------|---------------------------|----------|
| `numpy` | 1.26.x | `pip install "numpy<2.0"` | 数值计算，数组操作（pin numpy<2.0因monai兼容性） |
| `scipy` | 1.14.x | `pip install scipy` | 科学计算，插值与优化 |
| `pandas` | 2.2.x | `pip install pandas` | 数据分析，表格数据处理 |
| `scikit-learn` | 1.6.x | `pip install scikit-learn>=1.3.0` | 机器学习工具，指标计算 |
| `pyyaml` | 6.0.x | `pip install pyyaml>=6.0` | YAML配置文件读写 |
| `matplotlib` | 3.9.x | `pip install matplotlib>=3.7.0` | 数据可视化，绘制训练曲线 |
| `tqdm` | 4.66.x | `pip install tqdm>=4.65.0` | 进度条，训练过程可视化 |
| `einops` | 0.8.0 | `pip install einops==0.8.0` | 张量重塑，简化代码 |

---

## Installation Execution Log

### 完整安装命令序列 (按执行顺序)

```powershell
# 1. 创建环境
conda create -n ADynamics python=3.11 -y

# 2. 激活环境
conda activate ADynamics

# 3. 安装Core Framework (PyTorch + CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. 安装Medical Imaging
pip install monai==1.4.0
pip install "nibabel>=5.0.0,<6.0.0"
pip install SimpleITK>=2.2.0

# 5. 安装Math & ODE
pip install torchdiffeq>=0.2.5

# 6. 安装Data & Utils
pip install "numpy<2.0" scipy pandas scikit-learn pyyaml matplotlib tqdm einops

# 7. 安装可选开发包
pip install jupyterlab black ruff pytest ipykernel
```

### 一键运行安装脚本

```powershell
# 方法1: 直接运行PowerShell脚本
.\install_env.ps1

# 方法2: 指定CPU模式 (无GPU时)
.\install_env.ps1 -CPUOnly

# 方法3: 自定义环境名和Python版本
.\install_env.ps1 -EnvName "ADynamics" -PythonVersion "3.11"
```

---

## Environment Verification

### 验证脚本

```python
import sys
print('Python:', sys.version)

import torch
print('PyTorch:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Version:', torch.version.cuda)
    print('GPU Count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

import monai
print('MONAI:', monai.__version__)

import nibabel
print('nibabel:', nibabel.__version__)

import torchdiffeq
print('torchdiffeq: installed')

import numpy
print('numpy:', numpy.__version__)
```

### 预期输出

```
Python: 3.11.9
PyTorch: 2.5.1
CUDA Available: True
CUDA Version: 12.1
GPU Count: 2
  GPU 0: NVIDIA GeForce RTX 3090
  GPU 1: NVIDIA GeForce RTX 3090
MONAI: 1.4.0
nibabel: 5.3.1
torchdiffeq: installed
numpy: 1.26.4
```

---

## Export Current Environment

### 导出当前环境到YAML

```bash
conda env export > environment.yml

# 或者只导出显式安装的包（不含依赖）
conda env export --from-history > environment.yml
```

### 从YAML重建环境

```bash
conda env create -f environment.yml
```

---

## Known Compatibility Notes

### numpy<2.0 原因
MONAI 1.4.0 在某些transforms中依赖numpy 1.x API，numpy 2.0有breaking changes。建议固定在1.26.x版本。

### PyTorch CUDA版本选择
- RTX 3090 (Ampere, compute capability 8.6) 支持CUDA 12.x
- 推荐使用CUDA 12.1，比12.4更稳定
- 不推荐CUDA 11.x，会损失部分新GPU特性

### torchdiffeq注意事项
- 版本0.2.5与PyTorch 2.x兼容
- 用于CFM的Euler积分和轨迹计算

---

## Troubleshooting

### 问题1: conda安装pytorch失败

```bash
# 使用pip作为备选
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 问题2: monai安装失败

```bash
# 确保pip是最新的
pip install --upgrade pip
pip install monai==1.4.0
```

### 问题3: nibabel读取某些NIfTI报错

```bash
# 安装完整版（包含所有压缩支持）
pip install "nibabel[compression]>=5.0.0,<6.0.0"
```

---

**Maintainer:** ADynamics Development Team
**License:** MIT
