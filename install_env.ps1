# ADynamics Environment Installation Script
# For RTX 3090 (CUDA 12.1) / RTX 3090 / A100 / Similar GPUs
# Author: ADynamics Development Team
# Last Updated: 2026-04-22

#Requires -RunAsAdministrator

param(
    [switch]$CPUOnly = $false,
    [string]$EnvName = "ADynamics",
    [string]$PythonVersion = "3.11"
)

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  ADynamics Environment Installation Script" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Detect GPU
Write-Host "[INFO] Detecting GPU..." -ForegroundColor Yellow
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    if ($gpuInfo) {
        $gpuName = ($gpuInfo -split ",")[0].Trim()
        $gpuMem = ($gpuInfo -split ",")[1].Trim()
        Write-Host "[OK] Found GPU: $gpuName ($gpuMem)" -ForegroundColor Green
        $useGPU = $true
    }
} catch {
    Write-Host "[WARN] No NVIDIA GPU detected, using CPU mode" -ForegroundColor Yellow
    $useGPU = $false
}

# Check conda
Write-Host "[INFO] Checking conda..." -ForegroundColor Yellow
$condaPath = & conda info --base 2>$null
if (-not $condaPath) {
    Write-Host "[ERROR] Conda not found. Please install Miniconda or Anaconda first." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Conda found at: $condaPath" -ForegroundColor Green

# Step 1: Create conda environment
Write-Host ""
Write-Host "[STEP 1] Creating conda environment '$EnvName' with Python $PythonVersion..." -ForegroundColor Cyan

$envExists = & conda env list 2>$null | Select-String "^$EnvName\s"
if ($envExists) {
    Write-Host "[WARN] Environment '$EnvName' already exists. Removing..." -ForegroundColor Yellow
    & conda env remove -n $EnvName -y 2>$null
}

& conda create -n $EnvName python=$PythonVersion -y 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to create environment" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Environment '$EnvName' created" -ForegroundColor Green

# Step 2: Activate environment
Write-Host ""
Write-Host "[STEP 2] Activating environment..." -ForegroundColor Cyan
$envActivate = "conda activate $EnvName"

# Step 3: Install Core Framework (PyTorch + CUDA)
Write-Host ""
Write-Host "[STEP 3] Installing Core Framework (PyTorch + CUDA)..." -ForegroundColor Cyan

if ($CPUOnly) {
    Write-Host "[INFO] Installing PyTorch CPU version" -ForegroundColor Yellow
    & conda install -n $EnvName pytorch torchvision torchaudio cpuonly -c pytorch -y 2>&1 | Out-Null
} else {
    Write-Host "[INFO] Installing PyTorch with CUDA 12.1 support" -ForegroundColor Yellow
    & conda install -n $EnvName pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y 2>&1 | Out-Null
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Conda install failed, trying pip..." -ForegroundColor Yellow
    if ($CPUOnly) {
        & pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    } else {
        & pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    }
}

Write-Host "[OK] Core Framework installed" -ForegroundColor Green

# Step 4: Install Medical Imaging packages
Write-Host ""
Write-Host "[STEP 4] Installing Medical Imaging packages..." -ForegroundColor Cyan

# MONAI - use specific version for stability
& pip install --no-cache-dir monai==1.4.0 2>&1 | Out-Null

# nibabel - NIfTI I/O
& pip install --no-cache-dir "nibabel>=5.0.0,<6.0.0" 2>&1 | Out-Null

# SimpleITK - Image registration
& pip install --no-cache-dir SimpleITK>=2.2.0 2>&1 | Out-Null

Write-Host "[OK] Medical Imaging packages installed" -ForegroundColor Green

# Step 5: Install Math & ODE packages
Write-Host ""
Write-Host "[STEP 5] Installing Math & ODE packages..." -ForegroundColor Cyan

# torchdiffeq - ODE integration for CFM
& pip install --no-cache-dir torchdiffeq>=0.2.5 2>&1 | Out-Null

Write-Host "[OK] Math & ODE packages installed" -ForegroundColor Green

# Step 6: Install Data & Utils packages
Write-Host ""
Write-Host "[STEP 6] Installing Data & Utils packages..." -ForegroundColor Cyan

# Core data science packages - pin numpy<2.0 for compatibility
& pip install --no-cache-dir "numpy<2.0,scipy, pandas, pyyaml, matplotlib, tqdm, einops" 2>&1 | Out-Null

# scikit-learn for metrics
& pip install --no-cache-dir scikit-learn>=1.3.0 2>&1 | Out-Null

Write-Host "[OK] Data & Utils packages installed" -ForegroundColor Green

# Step 7: Install Optional/Development packages
Write-Host ""
Write-Host "[STEP 7] Installing Optional/Development packages..." -ForegroundColor Cyan

& pip install --no-cache-dir jupyterlab black ruff pytest ipykernel 2>&1 | Out-Null

Write-Host "[OK] Optional packages installed" -ForegroundColor Green

# Step 8: Verify installation
Write-Host ""
Write-Host "[STEP 8] Verifying installation..." -ForegroundColor Cyan

$verifyScript = @"
import sys
print('Python:', sys.version)

try:
    import torch
    print('PyTorch:', torch.__version__)
    print('CUDA Available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA Version:', torch.version.cuda)
        print('GPU Count:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
except ImportError as e:
    print('ERROR importing torch:', e)

try:
    import monai
    print('MONAI:', monai.__version__)
except ImportError as e:
    print('ERROR importing monai:', e)

try:
    import nibabel
    print('nibabel:', nibabel.__version__)
except ImportError as e:
    print('ERROR importing nibabel:', e)

try:
    import torchdiffeq
    print('torchdiffeq: installed')
except ImportError as e:
    print('ERROR importing torchdiffeq:', e)

try:
    import numpy
    print('numpy:', numpy.__version__)
except ImportError as e:
    print('ERROR importing numpy:', e)
"@

$verifyResult = & conda run -n $EnvName python -c $verifyScript 2>&1
Write-Host $verifyResult

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  Installation Complete!" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Yellow
Write-Host "  conda activate $EnvName" -ForegroundColor White
Write-Host ""
Write-Host "To verify, run:" -ForegroundColor Yellow
Write-Host "  conda activate $EnvName && python -c \"import torch; print(torch.cuda.is_available())\"" -ForegroundColor White
Write-Host ""
