# s03-cfm.ps1
# ===========
# Stage 3: MMSE-Conditional Flow Matching in latent space.
# Input:  ./checkpoints/stage1/vae_best.pt
# Output: ./checkpoints/stage3_cfm/cfm_best.pt
# Usage:  .\s03-cfm.ps1

param(
    [int]$Epochs     = 300,
    [int]$BatchSize  = 16,
    [float]$LR       = 1e-4,
    [int]$NumGPUs    = 2,
    [int]$Patience   = 50
)

Import-Module "$PSScriptRoot\scripts\ps\common.psm1" -Force
Initialize-ADynamicsEnv

$Stage1Ckpt = Join-Path $PSScriptRoot "checkpoints\stage1\vae_best.pt"
if (-not (Test-Path $Stage1Ckpt)) {
    Write-Host "Missing: $Stage1Ckpt  (run s01-train.ps1 first)" -ForegroundColor Red; exit 1
}

$Py = Get-PythonExe

& $Py -u scripts/train_stage3_cfm.py `
    --config ./configs/stage3_cfm.yaml `
    --encoder_checkpoint $Stage1Ckpt `
    --output_dir ./checkpoints/stage3_cfm `
    --num_gpus $NumGPUs `
    --batch_size $BatchSize `
    --epochs $Epochs `
    --learning_rate $LR `
    --early_stopping $Patience `
    --no_amp
