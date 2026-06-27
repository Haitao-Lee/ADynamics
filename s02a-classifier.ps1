# s02a-classifier.ps1
# ====================
# Stage 2a: Frozen encoder, train classifier head (5-modality).
# Input:  ./checkpoints/stage1/vae_best.pt
# Output: ./checkpoints/stage2_classifier/classifier_best.pt
# Usage:  .\s02a-classifier.ps1

param(
    [int]$Epochs     = 100,
    [int]$BatchSize  = 2,
    [float]$LR       = 1e-3,
    [int]$NumGPUs    = 2,
    [int]$Patience   = 30
)

Import-Module "$PSScriptRoot\scripts\ps\common.psm1" -Force
Initialize-ADynamicsEnv

$Stage1Ckpt = Join-Path $PSScriptRoot "checkpoints\stage1\vae_best.pt"
if (-not (Test-Path $Stage1Ckpt)) {
    Write-Host "Missing: $Stage1Ckpt  (run s01-train.ps1 first)" -ForegroundColor Red; exit 1
}

$Py = Get-PythonExe

& $Py -u scripts/train_stage2_classifier.py `
    --config ./configs/stage2a_classifier.yaml `
    --checkpoint $Stage1Ckpt `
    --output_dir ./checkpoints/stage2_classifier `
    --num_gpus $NumGPUs `
    --batch_size $BatchSize `
    --epochs $Epochs `
    --learning_rate $LR `
    --early_stopping $Patience `
    --no_amp
