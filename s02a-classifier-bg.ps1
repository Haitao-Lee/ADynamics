# s02a-classifier-bg.ps1
# =======================
# Stage 2a: Frozen encoder classifier. Detached background.
# Input:  ./checkpoints/stage1/vae_best.pt
# Output: ./checkpoints/stage2_classifier/classifier_best.pt
# Logs:   ./logs/stage2a.out, ./logs/stage2a.err
# Usage:  .\s02a-classifier-bg.ps1

param(
    [int]$Epochs     = 100,
    [int]$BatchSize  = 8,
    [float]$LR       = 1e-4,
    [int]$NumGPUs    = 2,
    [int]$Patience   = 30
)

Import-Module "$PSScriptRoot\scripts\ps\common.psm1" -Force
Initialize-ADynamicsEnv
Stop-ExistingStage -Pattern "train_stage2_classifier" -Label "Stage 2a"

$Stage1Ckpt = Join-Path $PSScriptRoot "checkpoints\stage1\vae_best.pt"
if (-not (Test-Path $Stage1Ckpt)) {
    Write-Host "Missing: $Stage1Ckpt  (run s01-train.ps1 first)" -ForegroundColor Red; exit 1
}

$argList = @(
    "-u", "scripts/train_stage2_classifier.py",
    "--config",          "./configs/stage2a_classifier.yaml",
    "--checkpoint",      $Stage1Ckpt,
    "--output_dir",      "./checkpoints/stage2_classifier",
    "--num_gpus",        "$NumGPUs",
    "--batch_size",      "$BatchSize",
    "--epochs",          "$Epochs",
    "--learning_rate",   "$LR",
    "--early_stopping",  "$Patience",
    "--no_amp"
)

Start-DetachedTraining `
    -ArgList $argList `
    -Stdout  (Join-Path $PSScriptRoot "logs\stage2a.out") `
    -Stderr  (Join-Path $PSScriptRoot "logs\stage2a.err") `
    -Label   "Stage 2a Classifier"
