# s02b-decoder-bg.ps1
# ====================
# Stage 2b: Decoder fine-tuning. Detached background.
# Input:  ./checkpoints/stage1/vae_best.pt
# Output: ./checkpoints/stage2_decoder/decoder_best.pt
# Logs:   ./logs/stage2b.out, ./logs/stage2b.err
# Usage:  .\s02b-decoder-bg.ps1

param(
    [int]$Epochs     = 100,
    [int]$BatchSize  = 2,
    [float]$LR       = 1e-4,
    [int]$NumGPUs    = 2,
    [int]$Patience   = 30
)

Import-Module "$PSScriptRoot\scripts\ps\common.psm1" -Force
Initialize-ADynamicsEnv
Stop-ExistingStage -Pattern "train_stage2_decoder" -Label "Stage 2b"

$Stage1Ckpt = Join-Path $PSScriptRoot "checkpoints\stage1\vae_best.pt"
if (-not (Test-Path $Stage1Ckpt)) {
    Write-Host "Missing: $Stage1Ckpt  (run s01-train.ps1 first)" -ForegroundColor Red; exit 1
}

$argList = @(
    "-u", "scripts/train_stage2_decoder.py",
    "--config",          "./configs/stage2b_decoder.yaml",
    "--checkpoint",      $Stage1Ckpt,
    "--output_dir",      "./checkpoints/stage2_decoder",
    "--num_gpus",        "$NumGPUs",
    "--batch_size",      "$BatchSize",
    "--epochs",          "$Epochs",
    "--learning_rate",   "$LR",
    "--early_stopping",  "$Patience",
    "--no_amp"
)

Start-DetachedTraining `
    -ArgList $argList `
    -Stdout  (Join-Path $PSScriptRoot "logs\stage2b.out") `
    -Stderr  (Join-Path $PSScriptRoot "logs\stage2b.err") `
    -Label   "Stage 2b Decoder"
