# s01-train-bg.ps1
# ================
# Stage 1: Multi-Modal VAE (5-modality). Detached background process.
# Output: ./checkpoints/stage1/vae_best.pt
# Logs:   ./logs/stage1.out, ./logs/stage1.err
# Usage:  .\s01-train-bg.ps1
#         .\s01-train-bg.ps1 -Epochs 1   (smoke test)

param(
    [int]$Epochs    = 300,
    [int]$BatchSize = 2,
    [int]$NumGPUs   = 2
)

Import-Module "$PSScriptRoot\scripts\ps\common.psm1" -Force
Initialize-ADynamicsEnv
Stop-ExistingStage -Pattern "train_stage1_multimodal" -Label "Stage 1"

$argList = @(
    "-u", "scripts/train_stage1_multimodal.py",
    "--config",          "./configs/stage1_vae.yaml",
    "--output_dir",      "./checkpoints/stage1",
    "--num_gpus",        "$NumGPUs",
    "--batch_size",      "$BatchSize",
    "--epochs",          "$Epochs",
    "--no_amp"
)

Start-DetachedTraining `
    -ArgList $argList `
    -Stdout  (Join-Path $PSScriptRoot "logs\stage1.out") `
    -Stderr  (Join-Path $PSScriptRoot "logs\stage1.err") `
    -Label   "Stage 1 VAE"
