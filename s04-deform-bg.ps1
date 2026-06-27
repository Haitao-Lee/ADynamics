# s04-deform-bg.ps1
# =================
# Stage 4: Deformation Generator. Detached background.
# Input:  ./checkpoints/stage1/vae_best.pt + ./checkpoints/stage3_cfm/cfm_best.pt
# Output: ./checkpoints/stage4_def/def_best.pt
# Logs:   ./logs/stage4.out, ./logs/stage4.err
# Usage:  .\s04-deform-bg.ps1

param(
    [int]$Epochs     = 200,
    [int]$BatchSize  = 2,
    [float]$LR       = 1e-4,
    [int]$NumGPUs    = 2,
    [int]$Patience   = 50
)

Import-Module "$PSScriptRoot\scripts\ps\common.psm1" -Force
Initialize-ADynamicsEnv
Stop-ExistingStage -Pattern "train_stage4_deformation" -Label "Stage 4"

$Stage1Ckpt = Join-Path $PSScriptRoot "checkpoints\stage1\vae_best.pt"
$Stage3Ckpt = Join-Path $PSScriptRoot "checkpoints\stage3_cfm\cfm_best.pt"
if (-not (Test-Path $Stage1Ckpt)) { Write-Host "Missing: $Stage1Ckpt" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Stage3Ckpt)) { Write-Host "Missing: $Stage3Ckpt" -ForegroundColor Red; exit 1 }

$argList = @(
    "-u", "scripts/train_stage4_deformation.py",
    "--config",            "./configs/stage4_deform.yaml",
    "--encoder_checkpoint",$Stage1Ckpt,
    "--cfm_checkpoint",    $Stage3Ckpt,
    "--output_dir",        "./checkpoints/stage4_def",
    "--num_gpus",          "$NumGPUs",
    "--batch_size",        "$BatchSize",
    "--epochs",            "$Epochs",
    "--learning_rate",     "$LR",
    "--early_stopping",    "$Patience",
    "--no_amp"
)

Start-DetachedTraining `
    -ArgList $argList `
    -Stdout  (Join-Path $PSScriptRoot "logs\stage4.out") `
    -Stderr  (Join-Path $PSScriptRoot "logs\stage4.err") `
    -Label   "Stage 4 Deformation"
