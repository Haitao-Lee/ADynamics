# s05-joint-bg.ps1
# ================
# Stage 5: Joint fine-tuning. Detached background.
# Input:  stage1 + stage3 + stage4 checkpoints
# Output: ./checkpoints/stage5_joint/joint_best.pt
# Logs:   ./logs/stage5.out, ./logs/stage5.err
# Usage:  .\s05-joint-bg.ps1

param(
    [int]$Epochs     = 100,
    [int]$BatchSize  = 2,
    [float]$LR       = 1e-5,
    [int]$NumGPUs    = 2,
    [int]$Patience   = 30
)

Import-Module "$PSScriptRoot\scripts\ps\common.psm1" -Force
Initialize-ADynamicsEnv
Stop-ExistingStage -Pattern "train_stage5_joint" -Label "Stage 5"

$Stage1Ckpt = Join-Path $PSScriptRoot "checkpoints\stage1\vae_best.pt"
$Stage3Ckpt = Join-Path $PSScriptRoot "checkpoints\stage3_cfm\cfm_best.pt"
$Stage4Ckpt = Join-Path $PSScriptRoot "checkpoints\stage4_def\def_best.pt"
if (-not (Test-Path $Stage1Ckpt)) { Write-Host "Missing: $Stage1Ckpt" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Stage3Ckpt)) { Write-Host "Missing: $Stage3Ckpt" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Stage4Ckpt)) { Write-Host "Missing: $Stage4Ckpt" -ForegroundColor Red; exit 1 }

$argList = @(
    "-u", "scripts/train_stage5_joint.py",
    "--config",            "./configs/stage5_joint.yaml",
    "--encoder_checkpoint",$Stage1Ckpt,
    "--cfm_checkpoint",    $Stage3Ckpt,
    "--deform_checkpoint", $Stage4Ckpt,
    "--output_dir",        "./checkpoints/stage5_joint",
    "--num_gpus",          "$NumGPUs",
    "--batch_size",        "$BatchSize",
    "--epochs",            "$Epochs",
    "--learning_rate",     "$LR",
    "--early_stopping",    "$Patience",
    "--no_amp"
)

Start-DetachedTraining `
    -ArgList $argList `
    -Stdout  (Join-Path $PSScriptRoot "logs\stage5.out") `
    -Stderr  (Join-Path $PSScriptRoot "logs\stage5.err") `
    -Label   "Stage 5 Joint"
