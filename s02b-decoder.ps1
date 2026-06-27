# s02b-decoder.ps1
# ================
# Stage 2b: Frozen encoder, fine-tune decoder.
# Input:  ./checkpoints/stage1/vae_best.pt
# Output: ./checkpoints/stage2_decoder/decoder_best.pt
# Usage:  .\s02b-decoder.ps1

param(
    [int]$Epochs     = 200,
    [int]$BatchSize  = 2,
    [float]$LR       = 5e-4,
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

& $Py -u scripts/train_stage2_decoder.py `
    --config ./configs/stage2b_decoder.yaml `
    --checkpoint $Stage1Ckpt `
    --output_dir ./checkpoints/stage2_decoder `
    --num_gpus $NumGPUs `
    --batch_size $BatchSize `
    --epochs $Epochs `
    --learning_rate $LR `
    --early_stopping $Patience `
    --no_amp
