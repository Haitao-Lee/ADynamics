# s01-train.ps1
# =============
# Stage 1: Multi-Modal VAE (5-modality). Foreground, streams to terminal.
# Output: ./checkpoints/stage1/vae_best.pt
# Usage:  .\s01-train.ps1
#         .\s01-train.ps1 -Epochs 1   (smoke test)

param(
    [int]$Epochs            = 300,
    [int]$BatchSize         = 2,
    [int]$AccumulationSteps = 4,
    [int]$NumGPUs           = 2
)

Import-Module "$PSScriptRoot\scripts\ps\common.psm1" -Force
Initialize-ADynamicsEnv
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$Py = Get-PythonExe

& $Py -u scripts/train_stage1_multimodal.py `
    --config ./configs/stage1_vae.yaml `
    --output_dir ./checkpoints/stage1 `
    --num_gpus $NumGPUs `
    --batch_size $BatchSize `
    --accumulation_steps $AccumulationSteps `
    --epochs $Epochs `
    --no_amp `
    --use_checkpointing `
    --fmri_t_target 100
