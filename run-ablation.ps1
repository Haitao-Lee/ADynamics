# run-ablation.ps1
# ================
# Ablation experiments: systematic component analysis.
# Usage: .\run-ablation.ps1

param([string]$Ablation = "all")

Import-Module "$PSScriptRoot\scripts\ps\common.psm1" -Force
Initialize-ADynamicsEnv
$Py = Get-PythonExe

& $Py -u scripts/run_ablation.py `
    --json ./core_data/dataset_manifest_merged_v2.json `
    --output_dir ./inference_results/ablation `
    --ablation $Ablation `
    --base_channels 32 `
    --num_gpus 2 `
    --no_amp
