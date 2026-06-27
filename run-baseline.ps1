# run-baseline.ps1
# ================
# Baseline comparison: CFM vs Linear/KNN/Regression baselines.
# Usage: .\run-baseline.ps1

Import-Module "$PSScriptRoot\scripts\ps\common.psm1" -Force
Initialize-ADynamicsEnv
$Py = Get-PythonExe

& $Py scripts/run_baseline_comparison.py `
    --encoder_checkpoint ./checkpoints/stage1/vae_best.pt `
    --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt `
    --output_dir ./inference_results/baseline_comparison
