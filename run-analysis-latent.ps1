# run-analysis-latent.ps1
# =======================
# Post-Stage 1: latent space analysis only.
# Usage: .\run-analysis-latent.ps1

Import-Module "$PSScriptRoot\scripts\ps\common.psm1" -Force
Initialize-ADynamicsEnv
$Py = Get-PythonExe

& $Py scripts/run_latent_analysis.py `
    --config ./configs/analysis.yaml `
    --checkpoint ./checkpoints/stage1/vae_best.pt `
    --output_dir ./inference_results/latent_analysis
