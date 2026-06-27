# run-analysis-all.ps1
# ====================
# Run all post-training validations (latent, recon, cls, flow, deform).
# Usage: .\run-analysis-all.ps1

Import-Module "$PSScriptRoot\scripts\ps\common.psm1" -Force
Initialize-ADynamicsEnv
$Py = Get-PythonExe

$S1  = "./checkpoints/stage1/vae_best.pt"
$S2  = "./checkpoints/stage2_classifier/classifier_best.pt"
$S3  = "./checkpoints/stage3_cfm/cfm_best.pt"
$S4  = "./checkpoints/stage4_def/def_best.pt"

Write-Host "=== 1. Latent Space Analysis ===" -ForegroundColor Cyan
& $Py scripts/run_latent_analysis.py --config ./configs/analysis.yaml --checkpoint $S1 --output_dir ./inference_results/latent_analysis

Write-Host "`n=== 2. Reconstruction Validation ===" -ForegroundColor Cyan
& $Py scripts/run_recon_validation.py --config ./configs/analysis.yaml --checkpoint $S1 --output_dir ./inference_results/recon_validation

Write-Host "`n=== 3. Classification Validation ===" -ForegroundColor Cyan
& $Py scripts/run_cls_validation.py --config ./configs/analysis.yaml --checkpoint $S2 --output_dir ./inference_results/cls_validation

Write-Host "`n=== 4. CFM Flow Visualization ===" -ForegroundColor Cyan
& $Py scripts/run_flow_visualization.py --config ./configs/analysis.yaml --encoder_checkpoint $S1 --cfm_checkpoint $S3 --output_dir ./inference_results/flow_visualization

Write-Host "`n=== 5. Deformation Validation ===" -ForegroundColor Cyan
& $Py scripts/run_deform_validation.py --config ./configs/analysis.yaml --encoder_checkpoint $S1 --cfm_checkpoint $S3 --deform_checkpoint $S4 --output_dir ./inference_results/deform_validation

Write-Host "`n=== All validations complete ===" -ForegroundColor Green
Write-Host "Results: ./inference_results/"
