# Run All Validations After Training
# All defaults loaded from configs/analysis.yaml. CLI overrides YAML.
# Usage: .\run_analysis_all.ps1

Write-Host "=== 1. Latent Space Analysis ===" -ForegroundColor Cyan
python scripts/run_latent_analysis.py `
    --config ./configs/analysis.yaml `
    --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --output_dir ./inference_results/latent_analysis

Write-Host "`n=== 2. Reconstruction Validation ===" -ForegroundColor Cyan
python scripts/run_recon_validation.py `
    --config ./configs/analysis.yaml `
    --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --output_dir ./inference_results/recon_validation

Write-Host "`n=== 3. Classification Validation ===" -ForegroundColor Cyan
python scripts/run_cls_validation.py `
    --config ./configs/analysis.yaml `
    --checkpoint ./checkpoints/stage2_classifier/classifier_best.pt `
    --output_dir ./inference_results/cls_validation

Write-Host "`n=== 4. CFM Flow Visualization ===" -ForegroundColor Cyan
python scripts/run_flow_visualization.py `
    --config ./configs/analysis.yaml `
    --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt `
    --output_dir ./inference_results/flow_visualization

Write-Host "`n=== 5. Deformation Validation ===" -ForegroundColor Cyan
python scripts/run_deform_validation.py `
    --config ./configs/analysis.yaml `
    --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt `
    --deform_checkpoint ./checkpoints/stage4_def/def_best.pt `
    --output_dir ./inference_results/deform_validation

Write-Host "`n=== All validations complete ===" -ForegroundColor Green
Write-Host "Results saved to: ./inference_results/"
