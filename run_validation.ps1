# Run All Validations After Training
# Usage: .\run_validation.ps1

Write-Host "=== 1. Latent Space Analysis ===" -ForegroundColor Cyan
python scripts/run_latent_analysis.py `
    --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --output_dir ./inference_results/latent_analysis

Write-Host "`n=== 2. Reconstruction Validation ===" -ForegroundColor Cyan
python scripts/run_recon_validation.py `
    --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --output_dir ./inference_results/recon_validation `
    --num_samples 20

Write-Host "`n=== 3. Classification Validation ===" -ForegroundColor Cyan
python scripts/run_cls_validation.py `
    --checkpoint ./checkpoints/stage2_classifier/classifier_best.pt `
    --output_dir ./inference_results/cls_validation

Write-Host "`n=== All validations complete ===" -ForegroundColor Green
Write-Host "Results saved to: ./inference_results/"
