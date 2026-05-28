# Baseline Comparison: CFM vs Linear/KNN/Regression baselines
# Validates that flow matching provides value over simpler approaches
# Usage: .\run_baseline.ps1

python scripts/run_baseline_comparison.py `
    --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt `
    --output_dir ./inference_results/baseline_comparison
