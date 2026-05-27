# Post-Stage 1: Latent Space Analysis
# Checks if encoder learned discriminative latent representation
# Usage: .\run_analysis.ps1

python scripts/run_latent_analysis.py `
    --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --output_dir ./inference_results/latent_analysis `
    --num_samples 500
