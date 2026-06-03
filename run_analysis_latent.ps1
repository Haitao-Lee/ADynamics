# Post-Stage 1: Latent Space Analysis
# All defaults loaded from configs/analysis.yaml. CLI overrides YAML.
# Usage: .\run_analysis_latent.ps1

python scripts/run_latent_analysis.py `
    --config ./configs/analysis.yaml `
    --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --output_dir ./inference_results/latent_analysis
