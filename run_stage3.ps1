# Stage 3: Conditional Flow Matching
# Learns disease progression vector field in latent space
# Usage: .\run_stage3.ps1

python scripts/train_stage3_cfm.py `
    --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --batch_size 16 `
    --epochs 300 `
    --learning_rate 0.0001 `
    --no_amp `
    --cfm_base_channels 64 `
    --time_embed_dim 128 `
    --cond_embed_dim 64 `
    --early_stopping 50 `
    --output_dir ./checkpoints/stage3_cfm
