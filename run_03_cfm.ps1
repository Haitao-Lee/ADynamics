# Stage 3: MMSE-Conditional Flow Matching (Forward-Only)
# Learns disease progression vector field in latent space
# Key: Only forward flows, distance-aware sampling, rectified flow regularization
# Usage: .\run_03_cfm.ps1

python scripts/train_stage3_cfm.py `
    --encoder_checkpoint ./checkpoints/stage1_multimodal_v4/vae_best.pt `
    --num_classes 3 `
    --batch_size 16 `
    --epochs 300 `
    --learning_rate 0.0001 `
    --no_amp `
    --cfm_base_channels 64 `
    --time_embed_dim 128 `
    --cond_embed_dim 64 `
    --rectified_flow_weight 0.01 `
    --early_stopping 50 `
    --output_dir ./checkpoints/stage3_cfm_v4
