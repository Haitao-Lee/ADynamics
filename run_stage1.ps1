# Stage 1: Multi-Modal VAE Training
# Trains encoder + lightweight decoder + classifier
# Usage: .\run_stage1.ps1

python scripts/train_stage1_multimodal.py `
    --batch_size 2 `
    --learning_rate 0.00005 `
    --num_gpus 2 `
    --no_amp `
    --cls_weight 2.0 `
    --kl_weight 0.1 `
    --latent_channels 32 `
    --base_channels 16 `
    --decoder_depth 4 `
    --dropout_rate 0.2 `
    --epochs 300 `
    --early_stopping 50 `
    --output_dir ./checkpoints/stage1_multimodal
