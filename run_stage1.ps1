# Stage 1: Multi-Modal VAE Training (3-class: NC / SCD+MCI / AD)
# Trains encoder + lightweight decoder + classifier
# Usage: .\run_stage1.ps1
#
# Key design choices:
#   - 3-class: NC(0) / SCD+MCI(1) / AD(2) — merges ambiguous middle stages
#   - cls_weight=3.0: Strong classification signal to learn discriminative latent space
#   - kl_weight=0.1: KL regularization for structured latent manifold
#   - contrastive_weight=0.05: Ordinal contrastive loss for disease stage separation
#   - base_channels=32: Larger encoder capacity for better feature extraction

python scripts/train_stage1_multimodal.py `
    --batch_size 2 `
    --learning_rate 0.00005 `
    --num_gpus 2 `
    --no_amp `
    --cls_weight 3.0 `
    --kl_weight 0.1 `
    --contrastive_weight 0.05 `
    --latent_channels 32 `
    --base_channels 32 `
    --decoder_depth 4 `
    --dropout_rate 0.2 `
    --num_classes 3 `
    --epochs 300 `
    --early_stopping 50 `
    --output_dir ./checkpoints/stage1_multimodal
