# Stage 2b: Freeze Encoder, Train Decoder
# Improves reconstruction quality for CFM
# Usage: .\run_stage2b.ps1

python scripts/train_stage2_decoder.py `
    --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --batch_size 2 `
    --epochs 200 `
    --learning_rate 0.0001 `
    --no_amp `
    --recon_loss_type l1 `
    --kl_weight 0.0 `
    --early_stopping 30 `
    --output_dir ./checkpoints/stage2_decoder
