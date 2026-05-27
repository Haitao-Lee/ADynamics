# Stage 1: Resume Training from Checkpoint
# Usage: .\run_stage1_resume.ps1

python scripts/train_stage1_multimodal.py `
    --batch_size 2 `
    --learning_rate 0.00003 `
    --num_gpus 2 `
    --no_amp `
    --cls_weight 2.0 `
    --kl_weight 0.1 `
    --early_stopping 50 `
    --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --output_dir ./checkpoints/stage1_multimodal
