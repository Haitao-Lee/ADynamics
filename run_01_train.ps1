# Stage 1: Train from Scratch (3-class: NC / SCD+MCI / AD)
# Free Bits + KL annealing to prevent posterior collapse
# Usage: .\run_stage1_resume.ps1

python scripts/train_stage1_multimodal.py `
    --batch_size 2 `
    --learning_rate 0.00003 `
    --num_gpus 2 `
    --no_amp `
    --cls_weight 1.0 `
    --kl_weight 0.5 `
    --kl_warmup_epochs 20 `
    --free_bits 0.01 `
    --num_classes 3 `
    --epochs 300 `
    --early_stopping 50 `
    --output_dir ./checkpoints/stage1_multimodal_v4
