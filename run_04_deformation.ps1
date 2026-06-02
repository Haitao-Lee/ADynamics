# Stage 4: Deformation Generator
# Learns to generate 3D displacement fields from latent
# Usage: .\run_04_deformation.ps1

python scripts/train_stage4_deformation.py `
    --encoder_checkpoint ./checkpoints/stage1_multimodal_v4/vae_best.pt `
    --cfm_checkpoint ./checkpoints/stage3_cfm_v4/cfm_best.pt `
    --num_classes 3 `
    --batch_size 2 `
    --epochs 200 `
    --learning_rate 0.0001 `
    --no_amp `
    --sim_weight 1.0 `
    --smooth_weight 0.1 `
    --jacobian_weight 0.01 `
    --early_stopping 50 `
    --output_dir ./checkpoints/stage4_def_v4
