# Stage 5: Joint Fine-tuning
# Fine-tunes all modules together end-to-end
# Usage: .\run_05_joint.ps1

python scripts/train_stage5_joint.py `
    --encoder_checkpoint ./checkpoints/stage1_multimodal_v4/vae_best.pt `
    --cfm_checkpoint ./checkpoints/stage3_cfm_v4/cfm_best.pt `
    --deform_checkpoint ./checkpoints/stage4_def_v4/def_best.pt `
    --num_classes 3 `
    --batch_size 2 `
    --epochs 100 `
    --learning_rate 1e-5 `
    --no_amp `
    --recon_weight 1.0 `
    --cfm_weight 0.1 `
    --def_weight 0.1 `
    --early_stopping 30 `
    --output_dir ./checkpoints/stage5_joint_v4
