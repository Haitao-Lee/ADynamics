# Stage 5: Joint Fine-tuning
# All hyperparameters are loaded from configs/stage5_joint.yaml.
# CLI args override YAML values. Usage: .\run_05_joint.ps1

python scripts/train_stage5_joint.py `
    --config ./configs/stage5_joint.yaml `
    --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt `
    --deform_checkpoint ./checkpoints/stage4_def/def_best.pt `
    --output_dir ./checkpoints/stage5_joint
