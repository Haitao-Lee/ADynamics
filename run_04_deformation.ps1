# Stage 4: Deformation Generator
# All hyperparameters are loaded from configs/stage4_deform.yaml.
# CLI args override YAML values. Usage: .\run_04_deformation.ps1

python scripts/train_stage4_deformation.py `
    --config ./configs/stage4_deform.yaml `
    --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt `
    --output_dir ./checkpoints/stage4_def
