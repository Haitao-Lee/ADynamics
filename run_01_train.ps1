# Stage 1: Train from Scratch (3-class: NC / SCD+MCI / AD)
# All hyperparameters are loaded from configs/stage1_vae.yaml.
# CLI args override YAML values. Usage: .\run_01_train.ps1

python scripts/train_stage1_multimodal.py `
    --config ./configs/stage1_vae.yaml `
    --output_dir ./checkpoints/stage1_multimodal_v4
