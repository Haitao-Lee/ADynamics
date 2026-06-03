# Stage 3: MMSE-Conditional Flow Matching (Forward-Only)
# All hyperparameters are loaded from configs/stage3_cfm.yaml.
# CLI args override YAML values. Usage: .\run_03_cfm.ps1

python scripts/train_stage3_cfm.py `
    --config ./configs/stage3_cfm.yaml `
    --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --output_dir ./checkpoints/stage3_cfm
