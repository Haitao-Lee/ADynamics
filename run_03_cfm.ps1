# Stage 3: MMSE-Conditional Flow Matching (Forward-Only).
# All hyperparameters in configs/stage3_cfm.yaml.
# Dual-GPU (canonical 2x RTX 3090). Usage: .\run_03_cfm.ps1

python scripts/train_stage3_cfm.py `
    --config ./configs/stage3_cfm.yaml `
    --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --output_dir ./checkpoints/stage3_cfm `
    --num_gpus 2 `
    --no_amp
