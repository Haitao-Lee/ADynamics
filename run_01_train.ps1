# Stage 1: Train Multi-Modal VAE from scratch.
# All hyperparameters in configs/stage1_vae.yaml (4-class, kl_weight=1.0, contrastive_weight=0.3).
# CLI args override YAML. Output: ./checkpoints/stage1_multimodal/vae_best.pt
# Estimated time: ~13h on 2 GPUs. Usage: .\run_01_train.ps1

python scripts/train_stage1_multimodal.py `
    --config ./configs/stage1_vae.yaml `
    --output_dir ./checkpoints/stage1_multimodal
