# Stage 1: Train Multi-Modal VAE from scratch.
# All hyperparameters in configs/stage1_vae.yaml (4-class, kl=1.0, contrastive=0.3, dual-GPU).
# CLI args override YAML. Output: ./checkpoints/stage1_multimodal/vae_best.pt
# Estimated time: ~13h on 2x RTX 3090. Usage: .\run_01_train.ps1

python scripts/train_stage1_multimodal.py `
    --config ./configs/stage1_vae.yaml `
    --output_dir ./checkpoints/stage1_multimodal `
    --num_gpus 2 `
    --no_amp
