# Stage 1: Train Multi-Modal VAE from scratch.
#
# Default config: T1 + fMRI + ASL + QSM + FLAIR + age/sex, cls_weight=4.0,
# cyclical KL, latent mixup, attention at mid-level + bottleneck.
#
# Edit configs/stage1_vae.yaml to change any setting — there is no v-prefixed
# variant. Output: ./checkpoints/stage1_multimodal/vae_best.pt
#
# Estimated time: ~6-10h on 2x RTX 3090.
# Usage: .\run_01_train.ps1

python scripts/train_stage1_multimodal.py `
    --config ./configs/stage1_vae.yaml `
    --num_gpus 2 `
    --no_amp
