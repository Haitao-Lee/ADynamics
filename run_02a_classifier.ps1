# Stage 2a: Freeze Encoder, Train Classifier Head.
# All hyperparameters in configs/stage2a_classifier.yaml.
# Dual-GPU (canonical 2x RTX 3090). Usage: .\run_02a_classifier.ps1
# IMPORTANT: --t1_only must match the modality config used in Stage 1.

python scripts/train_stage2_classifier.py `
    --config ./configs/stage2a_classifier.yaml `
    --checkpoint ./checkpoints/stage1_t1_demo/vae_best.pt `
    --output_dir ./checkpoints/stage2_t1_demo `
    --num_gpus 2 `
    --no_amp `
    --t1_only
