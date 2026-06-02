# Stage 2a: Freeze Encoder, Train Classifier Head
# Validates encoder's latent discriminability
# Usage: .\run_stage2a.ps1

python scripts/train_stage2_classifier.py `
    --checkpoint ./checkpoints/stage1_multimodal_v4/vae_best.pt `
    --num_classes 3 `
    --batch_size 4 `
    --epochs 100 `
    --learning_rate 0.0001 `
    --no_amp `
    --early_stopping 30 `
    --output_dir ./checkpoints/stage2_classifier_v4
