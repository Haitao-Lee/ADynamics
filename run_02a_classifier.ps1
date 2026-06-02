# Stage 2a: Freeze Encoder, Train Classifier Head
# All hyperparameters are loaded from configs/stage2a_classifier.yaml.
# CLI args override YAML values. Usage: .\run_02a_classifier.ps1

python scripts/train_stage2_classifier.py `
    --config ./configs/stage2a_classifier.yaml `
    --checkpoint ./checkpoints/stage1_multimodal_v4/vae_best.pt `
    --output_dir ./checkpoints/stage2_classifier_v4
