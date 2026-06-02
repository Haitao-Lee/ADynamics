# Stage 2b: Freeze Encoder, Train Decoder
# All hyperparameters are loaded from configs/stage2b_decoder.yaml.
# CLI args override YAML values. Usage: .\run_02b_decoder.ps1

python scripts/train_stage2_decoder.py `
    --config ./configs/stage2b_decoder.yaml `
    --checkpoint ./checkpoints/stage1_multimodal_v4/vae_best.pt `
    --output_dir ./checkpoints/stage2_decoder_v4
