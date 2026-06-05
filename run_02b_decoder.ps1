# Stage 2b: Freeze Encoder, Train Decoder.
# All hyperparameters in configs/stage2b_decoder.yaml.
# Dual-GPU (canonical 2x RTX 3090). Usage: .\run_02b_decoder.ps1

python scripts/train_stage2_decoder.py `
    --config ./configs/stage2b_decoder.yaml `
    --checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --output_dir ./checkpoints/stage2_decoder `
    --num_gpus 2 `
    --no_amp
