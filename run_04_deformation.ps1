# Stage 4: Deformation Generator.
# All hyperparameters in configs/stage4_deform.yaml.
# Dual-GPU (canonical 2x RTX 3090). Usage: .\run_04_deformation.ps1

python scripts/train_stage4_deformation.py `
    --config ./configs/stage4_deform.yaml `
    --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt `
    --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt `
    --output_dir ./checkpoints/stage4_def `
    --num_gpus 2 `
    --no_amp
