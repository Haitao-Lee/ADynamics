# Stage 4: Deformation Generator.
# All hyperparameters in configs/stage4_deform.yaml.
# Dual-GPU (canonical 2x RTX 3090). Usage: .\run_04_deformation.ps1
# IMPORTANT: --t1_only must match Stage 1's modality config.

python scripts/train_stage4_deformation.py `
    --config ./configs/stage4_deform.yaml `
    --encoder_checkpoint ./checkpoints/stage1_t1_demo/vae_best.pt `
    --cfm_checkpoint ./checkpoints/stage3_cfm_t1_demo/cfm_best.pt `
    --output_dir ./checkpoints/stage4_def_t1_demo `
    --num_gpus 2 `
    --no_amp `
    --t1_only
