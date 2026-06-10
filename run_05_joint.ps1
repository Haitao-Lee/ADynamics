# Stage 5: Joint Fine-tuning.
# All hyperparameters in configs/stage5_joint.yaml.
# Dual-GPU (canonical 2x RTX 3090). Usage: .\run_05_joint.ps1
# IMPORTANT: --t1_only must match Stage 1's modality config.

python scripts/train_stage5_joint.py `
    --config ./configs/stage5_joint.yaml `
    --encoder_checkpoint ./checkpoints/stage1_t1_demo/vae_best.pt `
    --cfm_checkpoint ./checkpoints/stage3_cfm_t1_demo/cfm_best.pt `
    --deform_checkpoint ./checkpoints/stage4_def_t1_demo/def_best.pt `
    --output_dir ./checkpoints/stage5_joint_t1_demo `
    --num_gpus 2 `
    --no_amp `
    --t1_only
