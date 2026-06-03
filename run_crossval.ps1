# Cross-Validation: 5-fold stratified CV for reliable performance estimates
# Reports mean +/- std for all metrics
# Usage: .\run_crossval.ps1

python scripts/run_cross_validation.py `
    --json ./core_data/dataset_manifest_merged_v2.json `
    --output_dir ./inference_results/cross_validation `
    --n_folds 5 `
    --epochs_per_fold 100 `
    --batch_size 2 `
    --learning_rate 0.00005 `
    --cls_weight 3.0 `
    --kl_weight 0.1 `
    --contrastive_weight 0.05 `
    --base_channels 32 `
    --num_gpus 2 `
    --no_amp
