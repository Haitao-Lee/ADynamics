# Ablation Experiments: Systematic component analysis
# Tests contribution of each component (KL, contrastive, cls_weight, etc.)
# Usage: .\run_ablation.ps1
#
# Run specific ablation:
#   python scripts/run_ablation.py --ablation kl_weight
#   python scripts/run_ablation.py --ablation contrastive
#   python scripts/run_ablation.py --ablation cls_weight

python scripts/run_ablation.py `
    --json ./core_data/dataset_manifest_merged_v2.json `
    --output_dir ./inference_results/ablation `
    --ablation all `
    --base_channels 32 `
    --num_gpus 2 `
    --no_amp
