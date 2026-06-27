"""Debug: run ONE forward+backward pass outside the trainer's try/except."""
import sys
import os
import traceback
import warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("nnssl_raw", "E:/LHT_workspace/AD/ADynamics/nnssl_raw")
os.environ.setdefault("nnssl_preprocessed", "E:/LHT_workspace/AD/ADynamics/nnssl_preprocessed")
os.environ.setdefault("nnssl_results", "E:/LHT_workspace/AD/ADynamics/nnssl_results")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
print("torch imported, version:", torch.__version__)

from nnssl_adapters.training.adynamics_trainer import ADynamicsTrainer
print("trainer imported")

# Build the trainer the same way main() does
trainer = ADynamicsTrainer(
    configuration="ADVAE_3d_fullres",
    fold=0,
    dataset_dir="E:/LHT_workspace/AD/ADynamics/nnssl_raw/Dataset501_ADynamics",
    output_dir="E:/LHT_workspace/AD/ADynamics/nnssl_results/ADVAE_3d_fullres",
    config={
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 3e-5,
        "num_gpus": 2,
        "latent_channels": 32,
        "mixup_alpha": 0.0,
        "mixup_prob": 0.0,
    },
)
print("Trainer built")

# Build network + optimizer explicitly (mirrors run_training)
print("Building network...")
trainer.network = trainer.build_network()
print(f"network: {type(trainer.network).__name__}")
print("Setting up DataParallel...")
from utils.multi_gpu import setup_data_parallel
trainer.network = setup_data_parallel(trainer.network, trainer.config.get("num_gpus", 2))
print(f"network after DP: {type(trainer.network).__name__}")
print("Configuring optimizers...")
trainer.optimizer, trainer.scheduler = trainer.configure_optimizers()
trainer.network.to(trainer.device)
print(f"network on {trainer.device}")

# Get dataloaders
print("\nGetting dataloaders...")
train_loader, val_loader = trainer.get_dataloaders()
print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}")

# Get one batch
print("\nFetching first batch...")
batch = next(iter(train_loader))
print(f"batch keys: {list(batch.keys())}")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: tensor {v.shape} dtype={v.dtype}")
    elif v is None:
        print(f"  {k}: None")
    else:
        print(f"  {k}: {type(v).__name__}")

# Run one train step (NO try/except)
print("\n=== Running first train_step WITHOUT try/except ===")
try:
    metrics = trainer.train_step(batch)
    print(f"\n*** train_step SUCCEEDED ***")
    print(f"loss: {metrics.get('loss', 'n/a')}")
    for k, v in metrics.items():
        if k != "per_class_acc":
            print(f"  {k}: {v}")
except Exception as e:
    print(f"\n*** TRAIN STEP FAILED ***")
    print(f"{type(e).__name__}: {e}")
    print("\n--- Full traceback ---")
    traceback.print_exc()
    sys.exit(1)

print("\nDone.")
