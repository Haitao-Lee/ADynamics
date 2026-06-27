"""Test: run 10 train_steps in a loop to see if any subsequent batch fails."""
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
from nnssl_adapters.training.adynamics_trainer import ADynamicsTrainer
from utils.multi_gpu import setup_data_parallel

trainer = ADynamicsTrainer(
    configuration="ADVAE_3d_fullres",
    fold=0,
    dataset_dir="E:/LHT_workspace/AD/ADynamics/nnssl_raw/Dataset501_ADynamics",
    output_dir="E:/LHT_workspace/AD/ADynamics/nnssl_results/ADVAE_3d_fullres",
    config={
        "epochs": 1, "batch_size": 2, "learning_rate": 3e-5,
        "num_gpus": 2, "latent_channels": 32, "mixup_alpha": 0.0, "mixup_prob": 0.0,
    },
)
trainer.network = trainer.build_network()
trainer.network = setup_data_parallel(trainer.network, 2)
trainer.optimizer, trainer.scheduler = trainer.configure_optimizers()
trainer.network.to(trainer.device)

train_loader, val_loader = trainer.get_dataloaders()
print(f"train batches: {len(train_loader)}")

# Run 10 batches
for i, batch in enumerate(train_loader):
    if i >= 10:
        break
    print(f"\n=== batch {i} ===")
    try:
        metrics = trainer.train_step(batch)
        print(f"  loss: {metrics['loss']:.4f}  recon: {metrics['recon_loss']:.4f}  "
              f"cls: {metrics['cls_loss']:.4f}  kl: {metrics['kl_loss']:.4f}  "
              f"ord: {metrics['ord_reg_loss']:.4f}  acc: {metrics['cls_acc']:.2f}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        break
    finally:
        if "t1" in batch:
            del batch
        torch.cuda.empty_cache()

print("\n\nDone with 10 batches.")
