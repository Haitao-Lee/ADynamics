"""Test: model + DataParallel forward pass to find the silent failure."""
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
from nnssl_adapters.models.ad_vaewrapper import ADynamicsVAEWrapper
from utils.multi_gpu import setup_data_parallel

# Build the inner model directly (no DataParallel, no trainer overhead)
print("Building wrapper...")
model = ADynamicsVAEWrapper(latent_channels=32, base_channels=16)
print(f"Model on CPU: {type(model).__name__}")

# Move to GPU 0 first
print("Moving to cuda:0...")
model = model.cuda(0)
print(f"Model on cuda:0: {next(model.parameters()).device}")

# Wrap in DataParallel
print("Wrapping in DataParallel...")
parallel = setup_data_parallel(model, 2)
print(f"Parallel type: {type(parallel).__name__}")
print(f"device_ids: {parallel.device_ids}")

# Build a fake batch
print("\nBuilding fake batch...")
batch = {
    "t1": torch.randn(2, 1, 256, 256, 192).cuda(0),
    "fmri": torch.randn(2, 1, 64, 64, 34, 200).cuda(0),
    "asl": torch.randn(2, 1, 64, 64, 32).cuda(0),
    "qsm": torch.randn(2, 1, 128, 128, 96).cuda(0),
    "flair": torch.randn(2, 1, 128, 128, 32).cuda(0),
    "label": torch.tensor([0, 1]).cuda(0).long(),
    "age": torch.tensor([70.0, 75.0]).cuda(0).float(),
    "sex": torch.tensor([0, 1]).cuda(0).long(),
}
print(f"t1 shape: {batch['t1'].shape}, on {batch['t1'].device}")

print("\n=== Forward pass ===")
try:
    out = parallel(batch, return_components=True, age=batch["age"], sex=batch["sex"])
    print(f"\n*** forward SUCCEEDED ***")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} on {v.device}")
        else:
            print(f"  {k}: {v}")
except Exception as e:
    print(f"\n*** FORWARD FAILED ***")
    print(f"{type(e).__name__}: {e}")
    print("\n--- Full traceback ---")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Backward pass ===")
try:
    loss = out["recon"].mean() + out["cls_logits"].mean() + out["mu"].mean() + out["logvar"].mean()
    loss.backward()
    print(f"*** backward SUCCEEDED ***")
except Exception as e:
    print(f"*** BACKWARD FAILED ***")
    print(f"{type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\nDone.")
