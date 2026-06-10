"""Compare num_workers=0 vs 2 epoch time using subprocess (Windows-safe)."""
import subprocess
import sys
import time

code = '''
import sys, time
sys.path.insert(0, "e:/LHT_workspace/AD/ADynamics")
import torch
from torch.utils.data import DataLoader
from core_data.dataset import multimodal_collate_fn
from models.vae3d import MultiModalVAE3D
from engine.trainer_vae import MultiModalVAETrainer
from utils.multi_gpu import setup_data_parallel

data_list = [{"t1": f"f{i}.nii.gz", "patient_id": f"p{i}", "label": i%4, "age": 65.0, "sex": 1} for i in range(16)]
class FakeT1:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i):
        time.sleep(0.05)  # simulate NIfTI load
        item = data_list[i % len(data_list)].copy()
        item["t1"] = torch.randn(1, 256, 256, 192) * 0.3 + 0.4
        item["age"] = torch.tensor(item["age"])
        item["sex"] = torch.tensor(item["sex"])
        item["label"] = torch.tensor(item["label"])
        return item

ds = FakeT1(16)
NW = int(sys.argv[1])
loader = DataLoader(ds, batch_size=2, collate_fn=multimodal_collate_fn, shuffle=False,
                    num_workers=NW, persistent_workers=(NW > 0), prefetch_factor=2 if NW > 0 else 2)

m = MultiModalVAE3D(
    spatial_size=(256,256,192), latent_channels=64, base_channels=16,
    num_classes=4, decoder_depth=4, optional_modalities=[],
    use_demographic_cond=True, use_attention=True, attention_levels=(1,2,3),
    use_checkpointing=True,
)
m = setup_data_parallel(m, 2)
opt = torch.optim.AdamW(m.parameters(), lr=3e-5)
config = {
    "cls_weight": 4.0, "kl_weight": 0.3, "kl_strategy": "cyclical",
    "kl_cycle_len": 15, "kl_cycle_low_frac": 0.1,
    "mixup_alpha": 0.4, "mixup_prob": 0.5,
    "free_bits": 0.05, "use_amp": False, "use_demographic_cond": True,
    "ordinal_reg_weight": 0.1, "num_classes": 4,
    "class_names": ["NC","SCD","MCI","AD"], "latent_channels": 64,
}

trainer = MultiModalVAETrainer(m, opt, loader, loader, torch.device("cuda"), config, scheduler=None)
trainer.current_kl_weight = 0.05

# Warmup
_ = trainer.train_epoch()

t0 = time.time()
_ = trainer.train_epoch()
dt = time.time() - t0
print(f"NW={NW}: {dt:.1f}s for 8 batches; projected {dt * 623/8 / 60:.1f} min/epoch")
'''

for nw in [0, 2]:
    print(f"--- Testing num_workers={nw} ---")
    subprocess.run([sys.executable, "-c", code, str(nw)], check=False)
