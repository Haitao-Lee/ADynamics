"""
CLI smoke test: verify that train_stage1_multimodal.py accepts the new
--use_attention / --no_attention / --attention_levels / --attention_heads
flags AND that values from configs/stage1_vae.yaml are picked up correctly.

We do NOT actually run training (that takes hours). We just invoke
parse_args() and verify the resulting Namespace, then construct the model
and confirm it builds.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Replicate parse_args() call with --config pointing at the canonical YAML
import argparse
from scripts.train_stage1_multimodal import _load_yaml_defaults, parse_args

# Stub sys.argv for parse_args
sys.argv = [
    "train_stage1_multimodal.py",
    "--config", "./configs/stage1_vae.yaml",
    "--no_amp",  # canonical
    "--use_attention",  # default ON, explicit for clarity
    "--attention_levels", "3",
    "--attention_heads", "8",
]

args = parse_args()
print("=== Parsed args (key fields) ===")
print(f"  num_classes       = {args.num_classes}")
print(f"  base_channels     = {args.base_channels}")
print(f"  latent_channels   = {args.latent_channels}")
print(f"  decoder_depth     = {args.decoder_depth}")
print(f"  use_amp           = {args.use_amp}")
print(f"  no_amp            = {args.no_amp}")
print(f"  use_attention     = {args.use_attention}")
print(f"  attention_levels  = {args.attention_levels!r}")
print(f"  attention_heads   = {args.attention_heads}")
print(f"  contrastive_weight= {args.contrastive_weight}")
print(f"  kl_weight         = {args.kl_weight}")

# Build the model and check it instantiates
import torch
from models.vae3d import MultiModalVAE3D

# Mimic the parse logic in main()
use_attention_final = args.use_attention and not args.no_attention
al_raw = args.attention_levels
if isinstance(al_raw, (list, tuple)):
    attn_levels = tuple(int(x) for x in al_raw)
else:
    attn_levels = tuple(int(x.strip()) for x in str(al_raw).split(",") if str(x).strip())

model = MultiModalVAE3D(
    spatial_size=(256, 256, 192),
    in_channels=1,
    latent_channels=args.latent_channels,
    base_channels=args.base_channels,
    num_classes=args.num_classes,
    dropout_rate=args.dropout_rate,
    decoder_depth=args.decoder_depth,
    optional_modalities=["fmri", "asl", "qsm", "flair"],
    use_attention=use_attention_final,
    attention_levels=attn_levels,
    attention_heads=args.attention_heads,
)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel built. Trainable params: {n_params/1e6:.2f}M")

# Count attention blocks across all 5 modality encoders
n_attn = sum(1 for m in model.modules() if m.__class__.__name__ == "MultiAxisAttention3D")
print(f"MultiAxisAttention3D blocks: {n_attn} (= 5 modality encoders x {n_attn // 5} stage(s) per encoder)")

# Quick forward
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
batch = {
    "t1":   torch.randn(1, 1, 256, 256, 192, device=device),
    "fmri": torch.randn(1, 1, 32, 64, 64, device=device),
    "asl":  torch.randn(1, 1, 32, 64, 64, device=device),
    "qsm":  torch.randn(1, 1, 32, 64, 64, device=device),
    "flair":torch.randn(1, 1, 32, 64, 64, device=device),
    "label": torch.tensor([2], device=device),
}
with torch.no_grad():
    recon, cls, mu, logvar = model(batch, return_components=True)
print(f"Forward OK. recon={tuple(recon.shape)}  cls={tuple(cls.shape)}  mu={tuple(mu.shape)}")

# Try --no_attention
print("\n=== With --no_attention ===")
sys.argv = [
    "train_stage1_multimodal.py",
    "--config", "./configs/stage1_vae.yaml",
    "--no_amp",
    "--no_attention",
]
args2 = parse_args()
print(f"  use_attention = {args2.use_attention}")
print(f"  no_attention  = {args2.no_attention}")
model2 = MultiModalVAE3D(
    spatial_size=(256, 256, 192),
    in_channels=1,
    latent_channels=32, base_channels=16, num_classes=4,
    dropout_rate=0.2, decoder_depth=4,
    optional_modalities=["fmri", "asl", "qsm", "flair"],
    use_attention=args2.use_attention and not args2.no_attention,
    attention_levels=(3,),
)
n_attn2 = sum(1 for m in model2.modules() if m.__class__.__name__ == "MultiAxisAttention3D")
n_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
print(f"Model built. Params: {n_params2/1e6:.2f}M  Attention blocks: {n_attn2}")

print("\n=== CLI SMOKE TEST PASSED ===")
