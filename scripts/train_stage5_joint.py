"""
Stage 5: Joint Fine-tuning for Multi-Modal VAE.

Fine-tunes all pretrained modules together end-to-end:
    - Multi-Modal VAE Encoder + Decoder (from Stage 1/2)
    - CFM Velocity Field (from Stage 3)
    - Deformation Generator (from Stage 4)

Combined loss:
    L_total = lambda_recon * L_recon + lambda_cfm * L_cfm + lambda_def * L_def

Where:
    L_recon: L1 reconstruction loss
    L_cfm: CFM velocity field loss
    L_def: Deformation smoothness + similarity loss

All modules are unfrozen but use lower learning rates for pretrained weights.

Usage:
    python scripts/train_stage5_joint.py \
        --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \
        --decoder_checkpoint ./checkpoints/stage2_decoder/decoder_best.pt \
        --cfm_checkpoint ./checkpoints/stage3_cfm/cfm_best.pt \
        --deform_checkpoint ./checkpoints/stage4_def/def_best.pt \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --output_dir ./checkpoints/stage5_joint
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_train_transforms, get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D
from models.vector_field import VelocityFieldNet
from models.spatial_transform import (
    DeformationGenerator,
    SpatialTransformer,
    compute_jacobian_penalty,
)
from engine.losses import GradientSmoothingLoss


def parse_args():
    from utils.config_loader import apply_yaml_defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    pre_args, _ = pre.parse_known_args()

    mapping = [
        (("input", "encoder_checkpoint"), "encoder_checkpoint"),
        (("input", "cfm_checkpoint"), "cfm_checkpoint"),
        (("input", "deform_checkpoint"), "deform_checkpoint"),
        (("input", "num_classes"), "num_classes"),
        (("model", "latent_channels"), "latent_channels"),
        (("model", "base_channels"), "base_channels"),
        (("model", "decoder_depth"), "decoder_depth"),
        (("model", "in_channels"), "in_channels"),
        (("model", "spatial_size"), "spatial_size"),
        (("model", "dropout_rate"), "dropout_rate"),
        (("training", "batch_size"), "batch_size"),
        (("training", "learning_rate"), "learning_rate"),
        (("training", "weight_decay"), "weight_decay"),
        (("training", "epochs"), "epochs"),
        (("training", "early_stopping_patience"), "early_stopping"),
        (("training", "num_gpus"), "num_gpus"),
        (("training", "use_amp"), "use_amp"),
        (("loss", "recon_weight"), "recon_weight"),
        (("loss", "cfm_weight"), "cfm_weight"),
        (("loss", "def_weight"), "def_weight"),
        (("loss", "smooth_weight"), "smooth_weight"),
        (("loss", "jacobian_weight"), "jacobian_weight"),
        (("lr_multipliers", "encoder"), "encoder_lr_mult"),
        (("lr_multipliers", "cfm"), "cfm_lr_mult"),
        (("lr_multipliers", "deform"), "deform_lr_mult"),
        (("output", "dir"), "output_dir"),
        (("seed",), "seed"),
    ]
    config_defaults = apply_yaml_defaults(pre_args.config, mapping) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Stage 5 Joint Fine-tuning", parents=[pre])
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage5_joint")
    parser.add_argument("--encoder_checkpoint", type=str,
                        default="./checkpoints/stage1_multimodal/vae_best.pt")
    parser.add_argument("--decoder_checkpoint", type=str, default=None)
    parser.add_argument("--cfm_checkpoint", type=str,
                        default="./checkpoints/stage3_cfm/cfm_best.pt")
    parser.add_argument("--deform_checkpoint", type=str,
                        default="./checkpoints/stage4_def/def_best.pt")

    # Model params (must match Stage 1)
    parser.add_argument("--latent_channels", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD)")
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--cfm_weight", type=float, default=0.1)
    parser.add_argument("--def_weight", type=float, default=0.1)
    parser.add_argument("--smooth_weight", type=float, default=0.05)
    parser.add_argument("--jacobian_weight", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--early_stopping", type=int, default=30)
    parser.add_argument("--no_amp", action="store_true", default=False)
    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="Enable AMP (default OFF; YAML use_amp: false maps here)")
    # Per-module learning rate multipliers (slower for pretrained modules)
    parser.add_argument("--encoder_lr_mult", type=float, default=0.1)
    parser.add_argument("--cfm_lr_mult", type=float, default=1.0)
    parser.add_argument("--deform_lr_mult", type=float, default=1.0)
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of GPUs for DataParallel (default 2; canonical setup is 2x RTX 3090)")
    # Apply YAML config defaults AFTER all add_argument calls
    # (set_defaults must come last so it isn't overridden by argparse defaults)
    parser.set_defaults(**config_defaults)

    return parser.parse_args()


# ===========================================================================
# Helpers
# ===========================================================================

def _load_data_list(json_path: str) -> list:
    """Load manifest JSON and validate the entries."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return [d for d in data if d.get("t1") and os.path.exists(d["t1"])]


def _build_dataloaders(args, train_tf, val_tf):
    """Build train/val DataLoaders using the shared MultiModalDataset."""
    from core_data.dataset import multimodal_collate_fn
    data_list = _load_data_list(args.json)
    train_size = int(0.8 * len(data_list))
    train_data = data_list[:train_size]
    val_data = data_list[train_size:]
    print(f"Total samples: {len(data_list)} (train={len(train_data)}, val={len(val_data)})")

    train_ds = MultiModalDataset(
        data_list=train_data,
        transform=train_tf,
        spatial_sizes=MULTI_MODAL_SPATIAL_SIZES,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
        preserve_temporal_dim=True,  # Plan C: keep fMRI time axis
    )
    val_ds = MultiModalDataset(
        data_list=val_data,
        transform=val_tf,
        spatial_sizes=MULTI_MODAL_SPATIAL_SIZES,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
        preserve_temporal_dim=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=multimodal_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=multimodal_collate_fn,
    )
    return train_loader, val_loader


def _build_model(args, device):
    """Build the Multi-Modal VAE with fMRI temporal encoder (Plan C)."""
    model = MultiModalVAE3D(
        spatial_size=tuple(args.spatial_size),
        in_channels=args.in_channels,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        decoder_depth=args.decoder_depth,
        optional_modalities=["fmri", "asl", "qsm", "flair"],
        use_fmri_temporal=True,  # Plan C
    ).to(device)
    return model


def _load_encoder_weights(model, ckpt_path: str, device) -> None:
    """Load Stage 1 (or Stage 2b decoder) weights into the VAE model.

    Uses shape-filtered load to tolerate the optional_encoders.fmri
    fMRITemporalEncoder (which may be absent in older checkpoints).
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[Stage 5] Skipping encoder load: {ckpt_path} not found")
        return
    print(f"[Stage 5] Loading encoder weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    from utils.stage23_compat import shape_filtered_load_state_dict
    shape_filtered_load_state_dict(model, sd, strict=False, verbose=True)


# ===========================================================================
# Training loop
# ===========================================================================

def train_one_epoch(
    model,
    cfm,
    deform_gen,
    spatial_transformer,
    train_loader,
    optimizer,
    device,
    args,
    smooth_loss_fn,
):
    """Joint training epoch: recon + cfm + deform."""
    model.train()
    cfm.train()
    deform_gen.train()
    spatial_transformer.train()

    total_loss = total_recon = total_cfm = total_def = 0.0
    n_batches = 0
    from utils.stage23_compat import normalize_fmri_batch
    from tqdm import tqdm
    pbar = tqdm(train_loader, desc="Train (joint)", leave=False)
    for batch in pbar:
        t1 = batch["t1"].to(device)
        x_dict = {"t1": t1}
        for mod in ["fmri", "asl", "qsm", "flair"]:
            if mod in batch and batch[mod] is not None:
                mod_tensor = batch[mod].to(device)
                if mod == "fmri":
                    mod_tensor = normalize_fmri_batch(mod_tensor)
                x_dict[mod] = mod_tensor

        labels = batch.get("label")
        if labels is not None:
            labels = labels.to(device)
            if labels.dim() > 1:
                labels = labels.squeeze()

        optimizer.zero_grad()

        # Forward VAE
        recon, cls_logits, mu, logvar = model(x_dict, return_components=True)
        recon_loss = F.l1_loss(recon, t1)
        cls_loss = F.cross_entropy(cls_logits, labels) if labels is not None else 0.0

        # CFM velocity on (NC, AD) latents — 2-class flow
        # Pick 2 random latents; if labels include both NC(0) and AD(3)
        # we can sample a real flow. Otherwise just learn identity-ish
        # transitions (mu -> mu + small noise).
        B = mu.shape[0]
        # Pairwise transition: latents[i] -> latents[(i+1)%B] (a proxy
        # for adjacent-disease-stage transition; in 4-class this is
        # the closest pair by label index).
        z0 = mu
        z1 = torch.roll(mu, shifts=-1, dims=0)
        t = torch.rand(B, device=device)
        z_t = (1 - t.view(-1, 1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1, 1) * z1
        v_pred = cfm(z_t, t)
        target_v = z1 - z0
        cfm_loss = F.mse_loss(v_pred, target_v)

        # Deformation: latent -> flow field -> spatial transform of T1
        # Take the encoder mu as input to the deformation generator.
        flow = deform_gen(mu)
        deformed = spatial_transformer(t1, flow)
        # Similarity loss: deformed should be close to T1 at inference
        sim_loss = F.l1_loss(deformed, t1)
        # Smoothness on flow
        sm_loss = smooth_loss_fn(flow)
        # Jacobian-determinant penalty (encourage diffeomorphism)
        jac_loss = compute_jacobian_penalty(flow)
        def_loss = sim_loss + args.smooth_weight * sm_loss + args.jacobian_weight * jac_loss

        loss = (
            args.recon_weight * recon_loss
            + cls_loss
            + args.cfm_weight * cfm_loss
            + args.def_weight * def_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(cfm.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(deform_gen.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_cfm += cfm_loss.item()
        total_def += def_loss.item()
        n_batches += 1
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "recon": f"{recon_loss.item():.3f}",
            "cfm": f"{cfm_loss.item():.3f}",
            "def": f"{def_loss.item():.3f}",
        })

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "cfm": total_cfm / n,
        "def": total_def / n,
    }


@torch.no_grad()
def validate(model, cfm, deform_gen, spatial_transformer, val_loader, device, args, smooth_loss_fn):
    """Validation pass — just sum the same losses on val set."""
    model.eval()
    cfm.eval()
    deform_gen.eval()
    spatial_transformer.eval()
    total_loss = total_recon = total_cfm = total_def = 0.0
    n_batches = 0
    from utils.stage23_compat import normalize_fmri_batch
    from tqdm import tqdm
    pbar = tqdm(val_loader, desc="Val (joint)", leave=False)
    for batch in pbar:
        t1 = batch["t1"].to(device)
        x_dict = {"t1": t1}
        for mod in ["fmri", "asl", "qsm", "flair"]:
            if mod in batch and batch[mod] is not None:
                mod_tensor = batch[mod].to(device)
                if mod == "fmri":
                    mod_tensor = normalize_fmri_batch(mod_tensor)
                x_dict[mod] = mod_tensor
        labels = batch.get("label")
        if labels is not None:
            labels = labels.to(device)
            if labels.dim() > 1:
                labels = labels.squeeze()
        recon, cls_logits, mu, logvar = model(x_dict, return_components=True)
        recon_loss = F.l1_loss(recon, t1)
        cls_loss = F.cross_entropy(cls_logits, labels) if labels is not None else torch.tensor(0.0, device=device)
        B = mu.shape[0]
        z0 = mu
        z1 = torch.roll(mu, shifts=-1, dims=0)
        t = torch.rand(B, device=device)
        z_t = (1 - t.view(-1, 1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1, 1) * z1
        v_pred = cfm(z_t, t)
        target_v = z1 - z0
        cfm_loss = F.mse_loss(v_pred, target_v)
        flow = deform_gen(mu)
        deformed = spatial_transformer(t1, flow)
        sim_loss = F.l1_loss(deformed, t1)
        sm_loss = smooth_loss_fn(flow)
        jac_loss = compute_jacobian_penalty(flow)
        def_loss = sim_loss + args.smooth_weight * sm_loss + args.jacobian_weight * jac_loss
        loss = (
            args.recon_weight * recon_loss
            + cls_loss
            + args.cfm_weight * cfm_loss
            + args.def_weight * def_loss
        )
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_cfm += cfm_loss.item()
        total_def += def_loss.item()
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "loss": total_loss / n,
        "recon": total_recon / n,
        "cfm": total_cfm / n,
        "def": total_def / n,
    }


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, num_gpus={args.num_gpus}")

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Data
    train_tf = get_multimodal_train_transforms()
    val_tf = get_multimodal_val_transforms()
    train_loader, val_loader = _build_dataloaders(args, train_tf, val_tf)

    # Model (Multi-Modal VAE with Plan C fMRI temporal encoder)
    model = _build_model(args, device)
    _load_encoder_weights(model, args.encoder_checkpoint, device)

    # CFM velocity field
    cfm = VelocityFieldNet(
        latent_channels=args.latent_channels,
        latent_spatial=tuple(s // 16 for s in MULTI_MODAL_SPATIAL_SIZES["t1"]),
        use_demographics=False,
    ).to(device)
    if args.cfm_checkpoint and os.path.exists(args.cfm_checkpoint):
        print(f"[Stage 5] Loading CFM from {args.cfm_checkpoint}")
        ck = torch.load(args.cfm_checkpoint, map_location=device, weights_only=False)
        from utils.stage23_compat import shape_filtered_load_state_dict
        shape_filtered_load_state_dict(cfm, ck.get("model_state_dict", ck), strict=False, verbose=True)

    # Deformation generator + spatial transformer
    deform_gen = DeformationGenerator(
        latent_channels=args.latent_channels,
        spatial_size=tuple(MULTI_MODAL_SPATIAL_SIZES["t1"]),
    ).to(device)
    spatial_transformer = SpatialTransformer(
        size=tuple(MULTI_MODAL_SPATIAL_SIZES["t1"]),
    ).to(device)
    if args.deform_checkpoint and os.path.exists(args.deform_checkpoint):
        print(f"[Stage 5] Loading deformation from {args.deform_checkpoint}")
        ck = torch.load(args.deform_checkpoint, map_location=device, weights_only=False)
        from utils.stage23_compat import shape_filtered_load_state_dict
        shape_filtered_load_state_dict(deform_gen, ck.get("model_state_dict", ck), strict=False, verbose=True)

    # Loss
    smooth_loss_fn = GradientSmoothingLoss()

    # Multi-GPU: wrap the VAE encoder (the heaviest module).
    from utils.multi_gpu import setup_data_parallel
    model = setup_data_parallel(model, args.num_gpus)
    # cfm and deform_gen are smaller; keep them on a single GPU for speed.

    # Optimizer with per-module LR multipliers
    encoder_params = list(model.parameters())
    cfm_params = list(cfm.parameters())
    deform_params = list(deform_gen.parameters()) + list(spatial_transformer.parameters())
    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": args.learning_rate * args.encoder_lr_mult},
            {"params": cfm_params, "lr": args.learning_rate * args.cfm_lr_mult},
            {"params": deform_params, "lr": args.learning_rate * args.deform_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # AMP is disabled by default (canonical 2x RTX 3090 setup).
    use_amp = getattr(args, "use_amp", False) and not args.no_amp
    scaler = GradScaler("cuda") if use_amp else None

    # Training loop
    log_file = os.path.join(args.output_dir, "train_log.csv")
    write_header = True
    best_loss = float("inf")
    best_epoch = 0
    print(f"Starting Stage 5 joint fine-tuning for {args.epochs} epochs")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(
            model, cfm, deform_gen, spatial_transformer, train_loader,
            optimizer, device, args, smooth_loss_fn,
        )
        val_m = validate(
            model, cfm, deform_gen, spatial_transformer, val_loader,
            device, args, smooth_loss_fn,
        )
        scheduler.step()
        elapsed = time.time() - t0

        is_best = val_m["loss"] < best_loss
        if is_best:
            best_loss = val_m["loss"]
            best_epoch = epoch

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_m['loss']:.4f} recon={train_m['recon']:.4f} "
            f"cfm={train_m['cfm']:.4f} def={train_m['def']:.4f} | "
            f"val loss={val_m['loss']:.4f} recon={val_m['recon']:.4f} "
            f"cfm={val_m['cfm']:.4f} def={val_m['def']:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s | best={best_loss:.4f}@e{best_epoch}"
        )

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "epoch", "train_loss", "train_recon", "train_cfm", "train_def",
                    "val_loss", "val_recon", "val_cfm", "val_def",
                    "lr", "elapsed_s", "is_best",
                ])
                write_header = False
            writer.writerow([
                epoch,
                f"{train_m['loss']:.6f}", f"{train_m['recon']:.6f}",
                f"{train_m['cfm']:.6f}", f"{train_m['def']:.6f}",
                f"{val_m['loss']:.6f}", f"{val_m['recon']:.6f}",
                f"{val_m['cfm']:.6f}", f"{val_m['def']:.6f}",
                f"{scheduler.get_last_lr()[0]:.6e}", f"{elapsed:.2f}", int(is_best),
            ])

        # Save best joint checkpoint
        if is_best:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "cfm_state_dict": cfm.state_dict(),
                "deform_gen_state_dict": deform_gen.state_dict(),
                "spatial_transformer_state_dict": spatial_transformer.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "best_loss": best_loss,
            }, os.path.join(args.output_dir, "joint_best.pt"))
            print(f"  >> Saved joint_best.pt (loss={best_loss:.4f})")

        if epoch - best_epoch >= args.early_stopping:
            print(f"Early stopping: no improvement for {args.early_stopping} epochs")
            break

    print(f"Stage 5 done. Best val loss = {best_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
