"""
Stage 3: Conditional Flow Matching for Multi-Modal VAE.

Freeze the pretrained multi-modal encoder from Stage 1, extract latent representations
for all disease stages (NC/SCD/MCI/AD), then train a VelocityFieldNet using
Conditional Flow Matching to model disease progression in latent space.

The CFM learns to morph between disease stages:
    - NC (0) ->?SCD (1) ->?MCI (2) ->?AD (3)
    - Training pairs: NC->MCI, NC->AD, SCD->MCI, SCD->AD, MCI->AD, etc.

CFM Loss: L = || v_theta(z_t, t) - (z_target - z_source) ||^2
where z_t = (1-t)*z_source + t*z_target (linear interpolation)

Usage:
    python scripts/train_stage3_cfm.py \
        --encoder_checkpoint ./checkpoints/stage1_multimodal/vae_best.pt \
        --json ./core_data/dataset_manifest_merged_v2.json \
        --output_dir ./checkpoints/stage3_cfm
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from core_data.dataset import MultiModalDataset
from core_data.transforms import get_multimodal_train_transforms, get_multimodal_val_transforms, MULTI_MODAL_SPATIAL_SIZES
from models.vae3d import MultiModalVAE3D
from models.vector_field import VelocityFieldNet
from engine.trainer_cfm import CFMTrainer


def parse_args():
    from utils.config_loader import apply_yaml_defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None, help="YAML config file")
    pre_args, _ = pre.parse_known_args()

    mapping = [
        (("input", "encoder_checkpoint"), "encoder_checkpoint"),
        (("input", "num_classes"), "num_classes"),
        (("model", "latent_channels"), "latent_channels"),
        (("model", "base_channels"), "base_channels"),
        (("model", "decoder_depth"), "decoder_depth"),
        (("model", "latent_spatial"), "latent_spatial"),
        (("model", "time_embed_dim"), "time_embed_dim"),
        (("model", "time_hidden_dim"), "time_hidden_dim"),
        (("model", "cond_embed_dim"), "cond_embed_dim"),
        (("model", "cond_hidden_dim"), "cond_hidden_dim"),
        (("model", "num_conditions"), "num_conditions"),
        (("model", "channel_mults"), "channel_mults"),
        (("model", "num_res_blocks"), "num_res_blocks"),
        (("model", "use_demographics"), "use_demographics"),
        (("training", "batch_size"), "batch_size"),
        (("training", "learning_rate"), "learning_rate"),
        (("training", "weight_decay"), "weight_decay"),
        (("training", "epochs"), "epochs"),
        (("training", "early_stopping_patience"), "early_stopping"),
        (("training", "num_gpus"), "num_gpus"),
        (("training", "use_amp"), "use_amp"),
        (("loss", "velocity_loss_weight"), "velocity_loss_weight"),
        (("loss", "rectified_flow_weight"), "rectified_flow_weight"),
        (("cfm", "forward_only"), "cfm_forward_only"),
        (("cfm", "distance_aware"), "cfm_distance_aware"),
        (("output", "dir"), "output_dir"),
        (("seed",), "seed"),
    ]
    config_defaults = apply_yaml_defaults(pre_args.config, mapping) if pre_args.config else {}

    parser = argparse.ArgumentParser(description="Stage 3 CFM Training", parents=[pre])
    parser.add_argument("--json", type=str, default="./core_data/dataset_manifest_merged_v2.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage3_cfm")
    parser.add_argument("--encoder_checkpoint", type=str,
                        default="./checkpoints/stage1_multimodal/vae_best.pt",
                        help="Path to Stage 1 checkpoint (encoder weights)")
    parser.add_argument("--decoder_checkpoint", type=str, default=None,
                        help="Optional: Path to Stage 2 decoder checkpoint")
    parser.add_argument("--latent_channels", type=int, default=32,
                        help="Must match Stage 1 latent_channels")
    parser.add_argument("--base_channels", type=int, default=16,
                        help="Must match Stage 1 base_channels")
    parser.add_argument("--decoder_depth", type=int, default=4,
                        help="Must match Stage 1 decoder_depth")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD)")
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    # CFM model params
    parser.add_argument("--cfm_base_channels", type=int, default=64,
                        help="Base channels for VelocityFieldNet")
    parser.add_argument("--time_embed_dim", type=int, default=128,
                        help="Time embedding dimension")
    parser.add_argument("--cond_embed_dim", type=int, default=64,
                        help="Condition embedding dimension")

    # Training
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for CFM training (pairs per batch)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--velocity_loss_weight", type=float, default=1.0)
    parser.add_argument("--rectified_flow_weight", type=float, default=0.0,
                        help="Rectified flow regularization weight (0=disabled, try 0.01)")
    parser.add_argument("--no_distance_aware", action="store_true", default=False,
                        help="Disable distance-aware sampling (use uniform pair sampling)")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--early_stopping", type=int, default=50)
    parser.add_argument("--no_amp", action="store_true", default=False)
    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="Enable AMP (default OFF; YAML use_amp: false maps here)")

    # Modality toggles (must match Stage 1)
    from utils.stage23_compat import add_modality_args
    add_modality_args(parser)
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of GPUs for DataParallel (default 2; canonical setup is 2x RTX 3090)")
    # Apply YAML config defaults AFTER all add_argument calls
    # (set_defaults must come last so it isn't overridden by argparse defaults)
    parser.set_defaults(**config_defaults)
    return parser.parse_args()


def load_data(json_path: str) -> list:
    """Load and validate multi-modal dataset."""
    import nibabel as nib
    from monai.transforms import LoadImaged, EnsureChannelFirstd, Orientationd, Compose

    with open(json_path, "r") as f:
        data = json.load(f)

    quick_transforms = Compose([
        LoadImaged(keys=['t1'], reader='NibabelReader'),
        EnsureChannelFirstd(keys=['t1']),
        Orientationd(keys=['t1'], axcodes='RAS'),
    ])

    valid_data = []
    corrupted = 0
    for item in data:
        t1_path = item.get("t1")
        if not t1_path or not os.path.exists(t1_path):
            continue
        try:
            img = nib.load(t1_path)
            if any(s == 0 for s in img.shape):
                corrupted += 1
                continue
            img_data = img.get_fdata()
            if img_data.min() == img_data.max():
                corrupted += 1
                continue
            quick_transforms({'t1': str(t1_path)})
        except Exception:
            corrupted += 1
            continue
        valid_data.append(item)

    if corrupted > 0:
        print(f"Warning: Skipped {corrupted} corrupted T1 files")
    print(f"Loaded {len(valid_data)} valid samples")
    return valid_data


CLASS_NAMES_MAP = {
    3: ["NC", "SCD+MCI", "AD"],
    4: ["NC", "SCD", "MCI", "AD"],
}


def build_latent_pools(encoder, dataloader, device, num_classes=4):
    """
    Encode all samples and build per-class latent pools.

    Returns:
        dict of {class_id: list of latent tensors}
    """
    encoder.eval()
    pools = {c: [] for c in range(num_classes)}
    class_names = CLASS_NAMES_MAP.get(num_classes, [f"Class_{i}" for i in range(num_classes)])

    with torch.no_grad():
        for batch in dataloader:
            t1 = batch["t1"].to(device)
            x_dict = {"t1": t1}
            for mod in ["fmri", "asl", "qsm", "flair"]:
                if mod in batch and batch[mod] is not None:
                    mod_tensor = batch[mod].to(device)
                    if mod == "fmri":
                        from utils.stage23_compat import normalize_fmri_batch
                        mod_tensor = normalize_fmri_batch(mod_tensor)
                    x_dict[mod] = mod_tensor

            labels = batch.get("label", torch.tensor([])).to(device)
            if labels.dim() > 1:
                labels = labels.squeeze()

            # Encode to get mu (frozen encoder, no reparameterization needed)
            _, _, mu, _ = encoder(x_dict, return_components=True)

            for i, label in enumerate(labels):
                lbl = int(label.item())
                pools[lbl].append(mu[i].cpu())

    for c in range(num_classes):
        pools[c] = [z.to(device) for z in pools[c]]
        print(f"  {class_names[c]}: {len(pools[c])} samples")

    return pools


class MultiClassCFMTrainer(CFMTrainer):
    """
    Extended CFMTrainer with FORWARD-ONLY disease stage flows.

    Enforces the biological constraint that disease progression is unidirectional:
        NC(0) ->?SCD+MCI(1) ->?AD(2)   [3-class mode]
        or NC(0) ->?SCD(1) ->?MCI(2) ->?AD(3)   [4-class mode]

    Only samples pairs where src_class < tgt_class (forward flow).
    Uses distance-aware sampling: adjacent stages (NC->SCD+MCI) are sampled more
    frequently than distant stages (NC->AD) to learn fine-grained transitions.

    This prevents the model from learning biologically meaningless reverse flows
    that would corrupt the velocity field.
    """

    def __init__(self, *args, latent_pools, class_names, distance_aware=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_pools = latent_pools  # {class_id: [tensor, ...]}
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.distance_aware = distance_aware

        # Build forward-only pair list with distance-aware weights
        self._build_pair_distribution()

    def _build_pair_distribution(self):
        """
        Build all valid forward-only pairs (src < tgt) with distance-aware weights.

        Closer pairs (NC->SCD, distance=1) get higher sampling weight than
        distant pairs (NC->AD, distance=3). This ensures the model learns
        fine-grained transitions, not just the extreme NC->AD mapping.

        Weight scheme: weight = 1 / distance^alpha
            alpha=1:  NC->SCD=1.0, NC->MCI=0.5, NC->AD=0.33
            alpha=0.5: NC->SCD=1.0, NC->MCI=0.71, NC->AD=0.58
        """
        self.valid_pairs = []
        self.pair_weights = []
        alpha = 1.0  # Distance decay exponent

        for src in range(self.num_classes):
            for tgt in range(src + 1, self.num_classes):
                if len(self.latent_pools[src]) > 0 and len(self.latent_pools[tgt]) > 0:
                    distance = tgt - src
                    weight = 1.0 / (distance ** alpha) if self.distance_aware else 1.0
                    self.valid_pairs.append((src, tgt))
                    self.pair_weights.append(weight)

        if not self.valid_pairs:
            raise RuntimeError("No valid forward-only pairs found. Check latent pools.")

        # Normalize weights to probability distribution
        total = sum(self.pair_weights)
        self.pair_probs = [w / total for w in self.pair_weights]

        print(f"Forward-only CFM pairs ({len(self.valid_pairs)}):")
        for (src, tgt), prob in zip(self.valid_pairs, self.pair_probs):
            n_pairs = len(self.latent_pools[src]) * len(self.latent_pools[tgt])
            print(f"  {self.class_names[src]} ->?{self.class_names[tgt]}: "
                  f"{n_pairs} pairs, sampling prob={prob:.3f}")

    def sample_latent_pairs(self, batch_size: int):
        """
        Sample forward-only disease stage pairs (src_class < tgt_class).

        Uses distance-aware sampling: adjacent transitions are more likely
        than distant transitions, ensuring fine-grained flow learning.

        Returns:
            Tuple of (z0, z1, None, None, None, None)
        """
        if not self.valid_pairs:
            raise RuntimeError("No valid pairs available.")

        # Select a pair type based on distance-aware probabilities
        pair_idx = torch.multinomial(
            torch.tensor(self.pair_probs), 1
        ).item()
        src_class, tgt_class = self.valid_pairs[pair_idx]

        src_pool = self.latent_pools[src_class]
        tgt_pool = self.latent_pools[tgt_class]

        # Sample individual latents from each class
        src_indices = torch.randint(0, len(src_pool), (batch_size,))
        tgt_indices = torch.randint(0, len(tgt_pool), (batch_size,))

        z0 = torch.stack([src_pool[i] for i in src_indices])
        z1 = torch.stack([tgt_pool[i] for i in tgt_indices])

        # Log which classes are being paired (for debugging)
        if torch.rand(1).item() < 0.01:
            print(f"  CFM pair: {self.class_names[src_class]} ->?{self.class_names[tgt_class]}")

        return z0, z1, None, None, None, None


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_list = load_data(args.json)
    print(f"Total samples: {len(data_list)}")

    # Remap 4-class labels to 3-class (SCD+MCI merged) if needed
    if args.num_classes == 3:
        from utils.config_loader import remap_labels_3class
        remap_labels_3class(data_list)
        print("Remapped labels to 3-class (NC / SCD+MCI / AD)")

    # Transforms for encoding (no augmentation, just preprocessing)
    val_transforms = get_multimodal_val_transforms()
    dataset = MultiModalDataset(data_list, transform=val_transforms)

    from core_data.dataset import multimodal_collate_fn
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0,
        collate_fn=multimodal_collate_fn
    )

    # Load encoder from Stage 1 checkpoint
    from utils.stage23_compat import resolve_optional_modalities, resolve_use_demographic
    optional_modalities = resolve_optional_modalities(args)
    print(f"[Modality switches] optional={optional_modalities}")
    encoder = MultiModalVAE3D(
        spatial_size=MULTI_MODAL_SPATIAL_SIZES["t1"],
        in_channels=1,
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        decoder_depth=args.decoder_depth,
        optional_modalities=optional_modalities,
        use_demographic_cond=resolve_use_demographic(args),
    )

    print(f"Loading encoder from {args.encoder_checkpoint}")
    checkpoint = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    sd = checkpoint["model_state_dict"]

    from utils.stage23_compat import shape_filtered_load_state_dict
    shape_filtered_load_state_dict(encoder, sd, strict=False, verbose=True)
    encoder = encoder.to(device)
    encoder.eval()

    # Multi-GPU: wrap encoder with DataParallel
    from utils.multi_gpu import setup_data_parallel
    encoder = setup_data_parallel(encoder, args.num_gpus)

    # Freeze encoder completely
    for param in encoder.parameters():
        param.requires_grad = False
    print("Encoder loaded and frozen")

    # Build latent pools for all 4 disease stages
    print("\nBuilding latent pools for all disease stages...")
    latent_pools = build_latent_pools(encoder, dataloader, device, num_classes=args.num_classes)

    # Check pool sizes
    total_pairs = 0
    for c in range(args.num_classes):
        for c2 in range(args.num_classes):
            if c != c2:
                total_pairs += len(latent_pools[c]) * len(latent_pools[c2])

    print(f"\nTotal available pairs (cross-class): {total_pairs}")
    if total_pairs < 100:
        print("WARNING: Very few pairs available. Consider using same-class pairs too.")
        print("         Using same-class pairs for more training data.")

    # Also build same-class pools for fallback
    same_class_pools = {c: latent_pools[c] for c in range(args.num_classes)}

    class_names = CLASS_NAMES_MAP.get(args.num_classes, [f"Class_{i}" for i in range(args.num_classes)])

    # Create CFM model
    latent_spatial = (16, 16, 12)  # After 4 downsamples from 256x256x192
    cfm_model = VelocityFieldNet(
        latent_channels=args.latent_channels,
        latent_spatial=latent_spatial,
        time_embed_dim=args.time_embed_dim,
        cond_embed_dim=args.cond_embed_dim,
        base_channels=args.cfm_base_channels,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        use_demographics=False,
    ).to(device)

    print(f"CFM model params: {sum(p.numel() for p in cfm_model.parameters()) / 1e6:.1f}M")

    # Optimizer
    optimizer = AdamW(
        cfm_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Config
    config = {
        "batch_size": args.batch_size,
        "velocity_loss_weight": args.velocity_loss_weight,
        "use_amp": getattr(args, "use_amp", False) and not args.no_amp,
    }

    use_amp = getattr(args, "use_amp", False) and not args.no_amp

    # Create GradScaler once before training loop (for AMP)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Create trainer with forward-only flows and distance-aware sampling
    trainer = MultiClassCFMTrainer(
        model=cfm_model,
        optimizer=optimizer,
        device=device,
        config=config,
        scheduler=scheduler,
        latent_pools=latent_pools,
        class_names=class_names,
        distance_aware=not args.no_distance_aware,
    )

    # Replace pool sampling with multi-class version
    trainer.sample_latent_pairs = lambda bs: MultiClassCFMTrainer.sample_latent_pairs(
        trainer, bs
    )

    print(f"\n{'='*60}")
    print("Stage 3: Conditional Flow Matching (Forward-Only)")
    print(f"{'='*60}")
    print(f"Encoder: {args.encoder_checkpoint}")
    print(f"Latent shape: [{args.latent_channels}, 16, 16, 12]")
    print(f"CFM model: {sum(p.numel() for p in cfm_model.parameters()) / 1e6:.1f}M params")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} pairs/batch")
    print(f"LR: {args.learning_rate}")
    direction_label = "NC->SCD+MCI->AD" if args.num_classes == 3 else "NC->SCD->MCI->AD"
    print(f"Direction: Forward-only ({direction_label})")
    print(f"Distance-aware sampling: {not args.no_distance_aware}")
    print(f"Rectified flow weight: {args.rectified_flow_weight}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop (reuse CFMTrainer's train_epoch structure)
    import csv
    import time

    log_file = os.path.join(args.output_dir, "train_log.csv")
    write_header = not os.path.exists(log_file)

    history = {"train_loss": [], "train_velocity_loss": [], "val_loss": [], "val_velocity_loss": []}
    best_val_loss = float("inf")
    best_epoch = 0
    patience_count = 0

    for epoch in range(args.epochs):
        t0 = time.time()

        # Train epoch
        cfm_model.train()
        train_loss = 0.0
        train_vel = 0.0
        n_train = 0

        # Compute number of batches per epoch
        n_pairs = sum(len(trainer.latent_pools[c]) for c in range(args.num_classes))
        n_batches = max(1, n_pairs * (args.num_classes - 1) // args.batch_size)
        # Cap at reasonable number
        n_batches = min(n_batches, 500)

        for _ in range(n_batches):
            z0, z1, *_ = trainer.sample_latent_pairs(args.batch_size)
            optimizer.zero_grad(set_to_none=True)

            batch_size_actual = z0.shape[0]
            t = torch.rand(batch_size_actual, device=device)
            z_t = (1 - t.view(-1, 1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1, 1) * z1
            v_pred = cfm_model(z_t, t)

            target_v = z1 - z0
            loss = F.mse_loss(v_pred, target_v)

            # Add rectified flow regularization if enabled
            if args.rectified_flow_weight > 0:
                from engine.losses import rectified_flow_regularization
                rf_loss = rectified_flow_regularization(
                    v_pred, z_t, t, cfm_model,
                    lambda_reg=args.rectified_flow_weight
                )
                loss = loss + rf_loss

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(cfm_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cfm_model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            # `train_vel` is the pure CFM velocity MSE — exclude any
            # rectified-flow regularizer so the logged column reflects
            # only the CFM objective (consistent across runs).
            train_vel += F.mse_loss(v_pred, target_v).item()
            n_train += 1

        train_loss /= max(1, n_train)
        train_vel /= max(1, n_train)

        # Val epoch
        cfm_model.eval()
        val_loss = 0.0
        val_vel = 0.0
        n_val = 0

        n_val_batches = min(n_batches, 100)
        with torch.no_grad():
            for _ in range(n_val_batches):
                z0, z1, *_ = trainer.sample_latent_pairs(args.batch_size)
                t = torch.rand(z0.shape[0], device=device)
                z_t = (1 - t.view(-1, 1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1, 1) * z1
                v_pred = cfm_model(z_t, t)
                loss = F.mse_loss(v_pred, z1 - z0)
                val_loss += loss.item()
                val_vel += loss.item()
                n_val += 1

        val_loss /= max(1, n_val)
        val_vel /= max(1, n_val)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"LR: {current_lr:.6f} | "
            f"Train: loss={train_loss:.6f} | "
            f"Val: loss={val_loss:.6f} | "
            f"Time: {epoch_time:.1f}s | "
            f"Best: {best_val_loss:.6f} (epoch {best_epoch+1}) | "
            f"Patience: {patience_count}/{args.early_stopping}"
        )

        # Write log
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "train_loss", "train_velocity_loss",
                               "val_loss", "val_velocity_loss", "lr", "epoch_time", "is_best"])
                write_header = False
            writer.writerow([
                epoch + 1,
                f"{train_loss:.6f}",
                f"{train_vel:.6f}",
                f"{val_loss:.6f}",
                f"{val_vel:.6f}",
                f"{current_lr:.8f}",
                f"{epoch_time:.2f}",
                "1" if is_best else "0",
            ])

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"cfm_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": cfm_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

        if is_best:
            best_path = os.path.join(args.output_dir, "cfm_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": cfm_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "encoder_checkpoint": args.encoder_checkpoint,
            }, best_path)
            print(f"  -> Best model saved (val_loss: {best_val_loss:.6f})")

        if patience_count >= args.early_stopping:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best val_loss: {best_val_loss:.6f} (epoch {best_epoch+1})")
            break

    print(f"\n{'='*60}")
    print("Stage 3 Training Complete")
    print(f"{'='*60}")
    print(f"Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {args.output_dir}/cfm_best.pt")
    print(f"\nInference example:")
    print(f"  z_final, traj = integrate_ode(z0, cfm_model, steps=20)")
    print(f"  recon_evolved = decoder(z_final)")


if __name__ == "__main__":
    main()
