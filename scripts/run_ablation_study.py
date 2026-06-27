"""
Ablation Study Runner for ADynamics Stage 1.

Runs multiple training configurations and collects metrics for comparison.

Ablation dimensions:
    1. Modality: T1-only | T1+FLAIR | T1+fMRI+FLAIR | T1+all (5-mod)
    2. Fusion:   T1-centric (recommended) | Legacy concat
    3. Loss:     Full | No ordinal CE | No contrastive | No SSIM

Each run produces:
    - val_cls_acc (higher is better)
    - val_recon_l1 (lower is better)
    - silhouette_score (higher = better class separation in latent)
    - kl_per_dim (posterior collapse check)

Usage:
    # Run all ablations sequentially
    python scripts/run_ablation_study.py --output_dir ./ablation_results

    # Run specific ablation group
    python scripts/run_ablation_study.py --group modality

    # Dry run: print commands without executing
    python scripts/run_ablation_study.py --dry_run
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AblationConfig:
    """One ablation experiment configuration."""
    name: str
    group: str  # "modality" | "fusion" | "loss"
    description: str
    extra_args: List[str] = field(default_factory=list)
    expected_runtime_hours: float = 3.0


# ─── Ablation Definitions ────────────────────────────────────────────

MODALITY_ABLATIONS = [
    AblationConfig(
        name="t1_only",
        group="modality",
        description="T1-only baseline (most conservative)",
        extra_args=["--t1_only"],
    ),
    AblationConfig(
        name="t1_flair",
        group="modality",
        description="T1 + FLAIR (recommended clean baseline)",
        extra_args=["--no_fmri", "--no_asl", "--no_qsm"],
    ),
    AblationConfig(
        name="t1_fmri_flair",
        group="modality",
        description="T1 + fMRI + FLAIR (functional + structural)",
        extra_args=["--no_asl", "--no_qsm"],
    ),
    AblationConfig(
        name="t1_all",
        group="modality",
        description="Full 5-modality (T1 + fMRI + ASL + QSM + FLAIR)",
        extra_args=[],
    ),
]

FUSION_ABLATIONS = [
    AblationConfig(
        name="t1_centric",
        group="fusion",
        description="T1-centric fusion (recommended, gated residual deltas)",
        extra_args=[],  # default is t1_centric=True
    ),
    AblationConfig(
        name="legacy_concat",
        group="fusion",
        description="Legacy concat fusion (baseline comparison)",
        extra_args=["--no_t1_centric_fusion"],
    ),
]

LOSS_ABLATIONS = [
    AblationConfig(
        name="full_loss",
        group="loss",
        description="Full loss (recon + ordinal CE + KL + contrastive + SSIM)",
        extra_args=[],
    ),
    AblationConfig(
        name="no_ordinal_ce",
        group="loss",
        description="Remove ordinal CE (use standard CE instead)",
        extra_args=["--cls_loss_type", "standard_ce"],
    ),
    AblationConfig(
        name="no_contrastive",
        group="loss",
        description="Remove ordinal contrastive loss",
        extra_args=["--contrastive_weight", "0.0"],
    ),
    AblationConfig(
        name="no_ssim",
        group="loss",
        description="Remove SSIM loss (L1 only + KL + CE)",
        extra_args=["--ssim_weight", "0.0"],
    ),
]

ALL_ABLATIONS = MODALITY_ABLATIONS + FUSION_ABLATIONS + LOSS_ABLATIONS


def build_command(
    config: AblationConfig,
    base_args: Dict[str, Any],
    output_dir: str,
) -> List[str]:
    """Build the training command for one ablation."""
    exp_dir = os.path.join(output_dir, config.group, config.name)

    cmd = [
        sys.executable,
        "scripts/train_stage1_multimodal.py",
        "--config", base_args.get("config", "./configs/stage1_vae.yaml"),
        "--output_dir", exp_dir,
        "--num_gpus", str(base_args.get("num_gpus", 2)),
        "--batch_size", str(base_args.get("batch_size", 8)),
        "--accumulation_steps", str(base_args.get("accumulation_steps", 1)),
        "--epochs", str(base_args.get("epochs", 300)),
        "--use_checkpointing",
        "--no_amp",
    ]

    # Add modality flags (always start from full 5-mod base)
    # Ablation configs add their specific flags on top
    cmd.extend(config.extra_args)

    return cmd


def run_ablation(
    config: AblationConfig,
    base_args: Dict[str, Any],
    output_dir: str,
    dry_run: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run one ablation and return results."""
    exp_dir = os.path.join(output_dir, config.group, config.name)
    os.makedirs(exp_dir, exist_ok=True)

    cmd = build_command(config, base_args, output_dir)
    log_path = os.path.join(exp_dir, "train.log")

    print(f"\n{'='*60}")
    print(f"Ablation: {config.group}/{config.name}")
    print(f"Description: {config.description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}")

    if dry_run:
        return {"name": config.name, "group": config.group, "status": "dry_run"}

    t0 = time.time()
    try:
        with open(log_path, "w") as log_f:
            log_f.write(f"# Ablation: {config.name}\n")
            log_f.write(f"# Command: {' '.join(cmd)}\n\n")
            log_f.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            proc.wait()

        elapsed = (time.time() - t0) / 3600
        status = "success" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
        print(f"  Status: {status} ({elapsed:.1f}h)")

    except Exception as e:
        elapsed = (time.time() - t0) / 3600
        status = f"error: {e}"
        print(f"  Error: {e}")

    return {
        "name": config.name,
        "group": config.group,
        "description": config.description,
        "status": status,
        "runtime_hours": elapsed,
        "exp_dir": exp_dir,
        "log_path": log_path,
    }


def collect_results(output_dir: str) -> List[Dict[str, Any]]:
    """Collect results from completed ablation runs."""
    results = []
    for group in ["modality", "fusion", "loss"]:
        group_dir = os.path.join(output_dir, group)
        if not os.path.isdir(group_dir):
            continue
        for name in os.listdir(group_dir):
            exp_dir = os.path.join(group_dir, name)
            if not os.path.isdir(exp_dir):
                continue

            result = {"group": group, "name": name, "exp_dir": exp_dir}

            # Look for trainer_log.csv
            csv_path = os.path.join(exp_dir, "trainer_log.csv")
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last = rows[-1]
                        result["final_epoch"] = len(rows)
                        result["val_cls_acc"] = float(last.get("val_cls_acc", 0))
                        result["val_recon_l1"] = float(last.get("val_recon_l1", 0))
                        result["kl_weight"] = float(last.get("kl_weight", 0))

            # Look for best checkpoint info
            ckpt_path = os.path.join(exp_dir, "vae_best.pt")
            result["has_checkpoint"] = os.path.exists(ckpt_path)

            results.append(result)

    return results


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print a comparison table across ablation groups."""
    by_group: Dict[str, List[Dict]] = {}
    for r in results:
        by_group.setdefault(r["group"], []).append(r)

    for group, group_results in sorted(by_group.items()):
        print(f"\n{'='*60}")
        print(f"Ablation Group: {group}")
        print(f"{'='*60}")
        print(f"{'Name':<20} {'val_cls_acc':>12} {'val_recon_l1':>12} {'epochs':>8} {'ckpt':>6}")
        print("-" * 60)
        for r in sorted(group_results, key=lambda x: x.get("val_cls_acc", 0), reverse=True):
            acc = r.get("val_cls_acc", -1)
            recon = r.get("val_recon_l1", -1)
            epochs = r.get("final_epoch", "?")
            has_ckpt = "✓" if r.get("has_checkpoint") else "✗"
            acc_str = f"{acc:.4f}" if acc >= 0 else "N/A"
            recon_str = f"{recon:.4f}" if recon >= 0 else "N/A"
            print(f"{r['name']:<20} {acc_str:>12} {recon_str:>12} {str(epochs):>8} {has_ckpt:>6}")


def main():
    parser = argparse.ArgumentParser(description="Run ADynamics ablation study")
    parser.add_argument("--output_dir", type=str, default="./ablation_results")
    parser.add_argument("--group", type=str, default=None,
                        choices=["modality", "fusion", "loss", "all"],
                        help="Run only one ablation group")
    parser.add_argument("--config", type=str, default="./configs/stage1_vae.yaml")
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--collect_only", action="store_true",
                        help="Only collect results from existing runs")
    args = parser.parse_args()

    base_args = {
        "config": args.config,
        "num_gpus": args.num_gpus,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "epochs": args.epochs,
    }

    if args.collect_only:
        results = collect_results(args.output_dir)
        print_comparison_table(results)
        # Save
        summary_path = os.path.join(args.output_dir, "comparison.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved: {summary_path}")
        return

    # Select which ablations to run
    if args.group == "modality" or args.group is None:
        ablations = list(MODALITY_ABLATIONS)
    else:
        ablations = []

    if args.group == "fusion" or args.group is None:
        ablations.extend(FUSION_ABLATIONS)

    if args.group == "loss" or args.group is None:
        ablations.extend(LOSS_ABLATIONS)

    if args.group == "all":
        ablations = list(ALL_ABLATIONS)

    print(f"Running {len(ablations)} ablation experiments")
    total_hours = sum(a.expected_runtime_hours for a in ablations)
    print(f"Estimated total time: {total_hours:.1f} hours")

    results = []
    for config in ablations:
        result = run_ablation(config, base_args, args.output_dir, dry_run=args.dry_run)
        if result:
            results.append(result)

    if not args.dry_run:
        # Collect and compare
        all_results = collect_results(args.output_dir)
        print_comparison_table(all_results)

        summary_path = os.path.join(args.output_dir, "comparison.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nSaved: {summary_path}")
    else:
        print("\n[DRY RUN] Commands printed above. Remove --dry_run to execute.")


if __name__ == "__main__":
    main()
