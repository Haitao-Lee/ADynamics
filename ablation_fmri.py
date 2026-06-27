"""
ablation_fmri.py — Compare fMRI encoder variants on AD classification.

Trains Stage 1 for 1-2 epochs on each configuration and reports the
val_cls_acc difference. Configurations:

  1. T1+FLAIR only  (no fMRI, baseline for 2-modality)
  2. T1+FLAIR+fMRI with OLD encoder (fMRITemporalEncoder: 91K params, 3 Conv1D + 2 Transformer)
  3. T1+FLAIR+fMRI with NEW encoder (fMRIDeepEncoder: 324K params, 5 dilated Conv1D
     + 3 Transformer + FC head)
  4. T1+FLAIR+fMRI+ASL+QSM (full 5-modality with new fMRI)

Run:
    python ablation_fmri.py --epochs 2

Outputs:
    Prints a side-by-side comparison table.
    Saves per-config training_log.csv to ./ablation_results/<config_name>/

The "winner" is determined by val_cls_acc (higher is better).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path


CONFIGS = [
    {
        "name": "T1+FLAIR",
        "use_fmri": False,
        "use_asl": False,
        "use_qsm": False,
        "use_flair": True,
        "use_fmri_deep": False,
        "description": "Baseline 2-modality (no fMRI, no ASL, no QSM).",
    },
    {
        "name": "T1+FLAIR+fMRI_OLD",
        "use_fmri": True,
        "use_asl": False,
        "use_qsm": False,
        "use_flair": True,
        "use_fmri_deep": False,  # use fMRITemporalEncoder
        "description": "3-modality with OLD fMRI encoder (lightweight, may underfit).",
    },
    {
        "name": "T1+FLAIR+fMRI_DEEP",
        "use_fmri": True,
        "use_asl": False,
        "use_qsm": False,
        "use_flair": True,
        "use_fmri_deep": True,  # use fMRIDeepEncoder
        "description": "3-modality with NEW deep fMRI encoder (multi-scale + FC).",
    },
    {
        "name": "T1+FLAIR+fMRI_DEEP+ASL+QSM",
        "use_fmri": True,
        "use_asl": True,
        "use_qsm": True,
        "use_flair": True,
        "use_fmri_deep": True,
        "description": "Full 5-modality with deep fMRI. Expected best (subject to compute).",
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2,
                    help="Epochs per config (smoke test: 1, signal: 5+)")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--output_root", default="./ablation_results",
                    help="Where to write per-config outputs.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip configs that already have results.")
    ap.add_argument("--data_dir", default=None,
                    help="Override data_dir (default: same as run_01_train.ps1)")
    ap.add_argument("--py", default="C:\\SoftwareInstallationFile\\Anaconda\\envs\\ADynamics\\python.exe")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Build per-config CLI args
    summary = []
    for cfg in CONFIGS:
        out_dir = output_root / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        log_csv = out_dir / "train_log.csv"

        print("=" * 70)
        print(f"CONFIG: {cfg['name']}")
        print(f"  {cfg['description']}")
        print("=" * 70)

        if args.skip_existing and log_csv.exists():
            print(f"  Skipping (--skip_existing): {log_csv} exists")
        else:
            cmd = [
                args.py, "-u",
                str(project_root / "scripts" / "train_stage1_multimodal.py"),
                "--config", "./configs/stage1_vae.yaml",
                "--output_dir", str(out_dir),
                "--num_gpus", "2",
                "--no_amp",
                "--batch_size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--learning_rate", str(args.lr),
                # Modality switches
            ]
            # Modality on/off flags
            if not cfg["use_fmri"]:
                cmd.append("--no_fmri")
            if not cfg["use_asl"]:
                cmd.append("--no_asl")
            if not cfg["use_qsm"]:
                cmd.append("--no_qsm")
            if not cfg["use_flair"]:
                cmd.append("--no_flair")
            # fMRI encoder choice
            if cfg["use_fmri_deep"]:
                cmd.append("--use_fmri_deep")
            else:
                cmd.append("--no_fmri_deep")

            if args.data_dir:
                cmd.extend(["--data_dir", args.data_dir])

            print(f"  CMD: {' '.join(cmd)}")
            t0 = time.time()
            result = subprocess.run(cmd, cwd=str(project_root))
            dt = time.time() - t0
            print(f"  Training took {timedelta(seconds=int(dt))} (exit={result.returncode})")

        # Read the final epoch's val_cls_acc from the CSV
        best_val_acc = float("nan")
        last_val_loss = float("nan")
        if log_csv.exists():
            with open(log_csv) as f:
                rows = list(csv.DictReader(f))
            if rows:
                # Find best val_acc
                best = max(rows, key=lambda r: float(r.get("val_acc", 0)))
                best_val_acc = float(best["val_acc"])
                # Last row's loss
                last = rows[-1]
                last_val_loss = float(last.get("val_loss", "nan"))
        summary.append({
            "config": cfg["name"],
            "description": cfg["description"],
            "best_val_acc": best_val_acc,
            "last_val_loss": last_val_loss,
        })

    # Print comparison table
    print()
    print("=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Config':<40s}  {'best_val_acc':>14s}  {'last_val_loss':>14s}")
    print("-" * 70)
    for s in summary:
        acc_str = f"{s['best_val_acc']:.4f}" if not (s['best_val_acc'] != s['best_val_acc']) else "  n/a"
        loss_str = f"{s['last_val_loss']:.4f}" if not (s['last_val_loss'] != s['last_val_loss']) else "  n/a"
        print(f"{s['config']:<40s}  {acc_str:>14s}  {loss_str:>14s}")
    print()
    # Save summary as JSON
    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
