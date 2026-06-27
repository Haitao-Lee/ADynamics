"""
view_stage1_results.py
======================

Library + CLI for visualizing ADynamics Stage 1 training progress.

As a library (called automatically by the trainer at end of each epoch):
    from view_stage1_results import plot_training_curves
    plot_training_curves(csv_path, output_png)

As a CLI (manual run anytime):
    python view_stage1_results.py
    python view_stage1_results.py --output progress_300ep.png

Output:
    - progress.png  (4-panel: train loss, val loss, val acc, per-class acc)
    - Printed table with epoch, train_loss, val_loss, val_acc, lr, kl_w
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Headless matplotlib — import only when needed to keep trainer fast
# when --no_progress_plot is set.
def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def find_log_paths(project_root: Path) -> Tuple[Path, Optional[Path]]:
    """Find the latest training_log.csv and trainer_log.json."""
    csvs = sorted(project_root.glob("nnssl_results/**/training_log.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise FileNotFoundError("No training_log.csv under nnssl_results/")
    csv_path = csvs[0]
    trainer_log = csv_path.parent / "trainer_log.json"
    if not trainer_log.exists():
        trainer_log = csv_path.parent / "fold_0" / "trainer_log.json"
    if not trainer_log.exists():
        candidates = list(csv_path.parent.rglob("trainer_log.json"))
        trainer_log = candidates[0] if candidates else None
    return csv_path, trainer_log


def parse_csv(csv_path: Path) -> dict:
    """Parse training_log.csv into a dict of lists, one per column."""
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"{csv_path} is empty")
    out = {col: [] for col in rows[0].keys()}
    for r in rows:
        for col, val in r.items():
            try:
                out[col].append(float(val))
            except (ValueError, TypeError):
                out[col].append(val)
    return out


def plot_training_curves(csv_path: Path, output_png: Optional[Path] = None) -> Optional[Path]:
    """
    Read the CSV at `csv_path` and write a 4-panel progress plot to
    `output_png` (default: progress.png next to the CSV).

    Returns the path to the PNG, or None if the CSV has no data.

    Safe to call from inside the trainer: matplotlib is imported lazily,
    and any error is caught and logged rather than crashing training.
    """
    if output_png is None:
        output_png = csv_path.parent / "progress.png"
    try:
        cols = parse_csv(csv_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[plot] {e}")
        return None
    if not cols.get("epoch"):
        return None
    try:
        plt = _import_matplotlib()
    except ImportError as e:
        print(f"[plot] matplotlib not available: {e}")
        return None

    epoch = cols["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"ADynamics Stage 1 — fold 0 (epochs 0..{int(epoch[-1])})", fontsize=14)

    # (0,0) Train/val total loss
    ax = axes[0, 0]
    ax.plot(epoch, cols["train_loss"], label="train_loss", marker="o", ms=3)
    ax.plot(epoch, cols["val_loss"],   label="val_loss",   marker="s", ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Total loss"); ax.legend(); ax.grid(True, alpha=0.3)

    # (0,1) Val accuracy
    ax = axes[0, 1]
    ax.plot(epoch, cols["val_acc"], color="tab:green", marker="o", ms=3, label="val_acc")
    ax.axhline(0.25, color="gray", ls="--", alpha=0.5, label="random (4-class)")
    ax.set_xlabel("epoch"); ax.set_ylabel("val_cls_acc")
    ax.set_title("Validation accuracy"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # (1,0) Val loss components
    ax = axes[1, 0]
    ax.plot(epoch, cols["val_recon"],      label="val_recon",  marker="o", ms=3)
    ax.plot(epoch, cols["val_cls"],        label="val_cls",    marker="s", ms=3)
    ax.plot(epoch, cols["val_kl"],         label="val_kl",     marker="^", ms=3)
    ax.plot(epoch, cols["val_ord_reg"],    label="val_ord",    marker="v", ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Validation loss components"); ax.legend(); ax.grid(True, alpha=0.3)

    # (1,1) Per-class val accuracy
    ax = axes[1, 1]
    plotted_any = False
    for col in cols.keys():
        if col.startswith("val_acc_") and col != "val_acc":
            ax.plot(epoch, cols[col], marker="o", ms=3, label=col.replace("val_acc_", ""))
            plotted_any = True
    if plotted_any:
        ax.set_xlabel("epoch"); ax.set_ylabel("val_cls_acc")
        ax.set_title("Per-class val accuracy"); ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "(per-class accuracy not yet logged)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(output_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return output_png


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", default="E:/LHT_workspace/AD/ADynamics")
    p.add_argument("--output", default=None, help="Output PNG path (default: progress.png next to CSV)")
    args = p.parse_args()

    project_root = Path(args.project_root)
    csv_path, trainer_log_path = find_log_paths(project_root)
    print(f"Loading: {csv_path}")
    if trainer_log_path and trainer_log_path.exists():
        print(f"  +     {trainer_log_path}")
    print()

    cols = parse_csv(csv_path)
    epoch = cols["epoch"]
    print(f"Epochs: {len(epoch)} (current = {int(epoch[-1]) if epoch else 0})")
    print()

    # 1) Per-epoch summary table
    print(f"{'ep':>4}  {'train_loss':>10}  {'val_loss':>9}  {'val_acc':>7}  {'val_recon':>9}  {'lr':>10}  {'kl_w':>6}  {'best':>5}")
    print("-" * 80)
    for i, ep in enumerate(epoch):
        tl = cols["train_loss"][i]
        vl = cols["val_loss"][i]
        va = cols["val_acc"][i]
        vr = cols["val_recon"][i]
        lr = cols["lr"][i]
        kw = cols["kl_weight"][i]
        is_best = cols.get("is_best", [0] * len(epoch))[i]
        marker = " *" if is_best else "  "
        print(f"{int(ep):>4}  {tl:>10.4f}  {vl:>9.4f}  {va:>7.4f}  {vr:>9.4f}  {lr:>10.2e}  {kw:>6.4f}  {marker}")

    # 2) Best metric summary
    if trainer_log_path and trainer_log_path.exists():
        with open(trainer_log_path) as f:
            meta = json.load(f)
        print()
        print(f"Best val_acc:   {meta.get('best_cls_acc', 0):.4f}")
        print(f"Best val_loss:  {meta.get('best_val_loss', 0):.4f}")
        print(f"Current epoch:  {meta.get('current_epoch', 0)}")
        print(f"KL weight:      {meta.get('kl_weight', 0)}")

    # 3) Plot
    out_png = Path(args.output) if args.output else None
    saved = plot_training_curves(csv_path, out_png)
    if saved is None:
        print("No epochs to plot.")
    else:
        print()
        print(f"Plot saved to: {saved}")


if __name__ == "__main__":
    main()
