# ADynamics

<!-- Badges -->
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-red.svg)](https://pytorch.org/)
[![MONAI 1.4+](https://img.shields.io/badge/monai-1.4+-green.svg)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

**ADynamics** models Alzheimer's Disease progression from NC to AD using **Conditional Flow Matching (CFM)** on cross-sectional multi-modal MRI data.

> **Key Insight**: We have cross-sectional data (different patients at different stages), NOT longitudinal data. CFM learns population-level disease trajectories without paired data.

---

## Multi-Modal 5-Stage Pipeline

```
T1 (required) ──→ Encoder_T1 ─────────┐
fMRI (optional) ─→ Encoder_fMRI ──────┤
ASL  (optional) ─→ Encoder_ASL  ──────┼──→ Fusion → Latent z
QSM  (optional) ─→ Encoder_QSM  ──────┤         ↓
FLAIR(optional) ─→ Encoder_FLAIR ─────┘    ┌────┴────┐
                                        Decoder  Classifier
```

| Stage | Script | Goal |
|-------|--------|------|
| **1** | `train_stage1_multimodal.py` | Train multi-modal encoder (recon + cls + KL + contrastive) |
| **2a** | `train_stage2_classifier.py` | Freeze encoder, train classifier head |
| **2b** | `train_stage2_decoder.py` | Freeze encoder, train decoder |
| **3** | `train_stage3_cfm.py` | Train CFM vector field (**forward-only** flows) |
| **4** | `train_stage4_deformation.py` | Train deformation generator |
| **5** | `train_stage5_joint.py` | Joint fine-tuning all modules |

### Key Design Improvements

- **Forward-only CFM**: Only learns NC→SCD→MCI→AD (no reverse flows)
- **Distance-aware sampling**: Adjacent stages (NC→SCD) sampled more frequently than distant (NC→AD)
- **Ordinal contrastive loss**: Enforces disease stage separation in latent space
- **Rectified flow regularization**: Encourages straight, efficient trajectories
- **Baseline comparisons**: Validates CFM vs linear/KNN/regression baselines
- **Cross-validation**: 5-fold stratified CV for reliable metrics
- **Ablation framework**: Systematic component analysis

---

## Quick Start

### Installation

```powershell
# Windows PowerShell
.\install_env.ps1

# Or manual
conda create -n ADynamics python=3.11 -y
conda activate ADynamics
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install monai==1.4.0 nibabel SimpleITK torchdiffeq scikit-learn matplotlib tqdm
```

### Train

```powershell
# Stage 1: Multi-modal VAE (with contrastive loss)
.\run_stage1.ps1

# After Stage 1: Check latent quality
.\run_analysis.ps1

# Stage 2a: Validate encoder with classifier
.\run_stage2a.ps1

# Stage 2b: Improve decoder
.\run_stage2b.ps1

# Stage 3: Train CFM (forward-only, distance-aware)
.\run_stage3.ps1

# Stage 4: Train deformation
.\run_stage4.ps1

# Stage 5: Joint fine-tuning
.\run_stage5.ps1
```

### Validate & Analyze

```powershell
# Run all validations
.\run_validation.ps1

# Baseline comparison (CFM vs linear/KNN/regression)
.\run_baseline.ps1

# Cross-validation (5-fold stratified)
.\run_crossval.ps1

# Ablation experiments
.\run_ablation.ps1
```

---

## Project Structure

```
ADynamics/
├── run_stage1.ps1              # Run scripts (PowerShell)
├── run_stage1_resume.ps1
├── run_analysis.ps1
├── run_stage2a.ps1 / run_stage2b.ps1
├── run_stage3.ps1 / run_stage4.ps1 / run_stage5.ps1
├── run_validation.ps1
├── run_baseline.ps1            # NEW: Baseline comparison
├── run_crossval.ps1            # NEW: Cross-validation
├── run_ablation.ps1            # NEW: Ablation experiments
│
├── core_data/                   # Data layer
│   ├── dataset.py              # MultiModalDataset, collate_fn
│   └── transforms.py           # MONAI preprocessing transforms
│
├── engine/                      # Training layer
│   ├── trainer_vae.py          # MultiModalVAETrainer (with KL + contrastive)
│   ├── trainer_cfm.py          # CFMTrainer
│   └── losses.py               # All loss functions (incl. rectified flow)
│
├── models/                      # Model layer
│   ├── vae3d.py                # MultiModalVAE3D, ModalityEncoder3D
│   ├── vector_field.py         # VelocityFieldNet (FiLM conditioning)
│   └── spatial_transform.py    # DeformationGenerator, SpatialTransformer
│
├── scripts/                     # All Python entry points
│   ├── train_stage1_multimodal.py
│   ├── train_stage2_classifier.py
│   ├── train_stage2_decoder.py
│   ├── train_stage3_cfm.py     # Forward-only CFM with rectified flow
│   ├── train_stage4_deformation.py
│   ├── train_stage5_joint.py
│   ├── run_latent_analysis.py  # Latent analysis (PCA/t-SNE/silhouette)
│   ├── run_cls_validation.py   # Classification validation
│   ├── run_recon_validation.py # Reconstruction validation
│   ├── run_flow_visualization.py # CFM flow visualization + straightness
│   ├── run_deform_validation.py  # Deformation validation
│   ├── run_baseline_comparison.py # NEW: CFM vs baselines
│   ├── run_cross_validation.py   # NEW: 5-fold stratified CV
│   ├── run_ablation.py           # NEW: Systematic ablation
│   └── inference_pipeline.py   # End-to-end inference
│
└── utils/                       # Utilities
    ├── io_utils.py             # NIfTI I/O
    └── preprocessing/          # Denoise, N4, registration
```

---

## Data Format

`dataset_manifest_merged_v2.json`:
```json
[
  {
    "t1": "/path/to/t1.nii.gz",
    "fmri": "/path/to/fmri.nii.gz",
    "asl": "/path/to/asl.nii.gz",
    "qsm": "/path/to/qsm.nii.gz",
    "flair": "/path/to/flair.nii.gz",
    "label": 0
  }
]
```

Labels: `0=NC, 1=SCD, 2=MCI, 3=AD`

T1 is required. Other modalities are optional (model handles missing modalities via dropout).

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--latent_channels` | 32 | Latent channels per modality encoder |
| `--base_channels` | 32 | Encoder base channels |
| `--decoder_depth` | 4 | Decoder upsampling levels (4 = 16x) |
| `--cls_weight` | 3.0 | Classification loss weight (higher = more discriminative) |
| `--kl_weight` | 0.1 | KL divergence weight |
| `--contrastive_weight` | 0.05 | Ordinal contrastive loss weight |
| `--dropout_rate` | 0.2 | Optional modality dropout |
| `--rectified_flow_weight` | 0.01 | Rectified flow regularization (Stage 3) |

---

## CFM Loss

$$L_{CFM} = \| v_\theta(z_t, t) - (z_1 - z_0) \|^2$$

Where $z_t = (1-t) \cdot z_0 + t \cdot z_1$ and $z_0 \sim p_{NC}$, $z_1 \sim p_{AD}$.

**Forward-only constraint**: Only pairs where `src_class < tgt_class` are used (NC→SCD, NC→MCI, NC→AD, SCD→MCI, SCD→AD, MCI→AD). Distance-aware sampling gives higher weight to adjacent transitions.

**Rectified flow regularization** (optional):
$$L_{RF} = \lambda \cdot \mathbb{E}_t[\|v_\theta(z_t, t)\|^2 \cdot t(1-t)] + \lambda \cdot \|\nabla_z v_\theta\|^2$$
Encourages straight trajectories for efficient ODE integration.

---

## Hardware

| Config | GPU VRAM | Batch Size |
|--------|----------|------------|
| Minimum | 16GB | 1-2 |
| Recommended | 24GB (RTX 3090) | 2-4 |
| Optimal | 2x 24GB | 4-8 |

---

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) - Lipman et al.
- [MONAI: Medical Open Network for Imaging](https://monai.io/)

---

## License

MIT License
