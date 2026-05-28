# ADynamics

<!-- Badges -->
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-red.svg)](https://pytorch.org/)
[![MONAI 1.4+](https://img.shields.io/badge/monai-1.4+-green.svg)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

**ADynamics** models Alzheimer's Disease progression using **MMSE-Conditional Flow Matching** on cross-sectional multi-modal MRI data.

> **Key Insight**: We have cross-sectional data (different patients at different stages) with continuous MMSE cognitive scores. CFM learns individualized disease trajectories conditioned on target MMSE, enabling fine-grained progression prediction.

---

## Technical Pipeline

### Stage 1: Multi-Modal VAE (Encoder Training)

```
MRI Input:
  T1 (required) ──→ Encoder_T1 ─────────┐
  fMRI (optional) ─→ Encoder_fMRI ──────┤
  ASL  (optional) ─→ Encoder_ASL  ──────┼──→ Fusion ──→ μ, σ ──→ Reparameterize → z
  QSM  (optional) ─→ Encoder_QSM  ──────┤                    ↓
  FLAIR(optional) ─→ Encoder_FLAIR ─────┘              ┌────┴────┐
                                                   Decoder   Classifier
                                                      ↓         ↓
                                                  Recon MRI   3-Class
                                                              (NC/SCD+MCI/AD)

Loss = Recon + ordinal CE + KL + contrastive
```

### Stage 2: Encoder Validation

```
Stage 2a: Freeze Encoder → Train Classifier Head (validate latent discriminability)
Stage 2b: Freeze Encoder → Train Decoder Head   (validate latent reconstructability)
```

### Stage 3: MMSE-Conditional Flow Matching (Core Innovation)

```
Training Pairs (forward-only, distance-aware):
  z_source (MMSE=X) ──→ z_target (MMSE=Y),  where X > Y
  Distance-aware: adjacent MMSE pairs sampled more frequently

Velocity Field:
  v = VelocityFieldNet(z_t, t, mmse_target)
      ├── FiLM conditioning on time t
      └── FiLM conditioning on target MMSE

Loss = ||v_pred - (z_target - z_source)||² + λ_RF · rectified_flow_reg
```

### Stage 4: Deformation Generator

```
z_latent ──→ DeformationGenerator ──→ 3D Displacement Field
                                           ↓
Original MRI ──→ SpatialTransformer(field) ──→ Warped MRI
```

### Stage 5: Joint Fine-Tuning

```
All modules end-to-end:
  Encoder + CFM + DeformationGenerator
  Loss = recon + λ_cfm · cfm + λ_def · deformation + smooth + jacobian
```

### Inference Pipeline

```
Patient MRI ──→ Encoder ──→ z₀ (current latent)
                                │
                    CFM: v(z_t, t | mmse_target)
                                │
                                ↓
                    z₁ (evolved latent at target MMSE)
                       ├──→ Decoder ──→ Predicted MRI
                       └──→ DeformGen → Warp(original MRI)
```

---

## 5-Stage Training Summary

| Stage | Script | Goal | Key Loss |
|-------|--------|------|----------|
| **1** | `train_stage1_multimodal.py` | Train multi-modal encoder | recon + ordinal CE + KL + contrastive |
| **2a** | `train_stage2_classifier.py` | Validate encoder (classifier) | ordinal CE |
| **2b** | `train_stage2_decoder.py` | Validate encoder (decoder) | recon |
| **3** | `train_stage3_cfm.py` | MMSE-conditional flow | velocity + rectified flow |
| **4** | `train_stage4_deformation.py` | Deformation generator | similarity + smooth + jacobian |
| **5** | `train_stage5_joint.py` | Joint fine-tuning | all combined |

### Evaluation & Analysis

| Script | Purpose |
|--------|---------|
| `run_latent_analysis.py` | PCA/t-SNE/silhouette (latent quality) |
| `run_cls_validation.py` | Per-class accuracy, confusion matrix |
| `run_recon_validation.py` | MAE/PSNR/SSIM (reconstruction quality) |
| `run_flow_visualization.py` | Trajectory straightness, velocity analysis |
| `run_deform_validation.py` | Jacobian/folding analysis |
| `run_baseline_comparison.py` | CFM vs linear/KNN/regression baselines |
| `run_cross_validation.py` | 5-fold stratified CV |
| `run_ablation.py` | Systematic component ablation |

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
# Stage 1: Multi-modal VAE (3-class, with contrastive loss)
.\run_stage1.ps1

# After Stage 1: Check latent quality
.\run_analysis.ps1

# Stage 2a: Validate encoder with classifier
.\run_stage2a.ps1

# Stage 2b: Improve decoder
.\run_stage2b.ps1

# Stage 3: MMSE-conditional CFM (forward-only, distance-aware)
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
├── run_stage1.ps1              # Stage 1: Multi-modal VAE (3-class)
├── run_stage1_resume.ps1       # Resume from checkpoint
├── run_analysis.ps1            # Latent space analysis
├── run_stage2a.ps1 / run_stage2b.ps1  # Stage 2: Encoder validation
├── run_stage3.ps1              # Stage 3: MMSE-conditional CFM
├── run_stage4.ps1              # Stage 4: Deformation generator
├── run_stage5.ps1              # Stage 5: Joint fine-tuning
├── run_validation.ps1          # Full validation suite
├── run_baseline.ps1            # CFM vs baseline comparison
├── run_crossval.ps1            # 5-fold cross-validation
├── run_ablation.ps1            # Systematic ablation
│
├── core_data/                   # Data layer
│   ├── dataset.py              # MultiModalDataset, collate_fn
│   └── transforms.py           # MONAI preprocessing transforms
│
├── engine/                      # Training layer
│   ├── trainer_vae.py          # MultiModalVAETrainer (KL + contrastive)
│   ├── trainer_cfm.py          # CFMTrainer
│   └── losses.py               # All losses (incl. rectified flow)
│
├── models/                      # Model layer
│   ├── vae3d.py                # MultiModalVAE3D, ModalityEncoder3D
│   ├── vector_field.py         # VelocityFieldNet (FiLM + MMSE conditioning)
│   └── spatial_transform.py    # DeformationGenerator, SpatialTransformer
│
├── scripts/                     # All Python entry points
│   ├── train_stage1_multimodal.py    # 3-class VAE training
│   ├── train_stage2_classifier.py    # Classifier validation
│   ├── train_stage2_decoder.py       # Decoder validation
│   ├── train_stage3_cfm.py           # MMSE-conditional CFM
│   ├── train_stage4_deformation.py   # Deformation training
│   ├── train_stage5_joint.py         # Joint fine-tuning
│   ├── run_latent_analysis.py        # PCA/t-SNE/silhouette
│   ├── run_cls_validation.py         # Classification metrics
│   ├── run_recon_validation.py       # Reconstruction metrics
│   ├── run_flow_visualization.py     # Flow trajectory analysis
│   ├── run_deform_validation.py      # Deformation analysis
│   ├── run_baseline_comparison.py    # CFM vs baselines
│   ├── run_cross_validation.py       # 5-fold stratified CV
│   ├── run_ablation.py               # Component ablation
│   └── inference_pipeline.py         # End-to-end inference
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
    "label": 0,
    "mmse": 28
  }
]
```

Labels: `0=NC, 1=SCD, 2=MCI, 3=AD` (auto-remapped to 3-class: 0=NC, 1=SCD+MCI, 2=AD)

MMSE: 1-30 continuous cognitive score (used in Stage 3 for conditional flow)

T1 is required. Other modalities are optional (model handles missing modalities via dropout).

---

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--latent_channels` | 32 | Latent channels per modality encoder |
| `--base_channels` | 16 | Encoder base channels |
| `--decoder_depth` | 4 | Decoder upsampling levels (4 = 16x) |
| `--num_classes` | 3 | Disease classes (3: NC/SCD+MCI/AD, 4: NC/SCD/MCI/AD) |
| `--cls_weight` | 2.0 | Classification loss weight |
| `--kl_weight` | 0.1 | KL divergence weight |
| `--dropout_rate` | 0.2 | Optional modality dropout |
| `--rectified_flow_weight` | 0.01 | Rectified flow regularization (Stage 3) |

---

## CFM Loss

$$L_{CFM} = \| v_\theta(z_t, t, \text{mmse\_target}) - (z_1 - z_0) \|^2$$

Where $z_t = (1-t) \cdot z_0 + t \cdot z_1$ and $z_0 \sim p_{\text{MMSE}=X}$, $z_1 \sim p_{\text{MMSE}=Y}$, $X > Y$.

**Forward-only constraint**: Only pairs where `source_MMSE > target_MMSE` are used. Distance-aware sampling gives higher weight to adjacent MMSE ranges.

**Rectified flow regularization** (optional):
$$L_{RF} = \lambda \cdot \mathbb{E}_t[\|v_\theta(z_t, t, \text{mmse})\|^2 \cdot t(1-t)] + \lambda \cdot \|\nabla_z v_\theta\|^2$$
Encourages straight trajectories for efficient ODE integration.

**MMSE conditioning**: Target MMSE is injected via FiLM (Feature-wise Linear Modulation) at every U-Net block, enabling fine-grained control over progression degree.

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
