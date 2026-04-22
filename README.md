# ADynamics

<!-- Badges Row -->
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.5.1-red.svg)](https://pytorch.org/)
[![MONAI Version](https://img.shields.io/badge/monai-1.4.0-green.svg)](https://monai.io/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/cuda-12.1-blue.svg)](https://developer.nvidia.com/cuda-downloads)
[![Paper](https://img.shields.io/badge/paper-arXiv:2210.02747-orange.svg)](https://arxiv.org/abs/2210.02747)

<!-- Logo / Teaser -->
<p align="center">
  <img src="assets/teaser.png" width="70%" alt="ADynamics Disease Progression">
</p>

<!-- Short Description -->
**ADynamics** models the dynamic progression of Alzheimer's Disease (AD) from Normal Control (NC) through Subjective Cognitive Decline (SCD) and Mild Cognitive Impairment (MCI) to AD dementia, using **Conditional Flow Matching (CFM)** to learn population-level disease trajectories from cross-sectional T1-weighted MRI data.

> **Key Insight**: We have cross-sectional MRI data (different patients at different disease stages), NOT longitudinal data. Flow Matching enables learning an optimal transport map between disease stage distributions without paired data.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Hardware Requirements](#hardware-requirements)
- [Citation](#citation)
- [License](#license)

---

## Features

- **5-Stage Disentangled Training Pipeline**: VAE → Classifier → CFM → Deformation → Joint Fine-tuning
- **3D Medical Imaging**: Native support for T1-weighted MRI with MONAI transforms
- **Conditional Flow Matching**: Learn disease progression vector fields in latent space
- **Anatomically Plausible Deformations**: Jacobian penalty prevents folding/self-intersection
- **FiLM Conditioning**: Time and clinical covariate conditioning for personalized trajectories
- **End-to-End Inference**: One-command evolution from NC brain to AD-like brain

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ADynamics 5-Stage Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: 3D VAE                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│  Input:  T1 MRI [B, 1, 128, 128, 128]                               │
│  Output: Reconstructed MRI, latent z [B, 64, 8, 8, 8]               │
│  Loss:   L1 Reconstruction + KL (VAE regularizer)                   │
│  Goal:   Learn compressed latent representation                      │
│                                                                      │
│  Stage 2: Disease Classifier                                         │
│  ─────────────────────────────────────────────────────────────────  │
│  Input:  VAE latent z                                               │
│  Output: Disease stage logits [B, 4] (NC/SCD/MCI/AD)               │
│  Loss:   Cross-Entropy                                               │
│  Goal:   Ensure latent encodes disease information                   │
│                                                                      │
│  Stage 3: Conditional Flow Matching                                  │
│  ─────────────────────────────────────────────────────────────────  │
│  Input:  z_t (interpolated latent), time t, condition c (age)       │
│  Output: Velocity field v(z_t, t, c)                                 │
│  Loss:   ||v_pred - (z_AD - z_NC)||²                                 │
│  Goal:   Learn disease progression vector field                       │
│                                                                      │
│  Stage 4: Deformation Generator + Spatial Transformer               │
│  ─────────────────────────────────────────────────────────────────  │
│  Input:  Evolved latent z_final                                     │
│  Output: 3D displacement field [B, 3, 128, 128, 128]                 │
│  Loss:   L1(image similarity) + 0.1*L_smooth + 0.01*L_jacobian       │
│  Goal:   Generate anatomically plausible warps                       │
│                                                                      │
│  Stage 5: Joint Fine-tuning                                          │
│  ─────────────────────────────────────────────────────────────────  │
│  Goal:   Fine-tune all modules together                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### CFM Loss Formula

$$L_{CFM} = \| v_\theta(z_t, t) - (z_1 - z_0) \|^2$$

Where $z_t = (1-t) \cdot z_0 + t \cdot z_1$ (linear interpolation)

---

## Installation

### Prerequisites

- **Python**: 3.11+
- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 3090 / A100 recommended)
- **CUDA**: 12.1 or higher

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/ADynamics.git
cd ADynamics
```

### Step 2: Environment Setup

**Option A: Use the installation script (Recommended)**

```powershell
# PowerShell on Windows
.\install_env.ps1

# Or Linux/Mac
bash install_env.sh
```

**Option B: Manual installation**

```bash
# Create conda environment
conda create -n ADynamics python=3.11 -y
conda activate ADynamics

# Install PyTorch with CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install medical imaging packages
pip install monai==1.4.0 "nibabel>=5.0.0,<6.0.0" SimpleITK>=2.2.0

# Install math/ODE packages
pip install torchdiffeq>=0.2.5

# Install data utilities
pip install "numpy<2.0" scipy pandas scikit-learn pyyaml matplotlib tqdm einops
```

**Option C: Reference [ENV_MATRIX.md](../docs/ENV_MATRIX.md) for exact versions**

### Step 3: Verify Installation

```bash
conda activate ADynamics
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python scripts/test_pipeline.py
```

---

## Quick Start

### 1. Test the Pipeline

```bash
cd ADynamics
python scripts/test_pipeline.py
```

Expected output:
```
# ADynamics Stage 1 Pipeline Integration Test
# ============================================================
✅ Transforms test PASSED
✅ VAE forward pass test PASSED
✅ VAE training step test PASSED
✅ NIfTI I/O test PASSED
✅ DataLoader pipeline test PASSED
✅ ADynamics Stage 1 Pipeline is fully integrated and tested!
```

### 2. Train with Dummy Data

```bash
# Stage 1: VAE
python scripts/train_stage1_vae.py --use_dummy_data --epochs 5

# Stage 2 & 3: Classifier + CFM
python scripts/train_stage2_cfm.py --stage 2 --use_dummy_data
python scripts/train_stage2_cfm.py --stage 3 --use_dummy_data
```

### 3. Train with Real Data

```bash
# Stage 1: VAE training
python scripts/train_stage1_vae.py \
    --data_dir /path/to/mri/data \
    --config configs/vae_train.yaml \
    --epochs 300

# Stage 2: Classifier training
python scripts/train_stage2_cfm.py \
    --stage 2 \
    --vae_checkpoint checkpoints/stage1_vae/vae_best.pt \
    --data_dir /path/to/mri/data

# Stage 3: CFM training
python scripts/train_stage2_cfm.py \
    --stage 3 \
    --vae_checkpoint checkpoints/stage1_vae/vae_best.pt \
    --data_dir /path/to/mri/data
```

### 4. Run End-to-End Inference

```bash
python scripts/inference_pipeline.py \
    --input path/to/NC_T1.nii.gz \
    --age 70 \
    --vae_checkpoint checkpoints/stage1_vae/vae_best.pt \
    --cfm_checkpoint checkpoints/stage3_cfm/cfm_best.pt \
    --deform_checkpoint checkpoints/stage4_deform/deform_best.pt \
    --output_dir ./inference_results
```

This generates:
- `{patient_id}_original.nii.gz` - Original input MRI
- `{patient_id}_evolved.nii.gz` - Evolved AD-like MRI
- `{patient_id}_flow_*.nii.gz` - Displacement field components
- `{patient_id}_trajectory.npz` - Latent trajectory for analysis

---

## Project Structure

```
ADynamics/
├── configs/                     # YAML configuration files
│   ├── vae_train.yaml          # Stage 1: VAE training
│   ├── cfm_train.yaml          # Stage 2&3: Classifier + CFM
│   └── deform_train.yaml       # Stage 4: Deformation
│
├── core_data/                   # Data pipeline
│   ├── __init__.py
│   ├── dataset.py              # Dataset and DataLoader utilities
│   └── transforms.py           # MONAI preprocessing transforms
│
├── models/                      # Neural network architectures
│   ├── __init__.py
│   ├── vae3d.py               # 3D VAE with residual blocks
│   ├── classifier.py          # Disease stage MLP classifier
│   ├── vector_field.py        # FiLM-conditioned velocity field U-Net
│   └── spatial_transform.py   # DeformationGenerator + SpatialTransformer
│
├── engine/                      # Training engine
│   ├── __init__.py
│   ├── losses.py              # VAE, CFM, Jacobian, Smoothness losses
│   ├── trainer_vae.py         # VAETrainer class
│   └── trainer_cfm.py        # CFMTrainer class
│
├── utils/                       # Utilities
│   ├── __init__.py
│   └── io_utils.py           # NIfTI I/O with RAS orientation
│
├── scripts/                     # Execution scripts
│   ├── test_pipeline.py       # Integration test (verify all modules)
│   ├── train_stage1_vae.py    # Stage 1: VAE training entry
│   ├── train_stage2_cfm.py    # Stage 2&3: Classifier + CFM training
│   └── inference_pipeline.py  # End-to-end evolution inference
│
├── docs/                        # Documentation
│   └── ENV_MATRIX.md          # Environment setup matrix
│
├── requirements.txt            # Pip dependencies (fallback)
├── install_env.ps1            # PowerShell installation script
└── README.md                  # This file
```

---

## Data Preparation

### Expected Directory Structure

```
data_dir/
├── NC/                          # Normal Control
│   ├── sub-NC001_T1.nii.gz
│   ├── sub-NC002_T1.nii.gz
│   └── ...
├── SCD/                         # Subjective Cognitive Decline
│   ├── sub-SCD001_T1.nii.gz
│   └── ...
├── MCI/                         # Mild Cognitive Impairment
│   ├── sub-MCI001_T1.nii.gz
│   └── ...
└── AD/                          # Alzheimer's Disease
    ├── sub-AD001_T1.nii.gz
    └── ...
```

### Minimum Sample Size

| Group | Minimum | Recommended |
|-------|---------|-------------|
| NC | 30 | 50+ |
| SCD | 30 | 50+ |
| MCI | 30 | 50+ |
| AD | 30 | 50+ |

### Preprocessing Pipeline

ADynamics expects preprocessed MRI data. The recommended preprocessing:

1. **Denoise** - ANTs or MRtrix3 dwidenoise
2. **Bias Correction** - N4BiasFieldCorrection
3. **Registration** - ANTs SyN to MNI152
4. **Brain Extraction** - HD-BET or BET
5. **Intensity Normalization** - Z-score or WhiteStripe

> **Note**: Raw NIfTI files should be resampled to 1mm³ isotropic resolution.

---

## Training

### Stage 1: VAE

Train the 3D VAE to compress MRI to latent space:

```bash
python scripts/train_stage1_vae.py \
    --data_dir /path/to/data \
    --config configs/vae_train.yaml \
    --epochs 300 \
    --batch_size 2
```

**Checkpoints**: `checkpoints/stage1_vae/vae_best.pt`

### Stage 2: Classifier

Train the disease stage classifier on VAE latents:

```bash
python scripts/train_stage2_cfm.py \
    --stage 2 \
    --vae_checkpoint checkpoints/stage1_vae/vae_best.pt \
    --data_dir /path/to/data
```

**Checkpoints**: `classifier_best.pt`

### Stage 3: CFM

Train the velocity field network:

```bash
python scripts/train_stage2_cfm.py \
    --stage 3 \
    --vae_checkpoint checkpoints/stage1_vae/vae_best.pt \
    --data_dir /path/to/data \
    --epochs 500 \
    --batch_size 32
```

**Checkpoints**: `checkpoints/stage3_cfm/cfm_best.pt`

### Stage 4: Deformation (Coming Soon)

```bash
# python scripts/train_stage4_deform.py ...
```

---

## Inference

### Evolve NC to AD

```bash
python scripts/inference_pipeline.py \
    --input /path/to/NC_T1.nii.gz \
    --age 70 \
    --output_dir ./results \
    --vae_checkpoint checkpoints/stage1_vae/vae_best.pt \
    --cfm_checkpoint checkpoints/stage3_cfm/cfm_best.pt \
    --deform_checkpoint checkpoints/stage4_deform/deform_best.pt
```

### Inference Pipeline

```python
from scripts.inference_pipeline import EvolvePipeline

pipeline = EvolvePipeline(
    vae=vae_model,
    vector_field=cfm_model,
    deform_generator=deform_model,
    device="cuda",
    spatial_size=(128, 128, 128),
)

results = pipeline.evolve(
    mri,                    # Input NC MRI
    c=age_normalized,       # Clinical condition (e.g., normalized age)
    ode_steps=20,           # Integration steps
)

# Results contain:
# - evolved_mri: AD-like brain
# - deformation_field: 3D displacement
# - trajectory: Latent evolution path
```

---

## Hardware Requirements

| Configuration | GPU VRAM | RAM | Storage | Training Time (approximate) |
|---------------|----------|-----|---------|------------------------------|
| Minimum | 16GB | 32GB | 100GB | ~24 hours per stage |
| Recommended | 24GB (RTX 3090/A4000) | 64GB | 200GB | ~6-8 hours per stage |
| Optimal | 2x 24GB | 128GB | 500GB | ~3-4 hours per stage |

**Batch Size Guidelines:**

| VRAM | Batch Size (3D MRI) |
|------|---------------------|
| 16GB | 1-2 |
| 24GB | 2-4 |
| 40GB (A100) | 4-8 |
| 80GB (A100) | 8-16 |

---

## Citation

If you find this work useful in your research, please cite:

```bib
@article{adynamics2026,
  title={AD-FlowMorph: Conditional Flow Matching for MRI-based Alzheimer's Disease Progression Modeling},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

---

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) - Lipman et al.
- [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) - Liu et al.
- [MONAI: Medical Open Network for Imaging](https://monai.io/)
- [Conditional Flow Matching for Generative Modeling](https://github.com/facebookresearch/flow_matching) - Facebook Research

---

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

## Acknowledgments

- MONAI team for medical imaging deep learning infrastructure
- Facebook Research Flow Matching implementation
- NVIDIA GPU grants for research computing

---

<p align="center">
  <strong>ADynamics</strong> — Modeling Disease Progression with Conditional Flow Matching
</p>
