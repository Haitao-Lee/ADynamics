# ADynamics

<!-- Badges -->
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-red.svg)](https://pytorch.org/)
[![MONAI 1.4+](https://img.shields.io/badge/monai-1.4+-green.svg)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

**ADynamics** models Alzheimer's Disease progression using **MMSE-Conditional Flow Matching** on cross-sectional multi-modal MRI data, learning individualized disease trajectories from healthy control (NC) through subjective cognitive decline (SCD) and mild cognitive impairment (MCI) to Alzheimer's Disease (AD).

> **Key Insight**: Cross-sectional data has different patients at different stages. Each sample carries a continuous MMSE cognitive score. CFM learns a velocity field conditioned on target MMSE, enabling fine-grained, ordinal progression prediction that respects the disease ordering.

---

## Table of Contents

- [Highlights](#highlights)
- [Disease Stages & Modalities](#disease-stages--modalities)
- [Technical Pipeline](#technical-pipeline)
- [Stage-Based Training](#stage-based-training)
- [Inference Pipeline](#inference-pipeline)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [Configuration](#configuration)
- [CFM Loss](#cfm-loss)
- [Multi-GPU Notes](#multi-gpu-notes)
- [Hardware](#hardware)
- [References](#references)
- [License](#license)

---

## Highlights

- **Multi-modal 3D VAE** with T1 (required) + optional fMRI / ASL / QSM / FLAIR, robust to missing modalities via per-modality dropout
- **MMSE-conditional CFM** with FiLM conditioning on target MMSE, trained forward-only (NC→SCD→MCI→AD) with distance-aware pair sampling
- **Ordinal contrastive loss** enforces disease-stage ordering in the latent space
- **Free bits + KL annealing** prevents posterior collapse in the high-dimensional latent
- **3D deformation generator** with smoothness + Jacobian-determinant (no-folding) regularization
- **Joint fine-tuning** with differential learning rates (encoder 10× slower than CFM/deformation)
- **Comprehensive analysis suite**: latent PCA/t-SNE/silhouette, reconstruction MAE/PSNR/SSIM, classifier metrics, flow trajectory, deformation validation, baseline comparison (vs linear/KNN/regression), 5-fold stratified CV, ablation experiments
- **Multi-GPU first**: custom `MultiModalDataParallel` correctly scatters dict inputs to both GPUs (verified at ~24 GB / 23.4 GB on 2× RTX 3090)

---

## Disease Stages & Modalities

### 4-class (canonical)

| Label | Stage | Description |
|-------|-------|-------------|
| 0 | **NC** | Normal Control |
| 1 | **SCD** | Subjective Cognitive Decline |
| 2 | **MCI** | Mild Cognitive Impairment |
| 3 | **AD** | Alzheimer's Disease |

> Earlier 3-class experiments (SCD+MCI merged) are deprecated. The codebase ships in 4-class form (`num_classes: 4` in every config); `load_data` only remaps to 3-class when explicitly requested.

### Modalities

- **T1** (required) — 256×256×192 structural MRI
- **fMRI** (optional) — 34×64×64 functional MRI
- **ASL** (optional) — 128×128×36 arterial spin labeling (perfusion)
- **QSM** (optional) — 192×192×128 quantitative susceptibility mapping (iron deposition)
- **FLAIR** (optional) — 256×256×192 T2 FLAIR (white-matter lesions)

Optional modalities are randomly dropped during training (default `p=0.2`); at inference the model handles any subset.

---

## Technical Pipeline

### Stage 1 — Multi-Modal VAE

```
MRI Input:
  T1 (required)   ─→ Encoder_T1   ─┐
  fMRI (optional) ─→ Encoder_fMRI ─┤
  ASL  (optional) ─→ Encoder_ASL  ─┼─→ Concat ─→ fusion_proj ─→ μ, logvar
  QSM  (optional) ─→ Encoder_QSM  ─┤                    │
  FLAIR(optional) ─→ Encoder_FLAIR ─┘            Reparameterize
                                                       ↓ z
                                                  ┌────┴────┐
                                              Decoder   Classifier
                                                  ↓         ↓
                                            Recon T1    4-class logits
                                                       (NC/SCD/MCI/AD)

Loss = L1_recon + cls_weight · ordinal_CE
     + kl_weight · KL(with free bits & warmup)
     + contrastive_weight · ordinal_contrastive
```

The classifier head trains alongside the encoder so the latent space is *discriminative* by construction, not just reconstructive.

### Stage 2 — Encoder Validation

After Stage 1, the encoder is frozen and two heads are trained from scratch on the same latent:

- **Stage 2a** (Classifier head) — confirms the latent is linearly separable by disease stage
- **Stage 2b** (Decoder head) — confirms the latent is reconstructable when given a high-capacity decoder

If either fails, the Stage 1 latent is not useful for CFM; revisit Stage 1 hyper-parameters.

### Stage 3 — MMSE-Conditional Flow Matching

```
Training pairs (forward-only, distance-aware):
  z_source  (MMSE = X)  ──→  z_target  (MMSE = Y),  where X > Y
  Adjacent MMSE ranges sampled more frequently than distant ones.

Velocity field:
  v = VelocityFieldNet(z_t, t, mmse_target)
      ├── time FiLM conditioning on t ∈ [0, 1]
      └── FiLM conditioning on target MMSE

Loss = ‖ v_θ(z_t, t, m) − (z_target − z_source) ‖²
     + λ_RF · rectified_flow_regularization
```

`mmse_target` is injected via FiLM at every U-Net block, allowing the model to interpolate continuously between any two disease stages.

### Stage 4 — Deformation Generator

```
z_latent (evolved) ──→ DeformationGenerator ──→ 3D displacement field
                                                      ↓
Original MRI ──→ SpatialTransformer(field) ──→ Warped MRI

Loss = sim_weight · L1(warped, target)
     + smooth_weight · ‖∇field‖²
     + jacobian_weight · max(0, −det J)   (no folding)
```

Produces anatomically plausible 3D warps that morph a baseline MRI toward a target-stage MRI, complementing the latent-space trajectory of Stage 3.

### Stage 5 — Joint Fine-Tuning

All modules end-to-end with **differential learning rates**:

```
L = recon_weight · recon
  + cfm_weight   · cfm_velocity
  + def_weight   · deformation_similarity
  + smooth_weight + jacobian_weight   (regularization)
```

```
lr_multipliers:
  encoder: 0.1     # 10× slower than CFM / deformation
  cfm:     1.0
  deform:  1.0
```

This preserves the discriminative latent learned in Stage 1 while letting CFM and deformation converge to a coherent end-to-end pipeline.

---

## Stage-Based Training

| Stage | Script | Goal | Key Loss |
|-------|--------|------|----------|
| **1** | `scripts/train_stage1_multimodal.py` | Multi-modal encoder + decoder + classifier | recon + ordinal CE + KL + contrastive |
| **2a** | `scripts/train_stage2_classifier.py` | Validate encoder (classifier) | ordinal CE |
| **2b** | `scripts/train_stage2_decoder.py` | Validate encoder (decoder) | recon |
| **3** | `scripts/train_stage3_cfm.py` | MMSE-conditional flow | velocity + rectified flow |
| **4** | `scripts/train_stage4_deformation.py` | Deformation generator | sim + smooth + jacobian |
| **5** | `scripts/train_stage5_joint.py` | Joint fine-tuning | all combined |

All hyperparameters are centralized in `configs/*.yaml`; CLI flags override YAML.

### Canonical run scripts (PowerShell, on Windows)

```powershell
# Stage 1: Multi-modal VAE (4-class, kl=1.0, contrastive=0.3, ~13h on 2x RTX 3090)
.\run_01_train.ps1

# Stage 2a: validate encoder with classifier
.\run_02a_classifier.ps1

# Stage 2b: validate encoder with decoder
.\run_02b_decoder.ps1

# Stage 3: MMSE-conditional CFM
.\run_03_cfm.ps1

# Stage 4: deformation generator
.\run_04_deformation.ps1

# Stage 5: joint fine-tuning
.\run_05_joint.ps1
```

### Analysis & validation

| Script | Purpose |
|--------|---------|
| `run_latent_analysis.py` | PCA / t-SNE / silhouette / variance on latent space |
| `run_recon_validation.py` | MAE / PSNR / SSIM on reconstructions |
| `run_cls_validation.py` | Per-class accuracy + confusion matrix |
| `run_flow_visualization.py` | Trajectory straightness + velocity analysis |
| `run_deform_validation.py` | Jacobian / folding analysis |
| `run_baseline_comparison.py` | CFM vs linear / KNN / regression baselines |
| `run_cross_validation.py` | 5-fold stratified CV |
| `run_ablation.py` | Systematic component ablation |

```powershell
.\run_analysis_latent.ps1     # latent space analysis
.\run_analysis_all.ps1        # full validation suite
.\run_baseline.ps1            # CFM vs simpler baselines
.\run_crossval.ps1            # 5-fold CV
.\run_ablation.ps1            # component ablation
```

---

## Inference Pipeline

```python
import torch
from models.vae3d import MultiModalVAE3D
from models.vector_field import VelocityFieldNet
from models.spatial_transform import DeformationGenerator, SpatialTransformer
from core_data.transforms import get_multimodal_val_transforms

# 1) Load models from the canonical checkpoints
device = torch.device("cuda")
vae = MultiModalVAE3D(num_classes=4).to(device).eval()
vae.load_state_dict(torch.load("./checkpoints/stage1_multimodal/vae_best.pt")["model_state_dict"])

cfm = VelocityFieldNet(...).to(device).eval()
cfm.load_state_dict(torch.load("./checkpoints/stage3_cfm/cfm_best.pt")["model_state_dict"])

deform = DeformationGenerator(...).to(device).eval()
deform.load_state_dict(torch.load("./checkpoints/stage4_def/def_best.pt")["model_state_dict"])

# 2) Encode baseline MRI
with torch.no_grad():
    z0 = vae.get_latent({"t1": mri_tensor})               # current latent

# 3) Evolve latent via ODE (Euler, 20 steps) to target MMSE
z_final, trajectory = integrate_ode(z0, cfm, mmse_target=18.0, steps=20)

# 4) Two complementary outputs
predicted_mri = vae.decode(z_final)                       # direct reconstruction
flow = deform(z_final)                                    # 3D displacement field
warped_mri = SpatialTransformer(mri_tensor, flow)         # anatomically warped MRI
```

`inference_results/latent_analysis/` ships pre-computed PCA/t-SNE/silhouette from the reference run for inspection without retraining.

---

## Quick Start

### Installation

```powershell
# Windows PowerShell (recommended)
.\install_env.ps1

# Or manual
conda create -n ADynamics python=3.11 -y
conda activate ADynamics
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

### Train (canonical)

```powershell
# Stage 1: multi-modal VAE (canonical 4-class, dual-GPU)
.\run_01_train.ps1

# Or call directly with overrides
python scripts/train_stage1_multimodal.py `
    --config ./configs/stage1_vae.yaml `
    --epochs 300 `
    --batch_size 2 `
    --num_gpus 2
```

The Stage 1 run takes ~13 hours on 2× RTX 3090; subsequent stages take 1-3 hours each.

### Analyze

```powershell
# Latent space analysis on all samples (num_samples=99999 in configs/analysis.yaml)
.\run_analysis_latent.ps1

# Full validation suite
.\run_analysis_all.ps1
```

Results are written to `inference_results/<task>/`.

---

## Project Structure

```
ADynamics/
├── README.md
├── LICENSE                              # MIT
├── requirements.txt                     # verified for CUDA 12.1 / PyTorch 2.5.1
├── environment.yml                      # conda environment
├── install_env.ps1                      # one-shot installer (incl. FSL/HD-BET notes)
│
├── run_01_train.ps1                     # Stage 1
├── run_02a_classifier.ps1               # Stage 2a
├── run_02b_decoder.ps1                  # Stage 2b
├── run_03_cfm.ps1                       # Stage 3
├── run_04_deformation.ps1               # Stage 4
├── run_05_joint.ps1                     # Stage 5
├── run_analysis_latent.ps1              # latent analysis
├── run_analysis_all.ps1                 # full validation suite
├── run_baseline.ps1                     # CFM vs baselines
├── run_crossval.ps1                     # 5-fold CV
├── run_ablation.ps1                     # component ablation
│
├── tests/                               # self-contained test suite
│   ├── README.md                        # how to run
│   ├── test_encoder_upgrade.py          # 6 integration tests for the multi-axis 3D attention upgrade
│   └── test_cli_smoke.py                # CLI + YAML + model build smoke test
│
├── configs/                             # all hyper-parameters (YAML)
│   ├── stage1_vae.yaml                  # 4-class, kl=1.0, contrastive=0.3, use_attention=true
│   ├── stage2a_classifier.yaml
│   ├── stage2b_decoder.yaml
│   ├── stage3_cfm.yaml                  # forward-only, distance-aware
│   ├── stage4_deform.yaml
│   ├── stage5_joint.yaml
│   └── analysis.yaml                    # shared analysis defaults
│
├── core_data/                           # data layer
│   ├── dataset.py                       # MultiModalDataset, multimodal_collate_fn
│   ├── transforms.py                    # MONAI preprocessing
│   └── dataset_manifest_merged_v2.json  # canonical 4-class manifest
│
├── engine/                              # training layer
│   ├── trainer_vae.py                   # VAETrainer + MultiModalVAETrainer (KL + contrastive + free bits)
│   ├── trainer_cfm.py                   # CFMTrainer
│   └── losses.py                        # all losses (incl. rectified flow, ordinal CE, SSIM)
│
├── models/                              # model layer
│   ├── vae3d.py                         # MultiModalVAE3D, ModalityEncoder3D, ADynamicsVAE3D
│   ├── vector_field.py                  # VelocityFieldNet (FiLM + MMSE)
│   ├── spatial_transform.py             # DeformationGenerator, SpatialTransformer, flow utils
│   └── attention_3d.py                  # AxialAttention3D + MultiAxisAttention3D (NeuroQuant CVPR 2026)
│
├── scripts/                             # all Python entry points
│   ├── train_stage1_multimodal.py       # Stage 1 training (multi-modal, 4-class, w/ attention)
│   ├── train_stage2_classifier.py
│   ├── train_stage2_decoder.py
│   ├── train_stage3_cfm.py
│   ├── train_stage4_deformation.py
│   ├── train_stage5_joint.py
│   ├── run_latent_analysis.py
│   ├── run_recon_validation.py
│   ├── run_cls_validation.py
│   ├── run_flow_visualization.py
│   ├── run_deform_validation.py
│   ├── run_baseline_comparison.py
│   ├── run_cross_validation.py
│   ├── run_ablation.py
│   └── inference_pipeline.py            # end-to-end inference
│
├── utils/                               # utilities
│   ├── config_loader.py                 # YAML → argparse defaults bridge
│   ├── multi_gpu.py                     # MultiModalDataParallel (dict inputs)
│   ├── io_utils.py                      # NIfTI I/O
│   └── preprocessing/                   # FSL-FAST wrapper, N4 bias correction, registration
│       ├── denoise.py                   # ANTsPy
│       ├── n4_bias_correction.py        # ANTsPy
│       ├── normalization.py             # percentile clipping
│       ├── registration.py              # ANTs / SimpleITK
│       ├── hdbet_brainmask.py           # HD-BET CLI wrapper
│       ├── seg_tissue.py                # FSL-FAST wrapper
│       ├── seg_tissue_LHT.py            # PVE -> GM/WM/CSF volume metrics
│       ├── seg_region.py                # cortical / subcortical parcellation
│       └── get_metadata.py
│
├── docs/                                # additional documentation
│   ├── TRAINING_PIPELINE.md             # detailed Chinese training guide
│   └── CODE_REVIEW_GUIDE.md
│
├── checkpoints/                         # model weights (gitignored)
│   ├── stage1_multimodal/               # Stage 1 outputs
│   ├── stage2_classifier/               # Stage 2a outputs
│   ├── stage2_decoder/                  # Stage 2b outputs
│   ├── stage3_cfm/                      # Stage 3 outputs
│   ├── stage4_def/                      # Stage 4 outputs
│   └── stage5_joint/                    # Stage 5 outputs
│
├── logs/                                # training logs
└── inference_results/                   # analysis outputs (PCAs, reconstructions, etc.)
    ├── latent_analysis/
    ├── recon_validation/
    ├── cls_validation/
    ├── flow_visualization/
    ├── deform_validation/
    ├── baseline_comparison/
    ├── cross_validation/
    └── ablation/
```

---

## Data Format

`core_data/dataset_manifest_merged_v2.json` is a JSON array of sample records:

```json
[
  {
    "t1":   "/abs/path/to/t1.nii.gz",
    "fmri": "/abs/path/to/fmri.nii.gz",
    "asl":  "/abs/path/to/asl.nii.gz",
    "qsm":  "/abs/path/to/qsm.nii.gz",
    "flair":"/abs/path/to/flair.nii.gz",
    "label": 0,
    "mmse":  28,
    "patient_id": "sub-001"
  }
]
```

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `t1` | path | yes | T1-weighted MRI (256×256×192) |
| `fmri`, `asl`, `qsm`, `flair` | path | no | any subset acceptable |
| `label` | int | yes | 0=NC, 1=SCD, 2=MCI, 3=AD (4-class canonical) |
| `mmse` | float | yes | 0–30 cognitive score, used by Stage 3 CFM |
| `patient_id` | str | recommended | for tracking across splits |

The loader skips corrupted NIfTI files (zero-dim shape, all-zero data, unreadable) and reports a per-class sample count.

---

## Configuration

All training defaults are YAML. CLI flags override YAML. The order of precedence is:

1. `add_argument(...)` defaults in the script (lowest)
2. `parser.set_defaults(**config_defaults)` from `--config <yaml>` (overrides 1)
3. CLI flags passed on the command line (overrides 2)

> **Implementation note**: `parser.set_defaults(...)` is called *after* all `add_argument` calls, so YAML values correctly win over script-level defaults.

### Stage 1 — `configs/stage1_vae.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.num_classes` | 4 | NC / SCD / MCI / AD (canonical) |
| `model.latent_channels` | 32 | per-modality latent channels |
| `model.base_channels` | 16 | encoder base channels |
| `model.decoder_depth` | 4 | upsampling blocks (4 = 16×) |
| `model.dropout_rate` | 0.2 | optional-modality dropout |
| `training.batch_size` | 2 | **total** (1 per GPU) |
| `training.num_gpus` | 2 | dual-GPU DataParallel |
| `training.use_amp` | false | off (AMP caused NaN in VAE) |
| `training.epochs` | 300 | |
| `loss.cls_weight` | 1.0 | classification weight |
| `loss.kl_weight` | 1.0 | KL divergence weight |
| `loss.kl_warmup_epochs` | 30 | linear warmup from 0 |
| `loss.free_bits` | 0.01 | min KL per latent dim |
| `loss.contrastive_weight` | 0.3 | ordinal contrastive loss |
| `loss.contrastive_temperature` | 0.1 | NT-Xent temperature |

### Stage 3 — `configs/stage3_cfm.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cfm.forward_only` | true | only `source > target` pairs |
| `cfm.distance_aware` | true | higher weight on adjacent MMSE pairs |
| `model.num_conditions` | 1 | 1-D MMSE conditioning |
| `loss.velocity_loss_weight` | 1.0 | |
| `loss.rectified_flow_weight` | 0.01 | 0 = disabled |

### Stage 4 — `configs/stage4_deform.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss.sim_weight` | 1.0 | warped-vs-target L1 |
| `loss.smooth_weight` | 0.1 | ‖∇field‖² |
| `loss.jacobian_weight` | 0.01 | anti-folding penalty |

### Stage 5 — `configs/stage5_joint.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_multipliers.encoder` | 0.1 | 10× slower than CFM / deform |
| `lr_multipliers.cfm` | 1.0 | |
| `lr_multipliers.deform` | 1.0 | |

### Analysis — `configs/analysis.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input.num_classes` | 4 | must match the trained checkpoint |
| `latent_analysis.num_samples` | 99999 | 99999 = use all available |
| `recon_validation.num_samples` | 20 | |
| `flow_visualization.ode_steps` | 20 | Euler steps |
| `cross_validation.n_folds` | 5 | stratified |

---

## CFM Loss

The conditional flow-matching loss is

$$
\mathcal{L}_{\mathrm{CFM}} = \bigl\| v_\theta(z_t, t, m) - (z_1 - z_0) \bigr\|^2
$$

where

- $z_0 \sim p_{\mathrm{MMSE}=X}$, $z_1 \sim p_{\mathrm{MMSE}=Y}$ with $X > Y$ (forward-only),
- $m$ is the target MMSE score (FiLM conditioning),
- $z_t = (1-t)\,z_0 + t\,z_1$ for $t \in [0, 1]$.

**Distance-aware sampling** — adjacent MMSE bins (e.g. NC→SCD) are sampled more often than distant pairs (NC→AD), matching the natural progression density.

**Rectified flow regularization** (optional)

$$
\mathcal{L}_{\mathrm{RF}} = \lambda \cdot \mathbb{E}_t\!\left[\|v_\theta(z_t, t, m)\|^2 \cdot t(1-t)\right] + \lambda \cdot \|\nabla_z v_\theta\|^2
$$

encourages straight trajectories for efficient ODE integration.

**MMSE conditioning via FiLM** lets the model produce continuous trajectories rather than discrete stage-to-stage jumps.

---

## Multi-GPU Notes

`utils/multi_gpu.py` provides `MultiModalDataParallel`, a `DataParallel` subclass that handles the **dict-input** shape of the multi-modal VAE (`x_dict = {"t1": tensor, "fmri": tensor, ...}`). Standard `nn.DataParallel` cannot scatter dicts.

Key implementation details:

- **Scatter**: replicates a small batch across all GPUs (then chunks evenly) so a batch of 2 is usable on 2 GPUs without replication bugs. Combined with `drop_last=True` in the DataLoader, every GPU sees a different shard.
- **Gather**: moves each shard to `output_device` *before* `torch.cat`, avoiding device-mismatch errors when each shard is on a different GPU.
- **Return shape**: matches `parallel_apply` — `(input_shards, kwargs_shards)` tuple, not a flat list of pairs.

`use_amp: false` is the canonical setting for Stage 1. AMP (`fp16`) caused NaN in the combined loss because the KL and reconstruction losses differ by orders of magnitude and `fp16` cannot represent both accurately.

---

## Hardware

| Config | GPU VRAM | Batch Size (per stage) | Notes |
|--------|----------|------------------------|-------|
| Minimum | 16 GB | 1 | Stage 1 with `base_channels=16` |
| Recommended | 24 GB (RTX 3090) | 2 | single-GPU |
| **Optimal (verified)** | **2× 24 GB** | **2 total (1/GPU)** | Stage 1 ≈ 24 GB / 23.4 GB, 2.5 s/iter |
| Multi-GPU | 4× 24 GB | 4 total (1/GPU) | also supported |

OOM tips: reduce `base_channels` (16→12→8), reduce `decoder_depth` (4→3), or enable gradient checkpointing.

---

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al.
- [Rectified Flow](https://arxiv.org/abs/2209.03003) — Liu et al.
- [FiLM: Feature-wise Linear Modulation](https://arxiv.org/abs/1709.07871) — Perez et al.
- [MONAI: Medical Open Network for Imaging](https://monai.io/)
- [torchdiffeq: Differentiable ODE Solvers](https://github.com/rtqichen/torchdiffeq)

---

## License

[MIT](LICENSE) — Copyright (c) 2026 ADynamics Development Team
