# ADynamics Tests

Self-contained tests for the ADynamics project. All tests are designed to
run on a single GPU (CUDA), with `cudnn.deterministic = True` where applicable
to avoid cuDNN benchmark noise breaking bitwise-equality assertions.

## Quickstart

```bash
# Make sure the project root is on PYTHONPATH (one level up from tests/)
cd /path/to/ADynamics
export PYTHONPATH=$(pwd):$PYTHONPATH   # or: set PYTHONPATH=%cd%;%PYTHONPATH% on Windows

# Run individual test files
python tests/test_encoder_upgrade.py    # 6 integration tests for the multi-axis 3D attention upgrade
python tests/test_cli_smoke.py          # CLI argument parsing + YAML loading + model build smoke test

# Run the unit self-test for the attention module
python models/attention_3d.py          # 11 unit tests for AxialAttention3D + MultiAxisAttention3D
```

## What's tested

### `tests/test_encoder_upgrade.py`
1. No-attention baseline forward + correct output shapes
2. **Zero-init identity** — with non-attention weights copied from baseline,
   the attention model output is **bitwise identical** to the no-attention
   model (max diff 0.00e+00) at initialization
3. **Warmup effect** — perturbing the attention output projection makes
   the output differ from initialization (attention is actually learning)
4. **Full forward + backward** — KL + reconstruction + cross-entropy all
   numerically stable, no NaN/Inf gradients over 3 training steps
5. **Multi-stage attention** — attention at stages (2, 3) also preserves
   bitwise identity at init
6. **Multi-GPU DataParallel** — `MultiModalDataParallel.scatter` works
   with the dict-shaped multi-modal input

### `tests/test_cli_smoke.py`
- Argparse handles `--config <yaml>`, `--use_attention`, `--no_attention`,
  `--attention_levels "3"` / `"2,3"`, `--attention_heads 8` correctly
- YAML values from `configs/stage1_vae.yaml` are loaded into the Namespace
- `MultiModalVAE3D` builds with the YAML-derived params
- `--no_attention` produces a 32.55M-param model; default produces 36.50M
- Forward pass works end-to-end on synthetic data

### `models/attention_3d.py` (in-module `_selftest`)
- `AxialAttention3D`: zero-init verified for all 3 axes × multiple shapes
- `MultiAxisAttention3D`: residual identity at init; gradient flow on
  both zero-init and un-zeroed weights
- Sub-axis enablement (`use_d_h_w=(True, False, False)`) verified
- `num_heads` auto-reduction when channel count isn't divisible
- Training signal verification (output changes after warmup)

## Adding a new test

Tests should:
1. Be self-contained (no external data dependencies — use synthetic tensors)
2. Set `torch.backends.cudnn.deterministic = True` at the top if they assert
   bitwise equality
3. Print clear PASS / FAIL messages so log-grep is easy
4. Be runnable as `python tests/your_test.py` with no CLI args

## Hardware

Tested on:
- 2× NVIDIA RTX 3090 (24 GB each)
- CUDA 12.1, PyTorch 2.5.1, Python 3.11
- `cudnn.deterministic = True`

The encoder integration test (`test_encoder_upgrade.py`) needs ~2 GB VRAM
per test case. The full suite runs in well under 5 minutes on a single GPU.
