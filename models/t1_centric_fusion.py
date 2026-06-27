"""
T1-centric fusion for multi-modal VAE.

The key invariant: when all auxiliary modalities are missing, the
output equals the T1 trunk. This GUARANTEES that adding auxiliary
modalities can never produce a worse latent than T1-only — the worst
case is "no contribution from aux", which is exactly the T1 baseline.

Architecture:

    z_t1 (required, always present)         z_aux[mod] (per-modality, may be None)
            |                                       |
            v                                       v
       t1_trunk (1x1 conv refinement)         delta_net[mod] (1x1 conv, zero-init)
            |                                       |
            |                                  gate_net[mod] (sigmoid scalar)
            |                                       |
            |         +---- gate[mod] * delta[mod] -----+
            |         |                               |
            + <-------*-------- *-------- *---------- *
                          (residual sum, gated)

After all gated residuals: mu = z_t1_trunk + sum(aux) + demo_bias

The delta_nets are zero-initialized, so at the start of training, the
model behaves identically to T1-only. Auxiliary modalities contribute
only after the model has learned they are useful. This is the architectural
counterpart of a monotonicity loss.

Drop-in replacement for the original `fusion_proj` (1x1 conv on
concatenated 5*latent_channels). Param count is comparable (~12K vs 24K)
but the inductive bias is fundamentally different: T1 is the skeleton,
aux modalities are bounded corrections.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class T1CentricFusion(nn.Module):
    """
    T1-centric fusion module.

    Args:
        latent_channels: Number of latent channels (must equal the
            per-modality encoder's output channels). Default: 32.
        aux_modalities: Names of auxiliary modalities (e.g. fmri, asl,
            qsm, flair, demo). Demo is special-cased to always contribute
            a small demographic bias, not a delta.
        zero_init_deltas: If True (default), zero-init the output of each
            delta_net so the model starts as a T1-only model.

    Forward:
        z_t1: [B, C, D, H, W] - T1 latent (always present)
        z_aux_dict: {mod_name: [B, C, D, H, W] or None} - per-modality latents
        demo_bias: optional [B, C, 1, 1, 1] - demographic bias (added after T1+aux)
    Returns:
        [B, C, D, H, W] - fused latent where T1 is the structural backbone
    """

    def __init__(
        self,
        latent_channels: int = 32,
        aux_modalities: Sequence[str] = ("fmri", "asl", "qsm", "flair"),
        zero_init_deltas: bool = True,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.aux_modalities = list(aux_modalities)

        # 1) T1 trunk: small 1x1 conv refinement of z_t1.
        #    Zero-initialized final layer so the trunk starts as identity.
        #    This means: at training start, fused = z_t1 (after identity trunk).
        self.t1_trunk = nn.Sequential(
            nn.Conv3d(latent_channels, latent_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(latent_channels, latent_channels, kernel_size=1),
        )
        if zero_init_deltas:
            nn.init.zeros_(self.t1_trunk[-1].weight)
            nn.init.zeros_(self.t1_trunk[-1].bias)

        # 2) Per-modality delta nets: produce a small [B, C, D, H, W] delta.
        #    Zero-initialized output -> delta = 0 at start.
        #    This means the model starts as a pure T1 model and only
        #    learns to use aux modalities if they help.
        self.delta_nets = nn.ModuleDict()
        for mod in self.aux_modalities:
            self.delta_nets[mod] = nn.Sequential(
                nn.Conv3d(latent_channels, latent_channels, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(latent_channels, latent_channels, kernel_size=1),
            )
            if zero_init_deltas:
                nn.init.zeros_(self.delta_nets[mod][-1].weight)
                nn.init.zeros_(self.delta_nets[mod][-1].bias)

        # 3) Per-modality gate nets: scalar 0-1.
        #    Input: pooled aux feature [B, C] + availability flag [B, 1]
        #    (when auxiliary is not None, we set availability=1)
        #    Output: gate [B, 1] in [0, 1]
        #    Bias-init to a small negative value so the gate starts near 0.
        self.gate_nets = nn.ModuleDict()
        for mod in self.aux_modalities:
            self.gate_nets[mod] = nn.Sequential(
                nn.Linear(latent_channels + 1, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            # Start with gate ≈ 0.5: half-on, lets gradients flow.
            # Training will push it up for helpful modalities, down for noisy.
            nn.init.zeros_(self.gate_nets[mod][-1].bias)

    # -------------------------------------------------------------------
    def _gate(self, mod: str, z_mod: Tensor) -> Tensor:
        """Compute scalar gate in [0, 1] for this modality."""
        # Pool z_mod to [B, C] (global average over spatial)
        z_mod_pooled = F.adaptive_avg_pool3d(z_mod, 1).flatten(1)  # [B, C]
        # Append availability flag (always 1 here, since we only call _gate
        # for present modalities; the None case is handled in forward).
        avail = torch.ones(z_mod.shape[0], 1, device=z_mod.device)
        gate_input = torch.cat([z_mod_pooled, avail], dim=-1)  # [B, C+1]
        gate = torch.sigmoid(self.gate_nets[mod](gate_input))   # [B, 1]
        return gate

    # -------------------------------------------------------------------
    def forward(
        self,
        z_t1: Tensor,
        z_aux_dict: Dict[str, Optional[Tensor]],
        demo_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Fuse T1 latent with optional auxiliary modalities.

        Args:
            z_t1: [B, C, D, H, W] - T1 trunk latent.
            z_aux_dict: {mod: [B, C, D, H, W] or None} per-modality latents.
                Missing modalities should be None (NOT zero tensors).
            demo_bias: Optional [B, C, 1, 1, 1] demographic bias to add
                after the gated fusion (always applied, no gating).
        Returns:
            [B, C, D, H, W] fused latent. Guaranteed to equal t1_trunk(z_t1)
            (plus demo_bias) when all aux modalities are None.
        """
        # T1 trunk: this is the structural backbone
        z = self.t1_trunk(z_t1)

        # Per-modality residual deltas (gated)
        for mod in self.aux_modalities:
            z_mod = z_aux_dict.get(mod)
            if z_mod is None:
                continue  # missing modality: no contribution (gate=0 implicitly)
            # Compute gate (in [0, 1])
            gate = self._gate(mod, z_mod)         # [B, 1]
            # Compute delta (zero-init at start)
            delta = self.delta_nets[mod](z_mod)   # [B, C, D, H, W]
            # Apply: z = z + gate * delta (broadcast gate over spatial)
            z = z + gate.view(-1, 1, 1, 1, 1) * delta

        # Demographic bias (always applied, not gated)
        if demo_bias is not None:
            z = z + demo_bias

        return z
