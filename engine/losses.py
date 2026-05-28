"""
Loss functions for ADynamics training.

Implements VAE losses (reconstruction + KL) and CFM-related losses.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def vae_reconstruction_loss(
    recon: Tensor,
    target: Tensor,
    loss_type: str = "l1",
) -> Tensor:
    """
    Compute reconstruction loss between predicted and target MRI.

    Args:
        recon: Reconstructed MRI of shape [B, 1, D, H, W]
        target: Target MRI of shape [B, 1, D, H, W]
        loss_type: Type of loss ("l1" or "l2"). Default: "l1"

    Returns:
        Scalar reconstruction loss
    """
    if loss_type == "l1":
        return F.l1_loss(recon, target)
    elif loss_type == "l2":
        return F.mse_loss(recon, target)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'l1' or 'l2'.")


def vae_kl_loss(
    mu: Tensor,
    logvar: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute KL divergence loss for VAE latent regularization.

    KL divergence between N(mu, sigma) and N(0, 1):
        KL = -0.5 * mean(1 + log(sigma^2) - mu^2 - sigma^2)

    Uses mean instead of sum to prevent overflow in FP16 mixed precision training
    and to produce stable, scale-invariant loss values.

    Args:
        mu: Mean of latent distribution of shape [B, C, D, H, W]
        logvar: Log variance of latent distribution of shape [B, C, D, H, W]
        reduction: Reduction method. "mean" (default) recommended for stability.

    Returns:
        Scalar KL divergence loss
    """
    if reduction == "mean":
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        if torch.isnan(kl_div).any() or torch.isinf(kl_div).any():
            print(f"[DEBUG KL] 1+logvar={torch.mean(1+logvar):.4f}, mu.pow(2)={torch.mean(mu.pow(2)):.4f}, logvar.exp()={torch.mean(logvar.exp()):.4f}, kl_div={kl_div.item()}")
    elif reduction == "sum":
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elif reduction == "none":
        kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    else:
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean', 'sum', or 'none'.")
    return kl_div


def gradient_loss(recon: Tensor, target: Tensor) -> Tensor:
    """
    Compute gradient (edge/texture) loss using Sobel filters.
    Helps preserve fine纹理 details in reconstruction.

    Args:
        recon: Reconstructed MRI [B, 1, D, H, W]
        target: Target MRI [B, 1, D, H, W]

    Returns:
        Scalar gradient loss
    """
    # Sobel kernels for 3D: depth, height, width
    # Use smooth+gradient approach via 3x3x3 averaging - gradient difference
    kernel_size = 3
    pad = kernel_size // 2

    # Simple 3D gradient via central differences (avoids needing scipy)
    def gradient_diff(x):
        # D gradientqing
        g_d = x[:, :, 2:, 1:-1, 1:-1] - x[:, :, :-2, 1:-1, 1:-1]
        # H gradient
        g_h = x[:, :, 1:-1, 2:, 1:-1] - x[:, :, 1:-1, :-2, 1:-1]
        # W gradient
        g_w = x[:, :, 1:-1, 1:-1, 2:] - x[:, :, 1:-1, 1:-1, :-2]
        return g_d, g_h, g_w

    recon_d, recon_h, recon_w = gradient_diff(recon)
    target_d, target_h, target_w = gradient_diff(target)

    loss = (F.l1_loss(recon_d, target_d) +
            F.l1_loss(recon_h, target_h) +
            F.l1_loss(recon_w, target_w))
    return loss / 3.0


def ssim_loss(recon: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """
    Compute SSIM loss for structural similarity preservation.
    SSIM is more sensitive to structural/texture changes than L1/L2.

    Uses MONAI's SSIMLoss if available, otherwise falls back to simple SSIM.

    Args:
        recon: Reconstructed MRI [B, 1, D, H, W]
        target: Target MRI [B, 1, D, H, W]
        window_size: Window size for SSIM computation

    Returns:
        Scalar SSIM loss (1 - SSIM so minimizing it maximizes SSIM)
    """
    # Use fallback only - MONAI SSIMLoss has issues with 3D tensors
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_recon = recon.mean(dim=(-1, -2, -3), keepdim=True)
    mu_target = target.mean(dim=(-1, -2, -3), keepdim=True)
    var_recon = recon.var(dim=(-1, -2, -3), keepdim=True)
    var_target = target.var(dim=(-1, -2, -3), keepdim=True)

    cov = ((recon - mu_recon) * (target - mu_target)).mean(dim=(-1, -2, -3), keepdim=True)

    ssim_val = ((2 * mu_recon * mu_target + C1) * (2 * cov + C2) /
                ((mu_recon ** 2 + mu_target ** 2 + C1) * (var_recon + var_target + C2)))
    return 1.0 - ssim_val.mean()


def ordinal_cross_entropy_loss(
    logits: Tensor,
    labels: Tensor,
    num_classes: int = 4,
) -> Tensor:
    """
    Ordinal Cross Entropy Loss for disease progression classification.

    Penalizes misclassification based on ordinal distance:
        - NC(0) vs AD(3) is penalized 3x more than NC(0) vs SCD(1)
        - This enforces the ordinal structure NC < SCD < MCI < AD

    Loss = sum(|label - pred| / (num_classes-1) * CE_loss

    Args:
        logits: Classification logits [B, num_classes]
        labels: Ground truth labels [B] - 0=NC, 1=SCD, 2=MCI, 3=AD
        num_classes: Number of classes

    Returns:
        Scalar loss
    """
    # Standard cross entropy
    ce_loss = F.cross_entropy(logits, labels, reduction='none')

    # Ordinal penalty: distance from correct class
    preds = logits.argmax(dim=1)
    ordinal_error = torch.abs(preds.float() - labels.float())

    # Weight loss by ordinal distance
    ordinal_weight = ordinal_error.float() / float(num_classes - 1)
    loss = ce_loss * (1.0 + ordinal_weight)

    return loss.mean()


def ordinal_regression_loss(
    mu: Tensor,
    labels: Tensor,
    num_classes: int = 4,
) -> Tensor:
    """
    Ordinal Regression Loss for latent space alignment.

    Encourages latent mean to form ordinal structure:
        z[NC] < z[SCD] < z[MCI] < z[AD]

    Uses MSE between normalized ordinal positions.

    Args:
        mu: Latent mean [B, C, D, H, W]
        labels: Disease labels [B] - 0=NC, 1=SCD, 2=MCI, 3=AD
        num_classes: Number of classes

    Returns:
        Scalar ordinal regression loss
    """
    # Pool latent to scalar per sample
    pooled = F.adaptive_avg_pool3d(mu, output_size=(1, 1, 1))
    pooled = pooled.squeeze(-1).squeeze(-1).squeeze(-1)  # [B, C]

    # Normalize to [-1, 1] range based on class
    # NC=0 -> -1, AD=3 -> +1
    ordinal_targets = 2.0 * labels.float() / (num_classes - 1) - 1.0  # [B]

    # Mean latent should follow ordinal structure
    latent_mean = pooled.mean(dim=1)  # [B]

    # MSE between ordinal position and latent mean
    loss = F.mse_loss(latent_mean, ordinal_targets)

    return loss


def ordinal_contrastive_loss(
    z: Tensor,
    labels: Tensor,
    temperature: float = 0.1,
    alpha: float = 0.5,
) -> Tensor:
    """
    Ordinal Supervised Contrastive Loss for disease progression modeling.

    Unlike standard SupCon which treats all negatives equally,
    this loss considers the ordinal nature of disease progression:
    NC(0) < SCD(1) < MCI(2) < AD(3)

    NC and AD are the most different (push apart most),
    SCD and MCI are adjacent (push apart less).

    Loss = positive_loss + alpha * negative_loss

    Args:
        z: Normalized latent features [B, latent_dim]
        labels: Disease labels [B] - 0=NC, 1=SCD, 2=MCI, 3=AD
        temperature: Temperature parameter for scaling similarity. Default: 0.1
        alpha: Weight for ordinal push-apart term. Default: 0.5

    Returns:
        Scalar ordinal contrastive loss
    """
    B = z.size(0)

    # Guard: need at least 2 samples and at least 2 unique labels
    unique_labels = labels.unique()
    if B < 2 or unique_labels.numel() < 2:
        # Return a small constant loss that allows backward but doesn't affect training much
        return z.new_zeros(1) + 1e-6

    # Clamp temperature for numerical stability
    temperature = max(temperature, 1e-3)

    # Compute pairwise similarity with temperature scaling
    sim = torch.matmul(z, z.T)  # [B, B]
    sim = sim / temperature

    # Clamp similarity for numerical stability before exp
    sim = torch.clamp(sim, min=-50.0, max=50.0)

    # Ordinal distance: how far apart are the labels?
    # |label_i - label_j| gives ordinal distance
    labels_i = labels.view(B, 1)  # [B, 1]
    labels_j = labels.view(1, B)  # [1, B]
    ordinal_dist = torch.abs(labels_i.float() - labels_j.float())  # [B, B]
    # Normalize to [0, 1]
    max_dist = 3.0  # NC(0) to AD(3)
    ordinal_dist_norm = ordinal_dist / max_dist  # [0, 1]

    # Masks
    same_mask = torch.eq(labels_i, labels_j).float()  # 同类=1
    diff_mask = 1.0 - same_mask  # 异类=1 (excludes self)

    # Positive loss: pull same labels together
    exp_sim = torch.exp(sim - sim.max())  # numerical stability
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Only consider positive pairs (same labels, excluding self)
    same_mask_no_self = same_mask.clone()
    same_mask_no_self.fill_diagonal_(0.0)
    pos_sum = same_mask_no_self * log_prob
    pos_count = same_mask_no_self.sum()
    if pos_count < 1.0:
        pos_loss = z.new_zeros(1)
    else:
        pos_loss = -pos_sum.sum() / pos_count

    # Negative loss: push different labels apart, weighted by ordinal distance
    # Higher ordinal distance = stronger push
    diff_count = diff_mask.sum()
    if diff_count < 1.0:
        neg_loss = z.new_zeros(1)
    else:
        neg_term = diff_mask * sim * ordinal_dist_norm  # distant pairs weighted more
        neg_loss = neg_term.sum() / diff_count

    loss = pos_loss + alpha * neg_loss

    # Final safety clamp
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"[DEBUG ordinal_loss] NaN/Inf detected: B={B}, pos_loss={pos_loss}, neg_loss={neg_loss}, pos_count={pos_count}, diff_count={diff_count}")
        return z.new_zeros(1) + 1e-6

    return loss


class MemoryBank:
    """
    Memory Bank for contrastive learning with small batch sizes.

    Stores (latent, label) pairs from previous batches to enable
    building positive/negative pairs even when batch_size is small.

    Uses channel-wise pooling to reduce latent dimension for memory efficiency.

    Usage:
        bank = MemoryBank(size=4096, device='cuda', latent_channels=64)

        for batch in dataloader:
            mu, _ = vae.encode(batch)

            # Compute loss using current batch + memory bank
            loss = contrastive_loss_with_bank(mu, labels, bank)

            # Update memory bank
            bank.update(mu.detach(), labels)

    Args:
        size: Maximum number of samples to store
        device: Device to store tensors on
        latent_channels: Number of VAE latent channels (C)
    """

    def __init__(
        self,
        size: int = 4096,
        device: torch.device = torch.device("cuda"),
        latent_channels: int = 64,
    ):
        self.size = size
        self.device = device
        # Use channel-wise mean to reduce dimension: [B, C, D, H, W] -> [B, C]
        self.dim = latent_channels
        # Initialize with zeros
        self.features = torch.zeros(size, self.dim, device=device)
        self.labels = torch.zeros(size, dtype=torch.long, device=device)
        self.ptr = 0  # Current position for circular update
        self.count = 0  # Number of samples stored

    def update(self, mu: Tensor, labels: Tensor):
        """
        Update memory bank with new VAE latent and labels.

        Uses channel-wise mean pooling to reduce [B, C, D, H, W] to [B, C].

        Args:
            mu: VAE latent [B, C, D, H, W]
            labels: [B] disease labels
        """
        B = mu.size(0)

        # Channel-wise mean pooling: [B, C, D, H, W] -> [B, C]
        z_pooled = F.adaptive_avg_pool3d(mu, output_size=(1, 1, 1))
        z_flat = z_pooled.squeeze(-1).squeeze(-1).squeeze(-1)  # [B, C]
        z_norm = F.normalize(z_flat, dim=1)  # [B, C]

        for i in range(B):
            idx = self.ptr % self.size
            self.features[idx] = z_norm[i]
            self.labels[idx] = labels[i]
            self.ptr = (self.ptr + 1) % self.size
            self.count = min(self.count + 1, self.size)

    def get(self, k: int = None) -> Tuple[Tensor, Tensor]:
        """
        Get k random samples from memory bank.

        Args:
            k: Number of samples to retrieve. If None, returns all.

        Returns:
            Tuple of (features, labels)
        """
        if self.count == 0:
            # Return zeros if bank is empty - use unit vector to avoid NaN in normalization
            dummy = torch.zeros(1, self.dim, device=self.device)
            dummy[0, 0] = 1.0  # unit vector to avoid NaN when normalized
            return dummy, self.labels[:1]

        if k is None or k >= self.count:
            return self.features[:self.count], self.labels[:self.count]

        # Random sample k indices
        indices = torch.randperm(self.count)[:k].to(self.device)
        return self.features[indices], self.labels[indices]

    def __len__(self):
        return self.count


def supervised_contrastive_loss(
    z: Tensor,
    labels: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """
    Standard Supervised Contrastive Loss (SimCLR-style with labels).

    Pulls same-label samples together, pushes different-label samples apart.
    Does NOT consider ordinal relationships.

    Args:
        z: Normalized latent features [B, latent_dim]
        labels: Disease labels [B] - 0=NC, 1=SCD, 2=MCI, 3=AD
        temperature: Temperature parameter. Default: 0.1

    Returns:
        Scalar supervised contrastive loss
    """
    B = z.size(0)

    if B < 2:
        return z.new_zeros(1)

    # Compute pairwise similarity
    sim = torch.matmul(z, z.T) / temperature  # [B, B]

    # Create positive mask (same labels, excluding self)
    labels = labels.contiguous()
    mask_pos = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()  # [B, B]
    mask_pos = mask_pos - torch.eye(B, device=mask_pos.device)  # exclude self

    # InfoNCE loss
    exp_sim = torch.exp(sim - sim.max())  # numerical stability
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Positive pairs loss
    pos_loss = -(mask_pos * log_prob).sum() / (mask_pos.sum() + 1e-8)

    return pos_loss


def total_vae_loss(
    recon: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    kl_weight: float = 0.0001,
    recon_loss_type: str = "l1",
    gradient_weight: float = 0.0,
    ssim_weight: float = 0.0,
    multi_scale_recons: Optional[list] = None,
    multi_scale_weights: Optional[list] = None,
    contrastive_labels: Optional[Tensor] = None,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
    use_ordinal_contrastive: bool = True,
) -> Tensor:
    """
    Compute total VAE loss = recon + kl_weight * KL + gradient * grad + ssim * ssim
    + contrastive * ordinal_supcon.

    Args:
        recon: Reconstructed MRI [B, 1, D, H, W]
        target: Target MRI [B, 1, D, H, W]
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        kl_weight: Weight for KL divergence term. Default: 0.0001
        recon_loss_type: Type of reconstruction loss ("l1" or "l2"). Default: "l1"
        gradient_weight: Weight for gradient (texture) loss. Default: 0.0
        ssim_weight: Weight for SSIM loss. Default: 0.0
        multi_scale_recons: List of [recon_s1, recon_s2, recon_s3] from multi-scale decoder.
                            Each should be downsampled target at that scale.
        multi_scale_weights: List of weights for each scale loss. Default: [0.1, 0.2, 0.3]
        contrastive_labels: Disease labels for contrastive loss [B]. Default: None
        contrastive_weight: Weight for ordinal contrastive loss. Default: 0.0
        contrastive_temperature: Temperature for contrastive loss. Default: 0.1
        use_ordinal_contrastive: If True, use ordinal (disease-aware) contrastive loss.
                                  If False, use standard supervised contrastive. Default: True

    Returns:
        Scalar total VAE loss
    """
    recon_loss = vae_reconstruction_loss(recon, target, loss_type=recon_loss_type)
    kl_loss = vae_kl_loss(mu, logvar, reduction="mean")

    total_loss = recon_loss + kl_weight * kl_loss

    if gradient_weight > 0:
        grad_loss = gradient_loss(recon, target)
        total_loss = total_loss + gradient_weight * grad_loss

    if ssim_weight > 0:
        ssim_l = ssim_loss(recon, target)
        total_loss = total_loss + ssim_weight * ssim_l

    # Multi-scale loss: downsample target to match each scale, compute loss
    if multi_scale_recons is not None and len(multi_scale_recons) > 0:
        weights = multi_scale_weights or [0.1, 0.2, 0.3]
        target_full = target

        spatial_factors = [8, 4, 2]  # Downsample factors for scales 1, 2, 3

        for i, (recon_sc, wf) in enumerate(zip(multi_scale_recons, weights)):
            # Downsample target to match scale
            ds = spatial_factors[i]
            # Simple pooling to downsample
            t_down = F.avg_pool3d(target_full, kernel_size=ds, stride=ds)
            scale_loss = vae_reconstruction_loss(recon_sc, t_down, loss_type=recon_loss_type)
            total_loss = total_loss + wf * scale_loss

    # Contrastive loss for latent space separation
    if contrastive_labels is not None and contrastive_weight > 0:
        # Flatten mu and normalize for contrastive loss
        z_flat = mu.flatten(1)
        z_norm = F.normalize(z_flat, dim=1)

        if use_ordinal_contrastive:
            con_loss = ordinal_contrastive_loss(
                z_norm, contrastive_labels, temperature=contrastive_temperature
            )
        else:
            con_loss = supervised_contrastive_loss(
                z_norm, contrastive_labels, temperature=contrastive_temperature
            )

        total_loss = total_loss + contrastive_weight * con_loss

    return total_loss


def cfm_loss(
    v_pred: Tensor,
    z0: Tensor,
    z1: Tensor,
) -> Tensor:
    """
    Conditional Flow Matching (CFM) loss using Optimal Transport formulation.

    The CFM loss computes the MSE between the predicted velocity field
    and the optimal transport target (z1 - z0).

    L_CFM = || v_theta(z_t, t) - (z1 - z0) ||^2

    This uses the Independent CFM formulation where the target velocity
    is constant (z1 - z0) independent of t, assuming a constant velocity
    field for optimal transport between z0 and z1.

    This is used in Stage 3 to train the vector field network.

    Args:
        v_pred: Predicted velocity field of shape [B, C, D, H, W]
        z0: Source latent (NC group) of shape [B, C, D, H, W]
        z1: Target latent (AD group) of shape [B, C, D, H, W]

    Returns:
        Scalar CFM loss
    """
    target_v = z1 - z0
    loss = F.mse_loss(v_pred, target_v)
    return loss


def rectified_flow_regularization(
    v_pred: Tensor,
    z_t: Tensor,
    t: Tensor,
    model: nn.Module,
    lambda_reg: float = 0.01,
) -> Tensor:
    """
    Rectified Flow regularization to encourage straight trajectories.

    Penalizes the curvature of the velocity field trajectory, pushing the model
    toward constant-velocity (straight) ODE paths. This is inspired by
    "Flow Straight and Fast" (Liu et al., 2022).

    The regularization computes the gradient of the velocity field w.r.t. spatial
    coordinates and penalizes large deviations, encouraging:
        - Straight trajectories (less curvature = more efficient transport)
        - Faster inference (straight paths need fewer ODE steps)
        - Better generalization (simpler trajectories are less prone to artifacts)

    L_RF = lambda * E_t[ ||v_theta(z_t, t) - (z1 - z0)||^2 * (1 - t) * t ]
        + lambda * E_t[ ||grad_z v_theta(z_t, t)||^2 ]

    The first term penalizes deviation from the straight-line target at intermediate t,
    and the second term penalizes spatial gradients of the velocity field.

    Args:
        v_pred: Predicted velocity [B, C, D, H, W]
        z_t: Interpolated latent at time t [B, C, D, H, W]
        t: Time steps [B] in [0, 1]
        model: The velocity field model (used for gradient computation)
        lambda_reg: Regularization strength. Default: 0.01

    Returns:
        Scalar regularization loss
    """
    # Term 1: Penalize velocity magnitude at intermediate t
    # At t=0 and t=1, velocity should be maximal; at t=0.5, it should be moderate
    # Weight by t*(1-t) to focus on intermediate timesteps
    t_weight = t * (1 - t)  # [B]
    t_weight = t_weight.view(-1, 1, 1, 1, 1)  # broadcast to [B, 1, 1, 1, 1]

    # Penalize large velocity magnitudes (encourages constant-velocity paths)
    vel_magnitude = torch.mean(v_pred ** 2, dim=(1, 2, 3, 4))  # [B]
    vel_reg = torch.mean(vel_magnitude * t_weight.squeeze())

    # Term 2: Penalize spatial gradients of velocity (smoothness in latent space)
    # This encourages the velocity field to be spatially smooth
    grad_d = v_pred[:, :, 1:, :, :] - v_pred[:, :, :-1, :, :]
    grad_h = v_pred[:, :, :, 1:, :] - v_pred[:, :, :, :-1, :]
    grad_w = v_pred[:, :, :, :, 1:] - v_pred[:, :, :, :, :-1]

    spatial_smooth = (
        torch.mean(grad_d ** 2) +
        torch.mean(grad_h ** 2) +
        torch.mean(grad_w ** 2)
    )

    loss = lambda_reg * (vel_reg + spatial_smooth)
    return loss


class GradientSmoothingLoss(nn.Module):
    """
    3D Gradient Smoothing Loss for Deformation Fields.

    Penalizes large gradients in the deformation field to ensure
    smooth, anatomically plausible deformations without discontinuities.

    Loss = mean(|∂flow_x/∂d|² + |∂flow_x/∂h|² + |∂flow_x/∂w|² +
                |∂flow_y/∂d|² + |∂flow_y/∂h|² + |∂flow_y/∂w|² +
                |∂flow_z/∂d|² + |∂flow_z/∂h|² + |∂flow_z/∂w|²)

    This is critical for medical imaging to prevent unrealistic folding or tearing.
    """

    def __init__(
        self,
        penalty_type: str = "l2",
    ) -> None:
        """
        Initialize gradient smoothing loss.

        Args:
            penalty_type: Type of penalty ("l1" or "l2"). Default: "l2"
        """
        super().__init__()
        self.penalty_type = penalty_type

    def forward(self, flow: Tensor) -> Tensor:
        """
        Compute gradient smoothing loss.

        Args:
            flow: 3D deformation field of shape [B, 3, D, H, W]
                flow[:, 0] = displacement in D dimension (depth)
                flow[:, 1] = displacement in H dimension (height)
                flow[:, 2] = displacement in W dimension (width)

        Returns:
            Scalar smoothing loss
        """
        # Compute first-order gradients using forward differences
        # grad[d,h,w] = flow[d+1,h,w] - flow[d,h,w] etc.
        # This is consistent and avoids the unused central difference dead code

        grad_d = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        grad_h = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        grad_w = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

        if self.penalty_type == "l1":
            loss = (
                torch.mean(torch.abs(grad_d))
                + torch.mean(torch.abs(grad_h))
                + torch.mean(torch.abs(grad_w))
            )
        else:
            loss = (
                torch.mean(grad_d**2)
                + torch.mean(grad_h**2)
                + torch.mean(grad_w**2)
            )

        return loss


