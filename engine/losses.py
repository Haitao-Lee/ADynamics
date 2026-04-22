"""
Loss functions for ADynamics training.

Implements VAE losses (reconstruction + KL) and CFM-related losses.
"""

from typing import Optional

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


def vae_kl_loss(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Compute KL divergence loss for VAE latent regularization.

    KL divergence between N(mu, sigma) and N(0, 1):
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        mu: Mean of latent distribution of shape [B, C, D, H, W]
        logvar: Log variance of latent distribution of shape [B, C, D, H, W]

    Returns:
        Scalar KL divergence loss
    """
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_div


def total_vae_loss(
    recon: Tensor,
    target: Tensor,
    mu: Tensor,
    logvar: Tensor,
    kl_weight: float = 0.0001,
    recon_loss_type: str = "l1",
) -> Tensor:
    """
    Compute total VAE loss = reconstruction + kl_weight * KL.

    The KL term encourages the latent distribution to be close to a
    standard normal distribution, which aids in regularizing the latent
    space and improving generalization.

    Args:
        recon: Reconstructed MRI of shape [B, 1, D, H, W]
        target: Target MRI of shape [B, 1, D, H, W]
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        kl_weight: Weight for KL divergence term. Default: 0.0001
        recon_loss_type: Type of reconstruction loss ("l1" or "l2"). Default: "l1"

    Returns:
        Scalar total VAE loss
    """
    recon_loss = vae_reconstruction_loss(recon, target, loss_type=recon_loss_type)
    kl_loss = vae_kl_loss(mu, logvar)

    # Normalize KL by number of elements in latent for stable training
    num_latent_elements = mu.numel()
    kl_loss_normalized = kl_loss / num_latent_elements

    total_loss = recon_loss + kl_weight * kl_loss_normalized
    return total_loss


def cfm_loss(
    v_pred: Tensor,
    z0: Tensor,
    z1: Tensor,
    t: Optional[Tensor] = None,
) -> Tensor:
    """
    Conditional Flow Matching (CFM) loss.

    The CFM loss computes the MSE between the predicted velocity field
    and the optimal transport target (z1 - z0).

    L_CFM = || v_theta(z_t, t) - (z1 - z0) ||^2

    where z_t = (1-t)*z0 + t*z1 (linear interpolation in latent space)

    This is used in Stage 3 to train the vector field network.

    Args:
        v_pred: Predicted velocity field of shape [B, C, D, H, W]
        z0: Source latent (NC group) of shape [B, C, D, H, W]
        z1: Target latent (AD group) of shape [B, C, D, H, W]
        t: Time interpolation values of shape [B, 1]. If None, uses t~U(0,1)

    Returns:
        Scalar CFM loss
    """
    # Target velocity is the difference between target and source
    target_v = z1 - z0

    # Compute MSE between predicted and target velocity
    loss = F.mse_loss(v_pred, target_v)

    return loss


def deformation_smooth_loss(
    flow: Tensor,
    penalty_type: str = "l2",
) -> Tensor:
    """
    Compute smoothness penalty for deformation fields.

    Encourages the deformation field to be smooth (small gradients)
    which promotes anatomically plausible warps without discontinuities.

    L_smooth = mean(|grad(dx)|^2 + |grad(dy)|^2 + |grad(dz)|^2)

    Args:
        flow: 3D deformation field of shape [B, 3, D, H, W]
        penalty_type: Type of penalty ("l1" or "l2"). Default: "l2"

    Returns:
        Scalar smoothness loss
    """
    # Compute spatial gradients using finite differences
    # Flow shape: [B, 3, D, H, W] - 3 channels for x, y, z displacements

    # Gradient in D dimension (depth)
    grad_d = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    # Gradient in H dimension (height)
    grad_h = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    # Gradient in W dimension (width)
    grad_w = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

    if penalty_type == "l1":
        loss = (
            torch.mean(torch.abs(grad_d))
            + torch.mean(torch.abs(grad_h))
            + torch.mean(torch.abs(grad_w))
        )
    else:  # l2
        loss = (
            torch.mean(grad_d**2)
            + torch.mean(grad_h**2)
            + torch.mean(grad_w**2)
        )

    return loss


def dice_loss(
    pred: Tensor,
    target: Tensor,
    smooth: float = 1e-5,
) -> Tensor:
    """
    Compute Dice coefficient loss for segmentation tasks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Dice loss = 1 - Dice

    Args:
        pred: Predicted probabilities of shape [B, C, D, H, W]
        target: Target binary mask of shape [B, C, D, H, W]
        smooth: Smoothing factor to avoid division by zero

    Returns:
        Scalar Dice loss
    """
    # Flatten predictions and targets
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    return 1.0 - dice


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
        # flow shape: [B, 3, D, H, W]

        # Compute gradients using central differences
        # For interior points
        grad_d = flow[:, :, 2:, 1:-1, 1:-1] - flow[:, :, :-2, 1:-1, 1:-1]
        grad_h = flow[:, :, 1:-1, 2:, 1:-1] - flow[:, :, 1:-1, :-2, 1:-1]
        grad_w = flow[:, :, 1:-1, 1:-1, 2:] - flow[:, :, 1:-1, 1:-1, :-2]

        # Compute gradient norms for each displacement component
        # D dimension gradient
        grad_d_disp = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        grad_h_disp = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        grad_w_disp = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

        if self.penalty_type == "l1":
            loss = (
                torch.mean(torch.abs(grad_d_disp))
                + torch.mean(torch.abs(grad_h_disp))
                + torch.mean(torch.abs(grad_w_disp))
            )
        else:  # l2
            loss = (
                torch.mean(grad_d_disp**2)
                + torch.mean(grad_h_disp**2)
                + torch.mean(grad_w_disp**2)
            )

        return loss


class NegativeJacobianPenalty(nn.Module):
    """
    Negative Jacobian Determinant Penalty for 3D Deformation Fields.

    Penalizes negative Jacobian determinants which indicate folding or
    self-intersection in the deformation field. A valid deformation should
    have Jacobian > 0 everywhere (orientation preserving).

    For a deformation field φ(x) = x + flow(x), the Jacobian J = ∂φ/∂x
    For 3D: det(J) should be > 0 for all voxels.

    This loss penalizes det(J) < 0 regions:
        L_jac = max(0, -det(J))

    This is CRITICAL for medical imaging applications to ensure
    anatomically plausible deformations.
    """

    def __init__(
        self,
        epsilon: float = 1e-5,
    ) -> None:
        """
        Initialize Jacobian penalty.

        Args:
            epsilon: Small value to prevent numerical issues. Default: 1e-5
        """
        super().__init__()
        self.epsilon = epsilon

    def compute_jacobian_determinant(
        self,
        flow: Tensor,
        spacing: tuple = (1.0, 1.0, 1.0),
    ) -> Tensor:
        """
        Compute Jacobian determinant for each voxel in the deformation field.

        Uses central differences to compute the Jacobian matrix ∂φ/∂x
        where φ(x) = x + flow(x).

        Args:
            flow: Deformation field of shape [B, 3, D, H, W]
                flow[:, 0] = displacement in D dimension
                flow[:, 1] = displacement in H dimension
                flow[:, 2] = displacement in W dimension
            spacing: Voxel spacing (dx, dy, dz) for gradient normalization

        Returns:
            Jacobian determinant of shape [B, D-2, H-2, W-2]
        """
        B, _, D, H, W = flow.shape
        # shape: [B, 3, D, H, W]

        # Compute gradients of the deformation field
        # ∂flow_x/∂d, ∂flow_x/∂h, ∂flow_x/∂w
        # Using central differences

        # Grid for computing gradients
        device = flow.device

        # Compute identity Jacobian (due to x + flow)
        # J = I + ∂flow/∂x

        # Partial derivatives of flow_x (channel 0)
        dfx_dd = (flow[:, 0, 2:, 1:-1, 1:-1] - flow[:, 0, :-2, 1:-1, 1:-1]) / (2 * spacing[0])
        dfx_dh = (flow[:, 0, 1:-1, 2:, 1:-1] - flow[:, 0, 1:-1, :-2, 1:-1]) / (2 * spacing[1])
        dfx_dw = (flow[:, 0, 1:-1, 1:-1, 2:] - flow[:, 0, 1:-1, 1:-1, :-2]) / (2 * spacing[2])

        # Partial derivatives of flow_y (channel 1)
        dfy_dd = (flow[:, 1, 2:, 1:-1, 1:-1] - flow[:, 1, :-2, 1:-1, 1:-1]) / (2 * spacing[0])
        dfy_dh = (flow[:, 1, 1:-1, 2:, 1:-1] - flow[:, 1, 1:-1, :-2, 1:-1]) / (2 * spacing[1])
        dfy_dw = (flow[:, 1, 1:-1, 1:-1, 2:] - flow[:, 1, 1:-1, 1:-1, :-2]) / (2 * spacing[2])

        # Partial derivatives of flow_z (channel 2)
        dfz_dd = (flow[:, 2, 2:, 1:-1, 1:-1] - flow[:, 2, :-2, 1:-1, 1:-1]) / (2 * spacing[0])
        dfz_dh = (flow[:, 2, 1:-1, 2:, 1:-1] - flow[:, 2, 1:-1, :-2, 1:-1]) / (2 * spacing[1])
        dfz_dw = (flow[:, 2, 1:-1, 1:-1, 2:] - flow[:, 2, 1:-1, 1:-1, :-2]) / (2 * spacing[2])

        # Jacobian determinant: det(I + J_flow)
        # J_full[i,j] = δij + ∂flow_i/∂x_j
        # det = (1 + dfx_dx) * ((1 + dfy_dy) * (1 + dfz_dz) - dfy_dz * dfz_dy)
        #        - dfx_dy * (dfy_dx * (1 + dfz_dz) - dfy_dz * dfz_dx)
        #        + dfx_dz * (dfy_dx * dfz_dy - (1 + dfy_dy) * dfz_dx)

        # Compute determinant
        det = (
            (1 + dfx_dd) * ((1 + dfy_dh) * (1 + dfz_dw) - dfy_dw * dfz_dh)
            - dfx_dh * (dfy_dd * (1 + dfz_dw) - dfy_dw * dfz_dd)
            + dfx_dw * (dfy_dd * dfz_dh - (1 + dfy_dh) * dfz_dd)
        )

        return det

    def forward(self, flow: Tensor, spacing: tuple = (1.0, 1.0, 1.0)) -> Tensor:
        """
        Compute negative Jacobian penalty loss.

        Penalizes regions where det(J) <= 0.

        L = mean(max(0, -det(J)))

        Args:
            flow: Deformation field of shape [B, 3, D, H, W]
            spacing: Voxel spacing for gradient computation

        Returns:
            Scalar loss (average negative log-likelihood of valid Jacobians)
        """
        det = self.compute_jacobian_determinant(flow, spacing)
        # det shape: [B, D-2, H-2, W-2]

        # Penalize negative determinants
        # Using softplus-like penalty: max(0, -det)^2 for smoother gradients
        negative_det = torch.clamp(-det, min=0.0)
        loss = torch.mean(negative_det**2)

        # Also add a small penalty for determinants approaching zero
        # This helps push all determinants positive
        near_zero_penalty = torch.clamp(self.epsilon - det, min=0.0)
        loss = loss + 0.1 * torch.mean(near_zero_penalty**2)

        return loss


def total_deformation_loss(
    flow: Tensor,
    sim_weight: float = 1.0,
    smooth_weight: float = 0.1,
    jacobian_weight: float = 0.01,
    similarity_target: Optional[Tensor] = None,
    similarity_pred: Optional[Tensor] = None,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> Tensor:
    """
    Compute total deformation loss combining multiple terms.

    L_total = sim_weight * L_sim + smooth_weight * L_smooth + jacobian_weight * L_jac

    Args:
        flow: Deformation field [B, 3, D, H, W]
        sim_weight: Weight for similarity loss
        smooth_weight: Weight for smoothing loss
        jacobian_weight: Weight for Jacobian penalty
        similarity_target: Target image for similarity [B, 1, D, H, W]
        similarity_pred: Predicted/warped image [B, 1, D, H, W]
        spacing: Voxel spacing for Jacobian computation

    Returns:
        Scalar total deformation loss
    """
    total_loss = 0.0

    # Similarity loss
    if similarity_target is not None and similarity_pred is not None and sim_weight > 0:
        sim_loss = F.l1_loss(similarity_pred, similarity_target)
        total_loss = total_loss + sim_weight * sim_loss

    # Smoothing loss
    if smooth_weight > 0:
        smooth_loss = GradientSmoothingLoss(penalty_type="l2")(flow)
        total_loss = total_loss + smooth_weight * smooth_loss

    # Jacobian penalty
    if jacobian_weight > 0:
        jac_loss = NegativeJacobianPenalty()(flow, spacing)
        total_loss = total_loss + jacobian_weight * jac_loss

    return total_loss
