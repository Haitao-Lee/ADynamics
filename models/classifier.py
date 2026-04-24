"""
Disease Classifier for Stage 2 of ADynamics.

A spatial pooling-based classifier that takes VAE latent features as input
and predicts the disease stage (NC, SCD, MCI, AD).

Uses AdaptiveAvgPool3d to reduce spatial dimensions before MLP classification,
preventing VRAM OOM and overfitting from direct flattening of HD latents.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiseaseClassifier(nn.Module):
    """
    MLP Classifier for Alzheimer's Disease Stage Prediction.

    Takes VAE latent features [B, C, D, H, W] and applies:
        1. AdaptiveAvgPool3d -> [B, C, 4, 4, 3]
        2. Flatten -> [B, C * 4 * 4 * 3]
        3. MLP -> logits for 4 disease stages

    Memory-efficient: Pooling reduces spatial dims from e.g. [16,16,12] to [4,4,3]
    regardless of original VAE latent size, giving fixed input dimension of
    ~latent_channels * 48 (e.g., 64 * 48 = 3072 with base_channels=32).

    Architecture:
        AdaptiveAvgPool3d -> Flatten -> Linear -> BatchNorm -> LeakyReLU -> Dropout
                          -> Linear -> BatchNorm -> LeakyReLU -> Dropout
                          -> ... (more layers)
                          -> Linear (output logits)

    Attributes:
        vae_latent_channels: Number of channels in VAE latent
        input_feature_dim: Fixed input dimension after pooling
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of disease stages (4)
        dropout: Dropout probability
    """

    def __init__(
        self,
        vae_latent_channels: int = 64,
        pooling_size: Tuple[int, int, int] = (4, 4, 3),
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize the Disease Classifier.

        Args:
            vae_latent_channels: Number of channels in VAE latent representation
            pooling_size: Target spatial size for AdaptiveAvgPool3d (D, H, W).
                          Default: (4, 4, 3) for HD VAE output [16, 16, 12]
            hidden_dims: List of hidden layer dimensions. If None, uses [512, 256, 128]
            num_classes: Number of disease stages (default: 4 for NC/SCD/MCI/AD)
            dropout: Dropout probability for regularization (default: 0.3)
        """
        super().__init__()

        self.vae_latent_channels = vae_latent_channels
        self.pooling_size = pooling_size
        self.num_classes = num_classes
        self.dropout = dropout

        # Calculate fixed input dimension after pooling
        self.input_feature_dim = vae_latent_channels * pooling_size[0] * pooling_size[1] * pooling_size[2]

        # Spatial pooling layer: reduces any [B, C, D, H, W] to [B, C, 4, 4, 3]
        self.spatial_pool = nn.AdaptiveAvgPool3d(pooling_size)

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Build MLP layers
        layers = []
        in_dim = self.input_feature_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize network weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z: Tensor) -> Tensor:
        """
        Forward pass through the classifier.

        Args:
            z: VAE latent features. Shape: [B, C, D, H, W] (5D from VAE)
               or [B, feature_dim] (1D, e.g., from flattened storage)

        Returns:
            Logits of shape [B, num_classes] for disease stages

        Note:
            When batch_size=1, BatchNorm1d with running stats may behave
            unexpectedly. For batch=1, consider using eval mode or increasing
            batch size for stable training.
        """
        # Handle 5D VAE latent: [B, C, D, H, W]
        if z.dim() == 5:
            # Pool to fixed spatial size: [B, C, 4, 4, 3]
            z = self.spatial_pool(z)
            # Flatten: [B, C * 4 * 4 * 3]
            z = torch.flatten(z, 1)
        elif z.dim() == 4:
            # 4D tensor [B, C, H, W] - treat as 3D with D=1
            z = z.unsqueeze(2)  # [B, C, 1, H, W]
            z = self.spatial_pool(z)
            z = torch.flatten(z, 1)
        elif z.dim() == 2:
            # Already 1D [B, feature_dim]
            pass
        else:
            raise ValueError(
                f"Expected 5D VAE latent [B,C,D,H,W] or 1D [B,feature_dim], "
                f"got shape {z.shape}"
            )

        # Batch size safety check: warn if BatchNorm1d sees batch=1
        if z.shape[0] == 1 and self.training:
            pass  # User should ensure adequate batch size for training

        # MLP forward
        h = self.mlp(z)
        # shape: [B, hidden_dims[-1]]

        logits = self.output_layer(h)
        # shape: [B, num_classes]

        return logits

    def predict(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Make predictions and return class indices and probabilities.

        Args:
            z: VAE latent features of shape [B, C, D, H, W]

        Returns:
            Tuple of (predicted_class_indices [B], predicted_probabilities [B, num_classes])
        """
        logits = self.forward(z)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    def get_class_weights(
        self,
        class_counts: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        Compute class weights for imbalanced datasets.

        Uses inverse frequency weighting to handle class imbalance.

        Args:
            class_counts: List of counts for each class [NC, SCD, MCI, AD]
            device: Device to place weights on

        Returns:
            Tensor of class weights [num_classes]
        """
        if class_counts is None:
            return torch.ones(self.num_classes, device=device)

        counts = torch.tensor(class_counts, dtype=torch.float32)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(weights)

        if device is not None:
            weights = weights.to(device)

        return weights


def classifier_ce_loss(
    logits: Tensor,
    targets: Tensor,
    weight: Optional[Tensor] = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    """
    Compute cross-entropy loss for disease classification.

    Args:
        logits: Model predictions of shape [B, num_classes]
        targets: Ground truth labels of shape [B] (class indices)
        weight: Optional class weights of shape [num_classes]
        label_smoothing: Label smoothing factor (0 = no smoothing)

    Returns:
        Scalar cross-entropy loss
    """
    loss = F.cross_entropy(
        logits,
        targets,
        weight=weight,
        label_smoothing=label_smoothing,
    )
    return loss


def classifier_accuracy(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Compute classification accuracy.

    Args:
        logits: Model predictions of shape [B, num_classes]
        targets: Ground truth labels of shape [B] (class indices)

    Returns:
        Scalar accuracy tensor
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).float()
    acc = correct.mean()
    return acc