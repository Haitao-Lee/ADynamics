"""
Disease Classifier for Stage 2 of ADynamics.

A simple MLP-based classifier that takes VAE latent features as input
and predicts the disease stage (NC, SCD, MCI, AD).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DiseaseClassifier(nn.Module):
    """
    MLP Classifier for Alzheimer's Disease Stage Prediction.

    Takes the flattened VAE latent representation as input and outputs
    logits for 4 disease stages: NC, SCD, MCI, AD.

    Architecture:
        Input -> Linear -> BatchNorm -> LeakyReLU -> Dropout
             -> Linear -> BatchNorm -> LeakyReLU -> Dropout
             -> ... (more layers)
             -> Linear (output logits)

    Attributes:
        latent_dim: Dimension of flattened VAE latent
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of disease stages (4)
        dropout: Dropout probability
    """

    def __init__(
        self,
        latent_dim: int = 524288,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 4,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize the Disease Classifier.

        Args:
            latent_dim: Dimension of flattened VAE latent (C * D' * H' * W')
            hidden_dims: List of hidden layer dimensions. If None, uses [2048, 1024, 512]
            num_classes: Number of disease stages (default: 4 for NC/SCD/MCI/AD)
            dropout: Dropout probability for regularization (default: 0.3)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.dropout = dropout

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512]

        # Build MLP layers
        layers = []
        in_dim = latent_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)

        # Initialize weights
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
            z: VAE latent features of shape [B, latent_dim] or [B, C, D', H', W']

        Returns:
            Logits of shape [B, num_classes] for disease stages
        """
        # Handle spatial latent by flattening
        if z.dim() == 5:
            # shape: [B, C, D', H', W'] -> [B, C * D' * H' * W']
            z = z.view(z.size(0), -1)
        elif z.dim() == 4:
            # shape: [B, C, H', W'] -> [B, C * H' * W']
            z = z.view(z.size(0), -1)

        # Ensure correct input dimension
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent_dim {self.latent_dim}, got {z.shape[1]}. "
                f"Ensure VAE latent dimensions are correct."
            )

        # MLP forward
        h = self.mlp(z)
        # shape: [B, hidden_dims[-1]]

        # Output logits
        logits = self.output_layer(h)
        # shape: [B, num_classes]

        return logits

    def predict(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Make predictions and return class indices and probabilities.

        Args:
            z: VAE latent features of shape [B, latent_dim] or [B, C, D', H', W']

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
            # Equal weights if no counts provided
            return torch.ones(self.num_classes, device=device)

        counts = torch.tensor(class_counts, dtype=torch.float32)
        # Inverse frequency
        weights = 1.0 / counts
        # Normalize
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
