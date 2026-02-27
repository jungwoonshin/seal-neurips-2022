"""Backward Reconstruction Model for BiStruct.

Reconstructs feature representations from ground-truth labels by:
1. Looking up embeddings for active labels (shared with LSA)
2. Mean-pooling over active label embeddings
3. Passing through a bottleneck MLP to produce feature-space reconstruction
"""

import torch
import torch.nn as nn


class BackwardReconstructionModel(nn.Module):
    """Backward model that reconstructs features from labels.

    Args:
        label_embed_dim: Dimension d_l of label embeddings (from LSA).
        feature_dim: Dimension d of the feature space to reconstruct into.
        bottleneck_dim: Dimension r of the MLP bottleneck.
    """

    def __init__(
        self,
        label_embed_dim: int,
        feature_dim: int,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.label_embed_dim = label_embed_dim
        self.feature_dim = feature_dim
        self.bottleneck_dim = bottleneck_dim

        # 2-layer bottleneck MLP: d_l -> r -> d
        self.mlp = nn.Sequential(
            nn.Linear(label_embed_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, feature_dim),
        )

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_labels) binary labels
        label_embeddings: nn.Parameter,  # (num_labels, label_embed_dim) shared from LSA
    ) -> torch.Tensor:
        """Reconstruct features from ground-truth labels.

        Args:
            y: Ground-truth binary labels, shape (batch, num_labels).
            label_embeddings: Shared label embeddings from LSA, shape (num_labels, d_l).

        Returns:
            Reconstructed features h_hat, shape (batch, feature_dim).
        """
        # z_i = y_i * E_i: select embeddings of active labels
        # y: (batch, L, 1), label_embeddings: (1, L, d_l) -> z: (batch, L, d_l)
        y_expanded = y.unsqueeze(-1)  # (batch, L, 1)
        emb_expanded = label_embeddings.unsqueeze(0)  # (1, L, d_l)
        z = y_expanded * emb_expanded  # (batch, L, d_l)

        # Mean pool over active labels: z = Σ z_i / Σ y_i
        num_active = y.sum(dim=1, keepdim=True).clamp(min=1.0)  # (batch, 1)
        z_pooled = z.sum(dim=1) / num_active  # (batch, d_l)

        # MLP: d_l -> r -> d
        h_hat = self.mlp(z_pooled)  # (batch, feature_dim)

        return h_hat
