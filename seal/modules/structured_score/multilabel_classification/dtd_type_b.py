"""DTD Type B Energy: input-mediated label dependencies.

Computes:
    z_i(x) = P * (e_i . (Q * T_E(x)))   per-label input-conditional embedding
    E^B(x, y) = ||Z(x)^T (y . t^B)||^2   squared norm of gated projection

where T_E(x) are task features stashed in buffer by DTD ScoreNN.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from seal.modules.structured_score import StructuredScore
from .dtd_type_gate import DTDTypeGate


@StructuredScore.register("dtd-type-b")
class DTDTypeBEnergy(StructuredScore):
    """Structured energy for input-mediated label dependencies.

    Uses SVD-initialized label embeddings and task network features
    to compute an input-conditional energy that captures how label
    dependencies vary with the input.
    """

    def __init__(
        self,
        type_gate: DTDTypeGate,
        num_labels: int,
        input_feature_dim: int,
        svd_rank: int = 32,
        low_rank_dim: int = 16,
        svd_embeddings_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.type_gate = type_gate
        self.num_labels = num_labels
        self.svd_rank = svd_rank
        self.low_rank_dim = low_rank_dim

        # Q: projects task features to SVD embedding space
        self.Q = nn.Linear(input_feature_dim, svd_rank, bias=False)

        # P: projects from SVD space to low-rank output space
        self.P = nn.Linear(svd_rank, low_rank_dim, bias=False)

        # Per-label SVD embeddings: (L, svd_rank)
        if svd_embeddings_init is not None:
            assert svd_embeddings_init.shape == (num_labels, svd_rank)
            self.label_embeddings = nn.Parameter(svd_embeddings_init.clone())
        else:
            self.label_embeddings = nn.Parameter(
                torch.randn(num_labels, svd_rank) * 0.01
            )

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch, num_samples, num_labels = y.shape

        # Get task features from buffer (stashed by DTD ScoreNN)
        task_features = buffer.get("task_features")  # (batch, input_feature_dim)
        if task_features is None:
            # Fallback: return zeros if no features available
            return y.new_zeros(batch, num_samples)

        # Compute input-conditional label embeddings
        # q(x) = Q * T_E(x): (batch, svd_rank)
        q_x = self.Q(task_features)  # (batch, svd_rank)

        # z_i(x) = P * (e_i . q(x)): per-label embedding modulated by input
        # label_embeddings: (L, svd_rank), q_x: (batch, svd_rank)
        # Elementwise: (batch, L, svd_rank) = (1, L, svd_rank) * (batch, 1, svd_rank)
        modulated = self.label_embeddings.unsqueeze(0) * q_x.unsqueeze(1)
        # (batch, L, svd_rank)

        # Project to low-rank: Z(x) of shape (batch, L, low_rank_dim)
        Z = self.P(modulated)  # (batch, L, low_rank_dim)

        # Get Type B gate: (L,)
        t_b = self.type_gate.gate_b_vector()  # (L,)

        # Gated label vector: y . t^B -> (batch, num_samples, L)
        y_gated = y * t_b.unsqueeze(0).unsqueeze(0)  # (batch, num_samples, L)

        # E^B = ||Z(x)^T (y . t^B)||^2
        # Z: (batch, L, low_rank_dim), y_gated: (batch, num_samples, L)
        # Z^T @ y_gated: (batch, num_samples, low_rank_dim)
        # We need: for each sample, compute Z^T @ y_gated_sample
        # y_gated: (batch, num_samples, L) -> (batch, num_samples, L, 1)
        # Z: (batch, L, low_rank_dim) -> (batch, 1, L, low_rank_dim)
        # matmul: (batch, num_samples, 1, low_rank_dim) via y^T @ Z
        projection = torch.einsum("bsl,bld->bsd", y_gated, Z)
        # (batch, num_samples, low_rank_dim)

        # Squared norm
        energy = (projection ** 2).sum(dim=-1)  # (batch, num_samples)

        return energy
