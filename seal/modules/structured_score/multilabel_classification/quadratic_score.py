"""Quadratic global energy: y^T W y.

Captures pairwise label interactions via a learnable weight matrix.
"""

from typing import Dict, Any, Optional
from seal.modules.structured_score import StructuredScore
import torch
import torch.nn as nn
import math


@StructuredScore.register("multi-label-quadratic")
class QuadraticGlobalScore(StructuredScore):
    def __init__(
        self,
        num_labels: int,
        symmetric: bool = True,
    ):
        """
        Args:
            num_labels: Number of labels (L).
            symmetric: If True, symmetrize W as (W + W^T) / 2.
        """
        super().__init__()
        self.num_labels = num_labels
        self.symmetric = symmetric
        self.W = nn.Parameter(
            torch.normal(
                0.0,
                math.sqrt(2.0 / num_labels),
                (num_labels, num_labels),
            )
        )

    def _get_W(self) -> torch.Tensor:
        W = self.W
        if self.symmetric:
            W = (W + W.T) / 2
        return W

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        W = self._get_W()

        # y^T W y for each sample in batch
        # (batch, num_samples, L) @ (L, L) -> (batch, num_samples, L)
        Wy = torch.matmul(y, W)  # (batch, num_samples, num_labels)
        # Element-wise multiply and sum: y * (Wy) summed over labels
        score = torch.sum(y * Wy, dim=-1)  # (batch, num_samples)

        return score

    def compute_vector_energy(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        """Per-label decomposition: y_i * (Wy)_i."""
        W = self._get_W()
        Wy = torch.matmul(y, W)  # (batch, num_samples, num_labels)
        return y * Wy  # (batch, num_samples, L)
