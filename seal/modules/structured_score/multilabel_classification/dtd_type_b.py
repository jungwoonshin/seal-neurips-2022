"""
DTD Type B Energy: Input-Conditional Interactions for Input-Mediated Dependencies.

For input-mediated dependencies (e.g., "ML" <-> "NLP"), provides input-dependent
label interactions through factored parameterization:

    z_i(x) = P * (e_i . (Q * T_E(x)))  in R^r

    E^B(x, y) = ||Z(x)^T (y . t^B)||^2

where:
    Q in R^{d x h}: shared input projection
    e_i in R^d: per-label SVD embedding
    P in R^{r x d}: shared rank projection
    r = 16, d = 32 typically

Computation is O(L*d + d*h), linear in L (no explicit O(L^2) pairwise computation).
"""

from typing import Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from seal.modules.structured_score import StructuredScore
from .dtd_type_gate import DTDTypeGate


@StructuredScore.register("dtd-type-b")
class DTDTypeBEnergy(StructuredScore):
    """Type B energy: input-conditional factored label interactions."""

    def __init__(
        self,
        type_gate: DTDTypeGate,
        num_labels: int,
        input_feature_dim: int,
        svd_rank: int = 32,
        low_rank_dim: int = 16,
        svd_embeddings_init: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            type_gate: Shared DTDTypeGate module for computing t^B gates.
            num_labels: Number of labels L.
            input_feature_dim: Dimension h of task network hidden features.
            svd_rank: Dimension d of SVD label embeddings.
            low_rank_dim: Dimension r of low-rank projection (interaction rank).
            svd_embeddings_init: Optional (L, svd_rank) tensor for initializing label embeddings.
        """
        super().__init__()
        self.type_gate = type_gate
        self.num_labels = num_labels
        self.svd_rank = svd_rank
        self.low_rank_dim = low_rank_dim

        # Q: shared input projection (h -> d)
        self.input_proj = nn.Linear(input_feature_dim, svd_rank, bias=False)
        nn.init.kaiming_uniform_(self.input_proj.weight, a=math.sqrt(5))

        # P: shared rank projection (d -> r)
        self.rank_proj = nn.Linear(svd_rank, low_rank_dim, bias=False)
        nn.init.kaiming_uniform_(self.rank_proj.weight, a=math.sqrt(5))

        # Per-label embeddings e_i in R^d
        if svd_embeddings_init is not None:
            assert svd_embeddings_init.shape == (num_labels, svd_rank), (
                f"Expected ({num_labels}, {svd_rank}), got {svd_embeddings_init.shape}"
            )
            self.label_embeddings = nn.Parameter(svd_embeddings_init.float())
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
        """Compute Type B energy via the squared-norm trick.

        E^B(x, y) = ||Z(x)^T (y . t^B)||^2

        Steps:
            1. q(x) = Q * T_E(x)                        -> (batch, d)
            2. z_i(x) = P * (e_i . q(x))                -> (batch, L, r)
            3. weighted_y = y * t^B                      -> (batch, num_samples, L)
            4. agg = Z(x)^T * weighted_y                 -> (batch, num_samples, r)
            5. energy = ||agg||^2                         -> (batch, num_samples)

        Args:
            y: Label predictions, shape (batch, num_samples, num_labels)
            buffer: Must contain "task_features" of shape (batch, h) from DTD ScoreNN.

        Returns:
            Energy tensor of shape (batch, num_samples)
        """
        # Get task features from buffer (stashed by DTD ScoreNN)
        task_features = buffer["task_features"]  # (batch, h)

        batch_size = task_features.shape[0]
        num_samples = y.shape[1]

        # Step 1: Q * T_E(x) -> (batch, d)
        q_x = self.input_proj(task_features)  # (batch, svd_rank)

        # Step 2: e_i . q(x) for all labels, then project through P
        # label_embeddings: (L, d), q_x: (batch, d)
        # Elementwise modulation: (batch, L, d)
        modulated = self.label_embeddings.unsqueeze(0) * q_x.unsqueeze(1)

        # Apply rank projection P: (batch, L, d) -> (batch, L, r)
        Z = self.rank_proj(modulated)  # (batch, L, low_rank_dim)

        # Step 3: Weight y by Type B gate
        t_b = self.type_gate.gate_b_vector()  # (L,)
        weighted_y = y * t_b.unsqueeze(0).unsqueeze(0)  # (batch, num_samples, L)

        # Step 4: Z(x)^T * weighted_y -> (batch, num_samples, r)
        # Z: (batch, L, r), weighted_y: (batch, num_samples, L)
        # We need: for each sample, Z^T @ weighted_y_s
        # weighted_y: (batch, num_samples, L) -> transpose to (batch, L, num_samples)
        # Z^T: (batch, r, L)
        # Result: (batch, r, num_samples) -> transpose to (batch, num_samples, r)
        agg = torch.bmm(
            Z.transpose(1, 2),  # (batch, r, L)
            weighted_y.transpose(1, 2),  # (batch, L, num_samples)
        ).transpose(1, 2)  # (batch, num_samples, r)

        # Step 5: Squared norm -> (batch, num_samples)
        energy = (agg * agg).sum(dim=-1)  # (batch, num_samples)

        return energy
