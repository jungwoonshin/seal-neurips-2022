"""Attention-based global energy over labels.

Treats each label as a token, applies multi-head self-attention to capture
label-label dependencies, then reduces to a scalar energy.
"""

from typing import Dict, Any, Optional
from seal.modules.structured_score import StructuredScore
import torch
import torch.nn as nn
import math


@StructuredScore.register("multi-label-attention")
class AttentionGlobalScore(StructuredScore):
    def __init__(
        self,
        num_labels: int,
        label_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        reduction: str = "sum",
    ):
        """
        Args:
            num_labels: Number of labels (L).
            label_dim: Embedding dimension per label for attention.
            num_heads: Number of attention heads.
            dropout: Attention dropout probability.
            reduction: How to reduce attention output to scalar ("sum" or "max").
        """
        super().__init__()
        self.num_labels = num_labels
        self.label_dim = label_dim
        self.reduction = reduction

        # Project each scalar label y_i to label_dim
        self.label_projection = nn.Linear(1, label_dim)

        # Learnable label position embeddings
        self.label_pos_embeddings = nn.Parameter(
            torch.normal(0.0, math.sqrt(2.0 / label_dim), (num_labels, label_dim))
        )

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=label_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection to scalar per label position
        self.output_projection = nn.Linear(label_dim, 1)

    def _compute_per_label_score(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
    ) -> torch.Tensor:
        """Shared computation returning per-label scores (batch, num_samples, L)."""
        batch_size, num_samples, num_labels = y.shape
        y_flat = y.view(batch_size * num_samples, num_labels)
        label_tokens = self.label_projection(y_flat.unsqueeze(-1))
        label_tokens = label_tokens + self.label_pos_embeddings.unsqueeze(0)
        attn_output, _ = self.attention(label_tokens, label_tokens, label_tokens)
        per_label_score = self.output_projection(attn_output).squeeze(-1)
        return per_label_score.view(batch_size, num_samples, num_labels)

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        per_label_score = self._compute_per_label_score(y)
        # Cache for compute_vector_energy to reuse
        buffer["_attn_per_label_score"] = per_label_score

        if self.reduction == "sum":
            score = per_label_score.sum(dim=-1)
        elif self.reduction == "max":
            score = per_label_score.max(dim=-1)[0]
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        return score  # (batch, num_samples)

    def compute_vector_energy(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        """Returns per_label_score before reduction.

        Uses cached result from forward() if available, otherwise computes fresh.
        """
        if "_attn_per_label_score" in buffer:
            return buffer.pop("_attn_per_label_score")
        return self._compute_per_label_score(y)


@StructuredScore.register("multi-label-input-attention")
class InputConditionedAttentionGlobalScore(StructuredScore):
    """Attention global energy conditioned on input features x.

    Uses x to generate query bias, making the attention input-dependent:
    E(x, y) = attention(y + q(x)) reduced to scalar.
    """

    def __init__(
        self,
        num_labels: int,
        input_dim: int,
        label_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        reduction: str = "sum",
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_dim = label_dim
        self.reduction = reduction

        self.label_projection = nn.Linear(1, label_dim)
        self.label_pos_embeddings = nn.Parameter(
            torch.normal(0.0, math.sqrt(2.0 / label_dim), (num_labels, label_dim))
        )

        # Input-conditioned query bias
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, label_dim),
            nn.ReLU(),
        )
        self.query_bias_generator = nn.Linear(label_dim, num_labels * label_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=label_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.output_projection = nn.Linear(label_dim, 1)

    def _compute_per_label_score(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
    ) -> torch.Tensor:
        """Shared computation returning per-label scores (batch, num_samples, L)."""
        x = buffer["score_nn_x"]
        batch_size, num_samples, num_labels = y.shape

        h_x = self.input_encoder(x)
        query_bias = self.query_bias_generator(h_x).view(
            batch_size, num_labels, self.label_dim
        )

        y_flat = y.view(batch_size * num_samples, num_labels)
        label_tokens = self.label_projection(y_flat.unsqueeze(-1))
        label_tokens = label_tokens + self.label_pos_embeddings.unsqueeze(0)

        query_bias_expanded = query_bias.unsqueeze(1).expand(
            -1, num_samples, -1, -1
        ).reshape(batch_size * num_samples, num_labels, self.label_dim)
        label_tokens = label_tokens + query_bias_expanded

        attn_output, _ = self.attention(label_tokens, label_tokens, label_tokens)
        per_label_score = self.output_projection(attn_output).squeeze(-1)
        return per_label_score.view(batch_size, num_samples, num_labels)

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        per_label_score = self._compute_per_label_score(y, buffer)
        buffer["_input_attn_per_label_score"] = per_label_score

        if self.reduction == "sum":
            score = per_label_score.sum(dim=-1)
        elif self.reduction == "max":
            score = per_label_score.max(dim=-1)[0]
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        return score  # (batch, num_samples)

    def compute_vector_energy(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        """Returns per_label_score before reduction.

        Uses cached result from forward() if available, otherwise computes fresh.
        """
        if "_input_attn_per_label_score" in buffer:
            return buffer.pop("_input_attn_per_label_score")
        return self._compute_per_label_score(y, buffer)
