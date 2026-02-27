from typing import Dict, Any
from seal.modules.structured_score import StructuredScore
import torch
import math


@StructuredScore.register("multi-label-transformer")
class MultilabelClassificationTransformerStructuredScore(StructuredScore):
    """Transformer-based global structured score for multi-label classification.

    Treats each label as a "position" in a sequence. Self-attention captures
    label-label dependencies (co-occurrence, exclusion, hierarchy).

    Input:  y of shape (batch, num_samples, num_labels)
    Output: score of shape (batch, num_samples)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Project each label's scalar value to d_model
        self.input_projection = torch.nn.Linear(1, d_model)

        # Learnable positional encoding for label positions
        self.label_position_embedding = torch.nn.Parameter(
            torch.randn(input_dim, d_model) * 0.02
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Score head: pool -> scalar
        hidden_dim = d_model
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch, num_samples, num_labels = y.shape

        # Reshape to process all samples together
        # (batch * num_samples, num_labels)
        y_flat = y.view(batch * num_samples, num_labels)

        # Each label becomes a "token": (batch*num_samples, num_labels, 1)
        y_tokens = y_flat.unsqueeze(-1)

        # Project to d_model and add positional encoding
        z = self.input_projection(y_tokens)  # (B*S, num_labels, d_model)
        z = z + self.label_position_embedding.unsqueeze(0)

        # Transformer encoder
        z = self.transformer(z)  # (B*S, num_labels, d_model)

        # Mean pool over labels
        z_pooled = z.mean(dim=1)  # (B*S, d_model)

        # Score
        score = self.score_head(z_pooled).squeeze(-1)  # (B*S,)

        return score.view(batch, num_samples)
