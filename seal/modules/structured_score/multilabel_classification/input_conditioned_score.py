"""Input-conditioned global energy: v(x)^T sigma(M(x) * y).

The transformation matrix M and projection vector v are both conditioned
on the input features x, making the global energy input-dependent.
"""

from typing import Dict, Any, Optional
from seal.modules.structured_score import StructuredScore
from allennlp.modules.feedforward import FeedForward
import torch
import torch.nn as nn


@StructuredScore.register("multi-label-input-conditioned")
class InputConditionedGlobalScore(StructuredScore):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int,
        feature_encoder: FeedForward,
        activation: str = "softplus",
    ):
        """
        Args:
            input_dim: Dimension of raw input features x.
            num_labels: Number of labels (L).
            hidden_dim: Dimension of the intermediate space.
            feature_encoder: Encodes raw features x to a representation h(x).
            activation: Activation function for the label transform.
        """
        super().__init__()
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

        # Encode x -> h(x)
        self.feature_encoder = feature_encoder
        enc_dim = self.feature_encoder.get_output_dim()

        # Generate M(x): h(x) -> (hidden_dim * num_labels)
        self.M_generator = nn.Linear(enc_dim, hidden_dim * num_labels)

        # Generate v(x): h(x) -> (hidden_dim,)
        self.v_generator = nn.Linear(enc_dim, hidden_dim)

        if activation == "softplus":
            self.activation = nn.Softplus()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        x = buffer["score_nn_x"]  # (batch, input_dim)
        batch_size = x.shape[0]
        num_samples = y.shape[1]

        # Encode input features
        h_x = self.feature_encoder(x)  # (batch, enc_dim)

        # Generate M(x) and v(x)
        M = self.M_generator(h_x).view(
            batch_size, self.hidden_dim, self.num_labels
        )  # (batch, hidden_dim, num_labels)
        v = self.v_generator(h_x)  # (batch, hidden_dim)

        # Compute M(x) * y for each sample: (batch, hidden_dim, L) x (batch, num_samples, L)^T
        # -> (batch, num_samples, hidden_dim)
        My = torch.bmm(
            M, y.transpose(1, 2)
        ).transpose(1, 2)  # (batch, num_samples, hidden_dim)

        # Apply activation
        activated = self.activation(My)  # (batch, num_samples, hidden_dim)

        # Dot product with v(x): (batch, num_samples, hidden_dim) * (batch, 1, hidden_dim)
        score = torch.sum(
            activated * v.unsqueeze(1), dim=-1
        )  # (batch, num_samples)

        return score

    def compute_vector_energy(
        self,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        """Default uniform fallback — no natural per-label decomposition
        in hidden_dim space. Distributes scalar uniformly."""
        scalar = self.forward(y, buffer, **kwargs)  # (batch, num_samples)
        num_labels = y.shape[-1]
        return scalar.unsqueeze(-1) / num_labels  # (batch, num_samples, L)
