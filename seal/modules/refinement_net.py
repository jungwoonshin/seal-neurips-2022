from typing import Dict, Any
from allennlp.common.registrable import Registrable
import torch
import torch.nn as nn


class RefinementNet(torch.nn.Module, Registrable):
    """Base class for refinement networks that produce a delta update to y_current."""

    def forward(self, x_features, y_current, energy_field, buffer) -> torch.Tensor:
        """
        Args:
            x_features: Input features, shape (batch, feature_dim)
            y_current: Current label predictions, shape (batch, num_samples, num_labels)
            energy_field: Energy values, shape (batch, num_samples, num_labels)
            buffer: Additional context (implementation-specific)

        Returns:
            delta: Update to be added to y_current, shape (batch, num_samples, num_labels)
        """
        raise NotImplementedError


@RefinementNet.register("multi-label-feedforward-refinement")
class MultilabelFeedforwardRefinement(RefinementNet):
    def __init__(
        self,
        feature_dim: int,
        num_labels: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_labels = num_labels

        input_dim = feature_dim + num_labels + num_labels

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_labels))
        layers.append(nn.Tanh())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_features, y_current, energy_field, buffer) -> torch.Tensor:
        # x_features: (batch, feature_dim)
        # y_current: (batch, num_samples, num_labels)
        # energy_field: (batch, num_samples, num_labels)

        num_samples = y_current.shape[1]

        # Expand x_features to (batch, num_samples, feature_dim)
        x_expanded = x_features.unsqueeze(1).expand(-1, num_samples, -1)

        # Concatenate along last dim: (batch, num_samples, feature_dim + num_labels + num_labels)
        combined = torch.cat([x_expanded, y_current, energy_field], dim=-1)

        delta = self.mlp(combined)

        return delta


@RefinementNet.register("multi-label-gated-refinement")
class MultilabelGatedRefinement(RefinementNet):
    def __init__(
        self,
        feature_dim: int,
        num_labels: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_labels = num_labels

        input_dim = feature_dim + num_labels + num_labels

        # Gate path
        gate_layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            gate_layers.append(nn.Linear(in_dim, hidden_dim))
            gate_layers.append(nn.ReLU())
            gate_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        gate_layers.append(nn.Linear(hidden_dim, num_labels))
        gate_layers.append(nn.Sigmoid())

        self.gate_path = nn.Sequential(*gate_layers)

        # Delta path
        delta_layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            delta_layers.append(nn.Linear(in_dim, hidden_dim))
            delta_layers.append(nn.ReLU())
            delta_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        delta_layers.append(nn.Linear(hidden_dim, num_labels))
        delta_layers.append(nn.Tanh())

        self.delta_path = nn.Sequential(*delta_layers)

    def forward(self, x_features, y_current, energy_field, buffer) -> torch.Tensor:
        # x_features: (batch, feature_dim)
        # y_current: (batch, num_samples, num_labels)
        # energy_field: (batch, num_samples, num_labels)

        num_samples = y_current.shape[1]

        # Expand x_features to (batch, num_samples, feature_dim)
        x_expanded = x_features.unsqueeze(1).expand(-1, num_samples, -1)

        # Concatenate along last dim: (batch, num_samples, feature_dim + num_labels + num_labels)
        combined = torch.cat([x_expanded, y_current, energy_field], dim=-1)

        gate = self.gate_path(combined)
        delta = self.delta_path(combined)

        return gate * delta
