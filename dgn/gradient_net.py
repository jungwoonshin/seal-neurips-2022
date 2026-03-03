"""Direct Gradient Network (GradientNet).

G_Theta(x, y_pred, y_true) -> v in R^L

Takes the input features, the task-net's current predictions, and the ground
truth, and outputs an L-dimensional gradient correction vector. This vector
is injected directly into the task-net's backward pass.
"""

import torch
import torch.nn as nn


class GradientNet(nn.Module):
    """Auxiliary network that predicts the structural gradient direction.

    Architecture:
        concat(x, y_pred, y_true) -> MLP -> v in R^L

    The output v has the same shape as y_pred and represents the predicted
    direction of steepest ascent for the non-differentiable metric S.
    """

    def __init__(
        self,
        input_dim: int,
        label_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        """
        Args:
            input_dim: Dimension of input features x.
            label_dim: Number of labels L.
            hidden_dim: Hidden layer width.
            num_layers: Number of hidden layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.label_dim = label_dim

        # Input: concat(x, y_pred, y_true) -> dim = input_dim + 2 * label_dim
        in_dim = input_dim + 2 * label_dim

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, label_dim))

        self.net = nn.Sequential(*layers)

        # Initialize final layer near zero so initial gradient predictions
        # are small and don't destabilize training
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the gradient direction.

        Args:
            x: (batch, input_dim) input features.
            y_pred: (batch, L) task-net predictions.
            y_true: (batch, L) ground truth labels.

        Returns:
            v: (batch, L) predicted gradient direction.
        """
        inp = torch.cat([x, y_pred, y_true], dim=-1)
        return self.net(inp)
