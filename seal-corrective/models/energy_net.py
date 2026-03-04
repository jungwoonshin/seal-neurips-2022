import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyNet(nn.Module):
    """
    Energy network with local and global terms.

    E(x, y) = E_local(x, y) + E_global(y)

    Local:  E_local(x, y) = sum_i  y_i * b_i^T h(x)
    Global: E_global(y)    = v^T softplus(M y)
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int,
                 energy_hidden: int, dropout: float = 0.2):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.B = nn.Linear(hidden_dim, num_labels)  # local scoring
        self.M = nn.Linear(num_labels, energy_hidden, bias=False)  # global label mixing
        self.v = nn.Parameter(torch.randn(energy_hidden))  # global scoring vector

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.feature_net(x)
        e_local = (y * self.B(h)).sum(dim=-1)
        e_global = self.energy_global(y)
        return e_local + e_global

    def energy_global(self, y: torch.Tensor) -> torch.Tensor:
        return (self.v * F.softplus(self.M(y))).sum(dim=-1)
