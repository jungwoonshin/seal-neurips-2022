"""
Model components for G-SEAL validation.

Contains:
  - TaskNet: Simple MLP for per-position tag prediction
  - StructureNetwork: Small transformer that scores output structures
  - OracleStructureLoss: Hand-crafted penalty for structural violations (upper bound)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1)]


class TaskNet(nn.Module):
    """Simple feedforward task network for per-position tag prediction.

    Architecture: Linear -> ReLU -> Linear -> per-position logits
    No recurrence — cannot learn transition constraints from architecture alone.
    """

    def __init__(
        self, input_dim: int = 16, hidden_dim: int = 64, num_tags: int = 4
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.projection = nn.Linear(hidden_dim, num_tags)
        self.num_tags = num_tags

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            logits: (batch, seq_len, num_tags)
            embeddings: (batch, seq_len, hidden_dim) — pre-projection features
        """
        h = self.encoder(x)  # (batch, seq_len, hidden_dim)
        logits = self.projection(h)  # (batch, seq_len, num_tags)
        return logits, h

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return hard predictions (argmax)."""
        logits, _ = self.forward(x)
        return logits.argmax(dim=-1)

    def predict_soft(self, x: torch.Tensor) -> torch.Tensor:
        """Return soft predictions (softmax probabilities)."""
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=-1)


class StructureNetwork(nn.Module):
    """Transformer-based structure scorer.

    Takes a soft output matrix Y (and optionally embeddings H) and produces
    a scalar score indicating how "structurally valid" the output is.

    Architecture:
        Input: Y ∈ R^{seq_len × num_tags} [optionally concatenated with H]
        -> Linear projection to d_model
        -> Positional encoding
        -> TransformerEncoder (num_layers layers, num_heads heads)
        -> Mean pool over sequence
        -> MLP -> scalar score
    """

    def __init__(
        self,
        num_tags: int = 4,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        use_embeddings: bool = False,
        embedding_dim: int = 64,
    ):
        super().__init__()
        self.use_embeddings = use_embeddings

        input_dim = num_tags + (embedding_dim if use_embeddings else 0)
        self.input_projection = nn.Linear(input_dim, d_model)

        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        y_soft: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_soft: (batch, seq_len, num_tags) soft/one-hot tag representations
            embeddings: (batch, seq_len, embedding_dim) optional task-net embeddings

        Returns:
            scores: (batch,) scalar structure scores
        """
        if self.use_embeddings and embeddings is not None:
            z = torch.cat([y_soft, embeddings], dim=-1)
        else:
            z = y_soft

        z = self.input_projection(z)  # (batch, seq_len, d_model)
        z = self.positional_encoding(z)
        z = self.transformer(z)  # (batch, seq_len, d_model)

        # Mean pool over sequence
        z_pooled = z.mean(dim=1)  # (batch, d_model)
        score = self.score_head(z_pooled).squeeze(-1)  # (batch,)

        return score


class OracleStructureLoss(nn.Module):
    """Hand-crafted penalty for structural violations.

    This serves as the upper bound — a perfect structure loss that directly
    penalizes invalid transitions. G-SEAL should approach this performance.

    For differentiability, operates on soft outputs (probabilities) rather
    than hard predictions. Penalizes P(tag_t = A) * P(tag_{t+1} != B) etc.
    """

    FORCED_NEXT = {0: 1, 2: 3}  # A->B, C->D

    def __init__(self, penalty_weight: float = 1.0):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, y_soft: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_soft: (batch, seq_len, num_tags) soft probabilities

        Returns:
            penalty: (batch,) non-negative penalty for violations
        """
        batch, seq_len, num_tags = y_soft.shape
        total_penalty = torch.zeros(batch, device=y_soft.device)

        for t in range(seq_len - 1):
            for src_tag, required_next in self.FORCED_NEXT.items():
                # P(tag_t = src_tag) * P(tag_{t+1} != required_next)
                p_src = y_soft[:, t, src_tag]
                p_wrong_next = 1.0 - y_soft[:, t + 1, required_next]
                total_penalty = total_penalty + p_src * p_wrong_next

        return self.penalty_weight * total_penalty


def compute_gradient_norms(
    model: nn.Module, loss: torch.Tensor
) -> Dict[str, float]:
    """Compute per-layer gradient norms for diagnostics.

    Args:
        model: the model whose parameters to inspect
        loss: loss tensor (must have been backward()'d or use retain_graph)

    Returns:
        dict mapping layer names to gradient L2 norms
    """
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms
