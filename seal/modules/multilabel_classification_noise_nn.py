from typing import Dict, Any, Optional
from seal.modules.noise_nn import NoiseNN
import torch
import torch.nn as nn


@NoiseNN.register("multi-label-autoregressive")
class AutoregressiveNoiseNN(NoiseNN):
    """LSTM-based autoregressive noise network for multi-label classification.

    Models P_N(y|x) = prod_i P(y_i | y_{<i}, x; psi) as a sequence of
    Bernoulli decisions, one per label, conditioned on previous labels and input.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

        # Encode input x to a context vector
        self.input_encoder = nn.Linear(input_dim, hidden_dim)

        # LSTM: at each step, input is [context, y_{i-1}] -> predict y_i
        # Input to LSTM: hidden_dim (context) + 1 (previous label)
        self.lstm = nn.LSTM(
            input_size=hidden_dim + 1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection: hidden -> logit for Bernoulli
        self.output_proj = nn.Linear(hidden_dim, 1)

    def _encode_input(self, x: Any) -> torch.Tensor:
        """Encode input features to context vector."""
        if isinstance(x, dict):
            # Text input — use score_nn_x from buffer if available
            raise ValueError(
                "AutoregressiveNoiseNN expects tensor input. "
                "Use buffer['score_nn_x'] for text inputs."
            )
        return self.input_encoder(x)  # (batch, hidden_dim)

    def log_prob(
        self,
        x: Any,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Teacher-forced log probability computation.

        Returns:
            log_prob: (batch, num_samples)
        """
        # Use pre-encoded input if available
        if "noise_nn_x" in buffer:
            context = buffer["noise_nn_x"]
        else:
            context = self._encode_input(x)  # (batch, hidden_dim)
            buffer["noise_nn_x"] = context

        batch_size, num_samples, num_labels = y.shape

        # Expand context for all samples
        context_expanded = context.unsqueeze(1).expand(
            -1, num_samples, -1
        )  # (batch, num_samples, hidden_dim)

        # Flatten batch and samples
        context_flat = context_expanded.reshape(
            batch_size * num_samples, self.hidden_dim
        )  # (B*S, hidden_dim)
        y_flat = y.reshape(
            batch_size * num_samples, num_labels
        ).float()  # (B*S, num_labels) — ensure float for BCE

        # Build LSTM input: at step i, input is [context, y_{i-1}]
        # y_{-1} = 0 (start token)
        y_shifted = torch.cat(
            [torch.zeros_like(y_flat[:, :1]), y_flat[:, :-1]], dim=1
        )  # (B*S, num_labels)

        # context repeated for each label step
        context_seq = context_flat.unsqueeze(1).expand(
            -1, num_labels, -1
        )  # (B*S, num_labels, hidden_dim)
        lstm_input = torch.cat(
            [context_seq, y_shifted.unsqueeze(-1)], dim=-1
        )  # (B*S, num_labels, hidden_dim+1)

        lstm_out, _ = self.lstm(lstm_input)  # (B*S, num_labels, hidden_dim)
        logits = self.output_proj(lstm_out).squeeze(-1)  # (B*S, num_labels)

        # Per-label log Bernoulli probability
        log_probs_per_label = -nn.functional.binary_cross_entropy_with_logits(
            logits, y_flat, reduction="none"
        )  # (B*S, num_labels)

        # Sum over labels
        log_prob = log_probs_per_label.sum(dim=-1)  # (B*S,)
        return log_prob.view(batch_size, num_samples)

    def sample(
        self,
        x: Any,
        num_samples: int,
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Sequential autoregressive sampling.

        Returns:
            samples: (batch, num_samples, num_labels)
        """
        if "noise_nn_x" in buffer:
            context = buffer["noise_nn_x"]
        else:
            context = self._encode_input(x)
            buffer["noise_nn_x"] = context

        batch_size = context.shape[0]

        # Expand context for all samples
        context_expanded = context.unsqueeze(1).expand(
            -1, num_samples, -1
        ).reshape(batch_size * num_samples, self.hidden_dim)  # (B*S, hidden_dim)

        samples = []
        prev_label = torch.zeros(
            batch_size * num_samples, 1, device=context.device
        )  # (B*S, 1) — start token

        h = None  # LSTM hidden state
        for i in range(self.num_labels):
            lstm_input = torch.cat(
                [context_expanded.unsqueeze(1), prev_label.unsqueeze(1)], dim=-1
            )  # (B*S, 1, hidden_dim+1)
            lstm_out, h = self.lstm(lstm_input, h)  # (B*S, 1, hidden_dim)
            logit = self.output_proj(lstm_out.squeeze(1))  # (B*S, 1)
            prob = torch.sigmoid(logit)  # (B*S, 1)
            sample = torch.bernoulli(prob)  # (B*S, 1)
            samples.append(sample)
            prev_label = sample

        samples = torch.cat(samples, dim=-1)  # (B*S, num_labels)
        return samples.view(batch_size, num_samples, self.num_labels)


@NoiseNN.register("multi-label-factored-bernoulli")
class FactoredBernoulliNoiseNN(NoiseNN):
    """Wraps existing factored Bernoulli noise into the NoiseNN interface.

    Uses the inference network's output as per-label probabilities
    for independent Bernoulli sampling. This provides backward
    compatibility with the original NCE sampling strategy.
    """

    def __init__(
        self,
        num_labels: int,
    ):
        super().__init__()
        self.num_labels = num_labels

    def log_prob(
        self,
        x: Any,
        y: torch.Tensor,  # (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Log probability under factored Bernoulli using inference_nn probs from buffer."""
        probs = buffer.get("noise_probs")  # (batch, num_labels)
        if probs is None:
            raise RuntimeError(
                "FactoredBernoulliNoiseNN requires 'noise_probs' in buffer. "
                "Make sure the inference network deposits its probabilities."
            )
        # probs: (batch, num_labels) -> expand to (batch, num_samples, num_labels)
        y_float = y.float()
        probs_expanded = probs.unsqueeze(1).expand_as(y_float)
        # Per-label log Bernoulli
        log_probs_per_label = (
            y_float * torch.log(probs_expanded + 1e-8)
            + (1 - y_float) * torch.log(1 - probs_expanded + 1e-8)
        )  # (batch, num_samples, num_labels)
        return log_probs_per_label.sum(dim=-1)  # (batch, num_samples)

    def sample(
        self,
        x: Any,
        num_samples: int,
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Independent Bernoulli sampling from inference_nn probs in buffer."""
        probs = buffer.get("noise_probs")  # (batch, num_labels)
        if probs is None:
            raise RuntimeError(
                "FactoredBernoulliNoiseNN requires 'noise_probs' in buffer."
            )
        # (num_samples, batch, num_labels) -> (batch, num_samples, num_labels)
        samples = torch.transpose(
            torch.distributions.Bernoulli(probs=probs).sample(
                [num_samples]
            ),
            0,
            1,
        )
        return samples
