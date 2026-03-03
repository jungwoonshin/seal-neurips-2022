from typing import Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch


class NoiseNN(torch.nn.Module, Registrable):
    """Base class for noise networks used in structured NCE.

    A noise network defines a distribution P_N(y|x) from which
    noise samples are drawn during NCE training.
    """

    def sample(
        self,
        x: Any,
        num_samples: int,
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Draw samples from the noise distribution.

        Args:
            x: Input features.
            num_samples: Number of samples to draw.
            buffer: Shared computation buffer.

        Returns:
            Tensor of shape (batch, num_samples, ...) with noise samples.
        """
        raise NotImplementedError

    def log_prob(
        self,
        x: Any,
        y: torch.Tensor,
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute log probability of y under the noise distribution.

        Args:
            x: Input features.
            y: Tensor of shape (batch, num_samples, ...).
            buffer: Shared computation buffer.

        Returns:
            Tensor of shape (batch, num_samples) with log probabilities.
        """
        raise NotImplementedError
