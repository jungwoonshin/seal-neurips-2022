from typing import Dict, Any, Optional
from seal.modules.loss import Loss
from seal.modules.noise_nn import NoiseNN
import torch


@Loss.register("multi-label-structured-nce-ranking")
class MultiLabelStructuredNCERankingLoss(Loss):
    """Structured NCE ranking loss for multi-label classification.

    Uses a NoiseNN to generate noise samples and their log probabilities,
    rather than relying on an inference network for sampling. The loss
    ranks the true label above noise samples using cross-entropy over
    the combined score (score_nn + log_noise).
    """

    def __init__(
        self,
        noise_nn: NoiseNN,
        num_samples: int = 10,
        use_scorenn: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.noise_nn = noise_nn
        self.num_samples = num_samples
        self.use_scorenn = use_scorenn
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, L)
        y_hat: torch.Tensor,  # (batch, num_samples, ...)
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert labels is not None

        # Draw noise samples from the noise distribution
        noise_samples = self.noise_nn.sample(
            x, self.num_samples, buffer
        )  # (batch, K, L)

        # Concatenate true labels and noise samples
        y = torch.cat(
            [labels.to(dtype=noise_samples.dtype), noise_samples], dim=1
        )  # (batch, 1+K, L)

        # Compute log probability under the noise distribution
        log_noise = self.noise_nn.log_prob(x, y, buffer)  # (batch, 1+K)

        # Compute score from score_nn if enabled
        if self.use_scorenn:
            score = self.score_nn(x, y, buffer)  # (batch, 1+K)
        else:
            score = 0

        # Combined score: score_nn + log noise probability
        new_score = score + log_noise  # (batch, 1+K)

        # Ranking loss: true label (index 0) should have the highest score
        ranking_loss = self.cross_entropy(
            new_score,
            torch.zeros(
                new_score.shape[0], dtype=torch.long, device=new_score.device
            ),  # (batch,)
        )

        return ranking_loss  # (batch,)
