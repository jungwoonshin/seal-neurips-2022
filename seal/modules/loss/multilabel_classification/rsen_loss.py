"""
RSEN Loss Functions for Multi-Label Classification.

Implements the RSEN task-net loss with dual gradient paths:
    L_F = λ_1 · E_Θ(x, R_Φ(x), F_Φ(x)) + λ_2 · ∑_j BCE(y_j, F_Φ(x)_j)

The gradient flows through two paths:
    1. Through ỹ (output path): standard SEAL gradient
    2. Through R_Φ(x) (representation path): directly shapes internal representations

Also provides the score-net (loss-net) side loss for NCE-based RSEN training.
"""

from typing import List, Tuple, Union, Dict, Any, Optional, Literal, cast
import torch
from seal.modules.loss import Loss
from seal.modules.loss.nce_loss import NCERankingLoss
from seal.modules.score_nn import ScoreNN
from seal.modules.logging import LoggedScalarScalar


def _normalize(y: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(y)


@Loss.register("rsen-score-loss")
class RSENScoreLoss(Loss):
    """Task-net loss that uses the RSEN energy as a learned loss.

    Computes L = -E_Θ(x, R_Φ(x), ỹ), where the energy function
    receives both the task-net's outputs and its internal representations.

    The key difference from standard score loss: the score_nn call
    populates buffer['representations'] via RSENMultilabelClassificationScoreNN,
    enabling the RSEN energy to evaluate representation-level coherence.
    The gradient flows through BOTH the output ỹ and representations R_Φ(x).
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.score_nn is None:
            raise ValueError("score_nn cannot be None for RSENScoreLoss")
        self.logging_buffer["rsen_score"] = LoggedScalarScalar()

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)

    def _forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # (batch, 1, num_labels)
        y_hat: torch.Tensor,  # (batch, 1, num_labels)
        y_hat_extra: Optional[torch.Tensor],
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.score_nn = cast(ScoreNN, self.score_nn)

        # score_nn.forward calls compute_local_score (which extracts representations
        # into buffer) and compute_global_score (which is the RSEN energy)
        predicted_score = self.score_nn(
            x, y_hat, buffer, **kwargs
        )  # (batch, num_samples)

        self.log("rsen_score", predicted_score.detach().mean().item())

        # Negate: we want to minimize this loss, and higher energy = better structure
        # So L = -E means minimizing L maximizes E
        return -predicted_score


@Loss.register("rsen-nce-ranking-with-discrete-sampling")
class RSENNCERankingLossWithDiscreteSamples(NCERankingLoss):
    """NCE ranking loss for training the RSEN score-net (loss-net).

    Identical to the standard NCE ranking loss, but uses the RSEN
    score_nn which populates representations in the buffer.
    """

    def __init__(
        self,
        sign: Literal["-", "+"] = "-",
        use_scorenn: bool = True,
        use_distance: bool = True,
        **kwargs: Any,
    ):
        super().__init__(use_scorenn=use_scorenn, **kwargs)
        self.sign = sign
        self.mul = -1 if sign == "-" else 1
        self.bce = torch.nn.BCELoss(reduction="none")
        self.use_distance = use_distance
        assert sign == "+" if not self.use_scorenn else True

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return _normalize(y)

    def sample(
        self,
        probs: torch.Tensor,  # (batch, 1, num_labels)
    ) -> torch.Tensor:  # (batch, num_samples, num_labels)
        assert probs.dim() == 3
        p = probs.squeeze(1)  # (batch, num_labels)
        samples = torch.transpose(
            torch.distributions.Bernoulli(probs=p).sample(
                [self.num_samples]
            ),
            0,
            1,
        )  # (batch, num_samples, num_labels)
        return samples

    def distance(
        self,
        samples: torch.Tensor,  # (batch, num_samples, num_labels)
        probs: torch.Tensor,  # (batch, num_samples, num_labels)
    ) -> torch.Tensor:  # (batch, num_samples)
        if not self.use_distance:
            return torch.zeros(
                [samples.shape[0], samples.shape[1]],
                dtype=torch.long,
                device=probs.device,
            )
        return self.mul * torch.sum(
            self.bce(probs, samples), dim=-1
        )  # (batch, num_samples)
