from typing import List, Tuple, Union, Dict, Any, Optional
from .score_nn import ScoreNN
import torch


@ScoreNN.register("multi-label-classification")
class MultilabelClassificationScoreNN(ScoreNN):
    def compute_local_score(
        self,
        x: torch.Tensor,  #: (batch, features_size)
        y: torch.Tensor,  #: (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        label_scores = self.task_nn(
            x, buffer
        )  # unormalized logit of shape (batch, num_labels)
        local_energy = torch.sum(
            label_scores.unsqueeze(1) * y, dim=-1
        )  #: (batch, num_samples)

        return local_energy


@ScoreNN.register("multi-label-classification-rsen")
class RSENMultilabelClassificationScoreNN(ScoreNN):
    """ScoreNN for RSEN that extracts per-label representations from the task-net
    and passes them to the energy function via the buffer.

    The energy function E_Θ(x, R_Φ(x), ỹ) operates on both the output space
    and the representation space, enabling the loss-net to evaluate and shape
    the structural coherence of the task-net's internal representations.
    """

    def compute_local_score(
        self,
        x: torch.Tensor,  #: (batch, features_size)
        y: torch.Tensor,  #: (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        # Use forward_with_representations to get both logits and representations
        logits, representations = self.task_nn.forward_with_representations(
            x, buffer
        )
        # logits: (batch, num_labels)
        # representations: (batch, num_labels, hidden_dim)

        # Store representations in buffer for the global_score (RSEN energy) to use
        buffer["representations"] = representations

        # Standard local energy: logits · y
        local_energy = torch.sum(
            logits.unsqueeze(1) * y, dim=-1
        )  #: (batch, num_samples)

        return local_energy
