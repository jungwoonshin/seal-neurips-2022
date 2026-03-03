from typing import List, Tuple, Union, Dict, Any, Optional
from .score_nn import ScoreNN
import torch


@ScoreNN.register("multi-label-classification")
class MultilabelClassificationScoreNN(ScoreNN):
    def _get_label_scores(
        self,
        x: torch.Tensor,
        buffer: Dict,
    ) -> torch.Tensor:
        """Get label scores from task_nn with caching in buffer."""
        buffer["score_nn_x"] = x
        label_scores = self.task_nn(
            x, buffer
        )  # unormalized logit of shape (batch, num_labels)
        return label_scores

    def compute_local_score(
        self,
        x: torch.Tensor,  #: (batch, features_size)
        y: torch.Tensor,  #: (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        label_scores = self._get_label_scores(x, buffer)
        local_energy = torch.sum(
            label_scores.unsqueeze(1) * y, dim=-1
        )  #: (batch, num_samples)

        return local_energy

    def compute_local_vector_energy(
        self,
        x: torch.Tensor,  #: (batch, features_size)
        y: torch.Tensor,  #: (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        """Per-label local energy before sum: label_scores * y."""
        label_scores = self._get_label_scores(x, buffer)
        return label_scores.unsqueeze(1) * y  # (batch, num_samples, L)
