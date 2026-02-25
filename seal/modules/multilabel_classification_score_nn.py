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


@ScoreNN.register("multi-label-classification-dtd")
class MultilabelClassificationDTDScoreNN(ScoreNN):
    """ScoreNN variant for DTD that stashes intermediate task features in buffer.

    This allows the Type B energy (global_score) to access input-conditional
    features without recomputing the feature network.
    """

    def compute_local_score(
        self,
        x: torch.Tensor,  #: (batch, features_size)
        y: torch.Tensor,  #: (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        # Compute features and stash them for Type B energy
        features = self.task_nn.feature_network(x)  # (batch, hidden_dim)
        buffer["task_features"] = features

        # Compute logits manually (same as task_nn.forward but reuses features)
        logits = torch.matmul(
            features, self.task_nn.label_embeddings.weight.T
        )  # (batch, num_labels)

        local_energy = torch.sum(
            logits.unsqueeze(1) * y, dim=-1
        )  #: (batch, num_samples)

        return local_energy
