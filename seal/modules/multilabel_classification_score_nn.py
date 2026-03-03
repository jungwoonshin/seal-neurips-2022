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
    """ScoreNN variant for DTD that stashes task features in the buffer.

    Type B energy requires access to the task network's intermediate features
    (the output of feature_network before the label embedding projection).
    This ScoreNN variant extracts those features and stores them in buffer["task_features"]
    so the DTD global_score modules can access them without recomputation.
    """

    def compute_local_score(
        self,
        x: torch.Tensor,  #: (batch, features_size)
        y: torch.Tensor,  #: (batch, num_samples, num_labels)
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        # Extract features from the feature network
        features = self.task_nn.feature_network(x)  # (batch, hidden_dim)

        # Stash features in buffer for Type B energy
        buffer["task_features"] = features

        # Compute logits via label embeddings (same as standard ScoreNN)
        label_scores = torch.matmul(
            features, self.task_nn.label_embeddings.weight.T
        )  # (batch, num_labels)

        # Local energy: sum of logit * y per label
        local_energy = torch.sum(
            label_scores.unsqueeze(1) * y, dim=-1
        )  #: (batch, num_samples)

        return local_energy
