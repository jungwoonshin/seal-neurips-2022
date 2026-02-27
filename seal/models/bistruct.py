"""BiStruct: Bidirectional Structured Prediction via Feature-Label Alignment.

A feedforward model that predicts labels from features (forward) while
simultaneously learning to reconstruct features from labels (backward).
The shared label embeddings encode output dependencies. The backward path
is discarded at inference.
"""

import logging
from typing import Dict, Any, Optional, Union

import torch
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.nn import InitializerApplicator, RegularizerApplicator

from seal.metrics import (
    MultilabelClassificationF1,
    MultilabelClassificationMeanAvgPrecision,
    MultilabelClassificationRelaxedF1,
)
from seal.modules.bistruct import (
    LabelSelfAttention,
    BackwardReconstructionModel,
    SymmetricAlignmentLoss,
)
from seal.modules.task_nn import TextEncoder

logger = logging.getLogger(__name__)


@Model.register("bistruct-multilabel-classification")
class BiStructMultilabelClassification(Model):
    """BiStruct model for multi-label classification.

    Forward path:
        x → Encoder → h → Classifier (h @ G^T) → σ(logits) = ŷ → LSA → ŷ_refined

    Backward path (training only):
        y → label_embed_lookup → mean_pool → MLP → ĥ

    Loss:
        L_total = L_task + α * L_align + β * L_refine
    """

    def __init__(
        self,
        vocab: Vocabulary,
        feature_network: Union[FeedForward, TextEncoder],
        label_embeddings: Embedding,
        num_labels: int,
        label_embed_dim: int = 64,
        lsa_attn_dim: int = 64,
        lsa_num_heads: int = 4,
        lsa_dropout: float = 0.1,
        backward_bottleneck_dim: int = 128,
        alpha: float = 0.1,
        beta: float = 1.0,
        regularizer: Optional[RegularizerApplicator] = None,
        initializer: Optional[InitializerApplicator] = None,
    ) -> None:
        super().__init__(vocab=vocab, regularizer=regularizer)

        self.feature_network = feature_network
        self.label_embeddings = label_embeddings
        self.num_labels = num_labels
        self.alpha = alpha
        self.beta = beta

        feature_dim = feature_network.get_output_dim()

        # Verify classifier label embeddings match feature dim
        assert label_embeddings.weight.shape[1] == feature_dim, (
            f"label_embeddings dim ({label_embeddings.weight.shape[1]}) "
            f"and feature_network output dim ({feature_dim}) must match."
        )

        # LSA module for refining predictions
        self.lsa = LabelSelfAttention(
            num_labels=num_labels,
            label_embed_dim=label_embed_dim,
            attn_dim=lsa_attn_dim,
            num_heads=lsa_num_heads,
            dropout=lsa_dropout,
        )

        # Backward reconstruction model (shares label embeddings from LSA)
        self.backward_model = BackwardReconstructionModel(
            label_embed_dim=label_embed_dim,
            feature_dim=feature_dim,
            bottleneck_dim=backward_bottleneck_dim,
        )

        # Symmetric alignment loss
        self.alignment_loss = SymmetricAlignmentLoss()

        # Metrics
        self.f1 = MultilabelClassificationF1()
        self.map = MultilabelClassificationMeanAvgPrecision()
        self.relaxed_f1 = MultilabelClassificationRelaxedF1()

        if initializer is not None:
            initializer(self)

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        meta: Optional[Dict] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        # === Forward path ===
        # Encode features: x → h
        h = self.feature_network(x)  # (batch, feature_dim)

        # Initial predictions: h @ G^T → logits → σ(logits) = ŷ
        logits = torch.matmul(h, self.label_embeddings.weight.T)  # (batch, num_labels)
        y_hat = torch.sigmoid(logits)  # (batch, num_labels)

        # Refined predictions via LSA: ŷ → LSA → ŷ_refined
        refined_logits = self.lsa(y_hat)  # (batch, num_labels)
        y_refined = torch.sigmoid(refined_logits)  # (batch, num_labels)

        results["y_pred"] = y_refined

        if labels is not None:
            # L_task: BCE on initial predictions
            loss_task = F.binary_cross_entropy_with_logits(logits, labels.float())

            # L_refine: BCE on refined predictions
            loss_refine = F.binary_cross_entropy_with_logits(
                refined_logits, labels.float()
            )

            # === Backward path (training only) ===
            # Reconstruct features from ground-truth labels
            h_hat = self.backward_model(
                labels.float(), self.lsa.label_embeddings
            )  # (batch, feature_dim)

            # L_align: symmetric alignment loss
            loss_align = self.alignment_loss(h, h_hat)

            # Combined loss
            loss = loss_task + self.alpha * loss_align + self.beta * loss_refine
            results["loss"] = loss

            # Log individual losses
            results["loss_task"] = loss_task.item()
            results["loss_align"] = loss_align.item()
            results["loss_refine"] = loss_refine.item()

            # Calculate metrics
            self._calculate_metrics(labels, y_refined)

        return results

    @torch.no_grad()
    def _calculate_metrics(
        self,
        labels: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> None:
        self.f1(y_hat, labels)
        self.map(y_hat, labels)
        self.relaxed_f1(y_hat, labels)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "fixed_f1": self.f1.get_metric(reset),
            "MAP": self.map.get_metric(reset),
            "relaxed_f1": self.relaxed_f1.get_metric(reset),
        }

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Threshold predictions at 0.5 for human-readable output."""
        if "y_pred" in output_dict:
            output_dict["predictions"] = (output_dict["y_pred"] >= 0.5).int()
        return output_dict
