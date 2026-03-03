from typing import List, Tuple, Dict, Any, Optional, Iterator, overload
import torch
from seal.modules.sampler import Sampler
from seal.modules.sampler.inference_net import InferenceNetSampler
from seal.modules.score_nn import ScoreNN
from seal.modules.oracle_value_function import OracleValueFunction
from seal.modules.task_nn import TaskNN
from seal.modules.loss import Loss
from seal.modules.refinement_net import RefinementNet
from seal.common import ModelMode


@Sampler.register("multi-label-refinement")
@InferenceNetSampler.register("multi-label-refinement")
class MultilabelRefinementSampler(Sampler):
    """Iterative amortized refinement sampler for multi-label classification.

    Starting from an initial prediction produced by an inference network,
    this sampler iteratively refines the prediction using a refinement network
    that takes per-label energy feedback from the score network as input.

    At each refinement step k:
        1. Compute vector energy e_k from the score network for the current prediction y_k.
        2. Pass (x, y_k, e_k) through the refinement network to obtain a delta update.
        3. Update: y_{k+1} = clamp(y_k + delta, 0, 1).

    The training loss is a weighted sum of per-step losses, with linearly
    increasing weights by default so that later (more refined) predictions
    receive higher weight.
    """

    def __init__(
        self,
        inference_nn: TaskNN,
        refinement_net: RefinementNet,
        loss_fn: Loss,
        score_nn: ScoreNN,
        num_refinement_steps: int = 2,
        step_weights: Optional[List[float]] = None,
        oracle_value_function: Optional[OracleValueFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            score_nn=score_nn,
            oracle_value_function=oracle_value_function,
            **kwargs,
        )
        self.inference_nn = inference_nn
        self.refinement_net = refinement_net
        self.loss_fn = loss_fn
        self.num_refinement_steps = num_refinement_steps

        # Compute default step weights: linearly increasing
        # There are K+1 weights total (one for initial prediction + K refinement steps)
        if step_weights is not None:
            assert len(step_weights) == num_refinement_steps + 1, (
                f"step_weights must have length num_refinement_steps + 1 "
                f"({num_refinement_steps + 1}), got {len(step_weights)}"
            )
            self.step_weights = step_weights
        else:
            # Linearly increasing: w_k = (k+1) / sum(1..K+1)
            total_steps = num_refinement_steps + 1
            denom = sum(k + 1 for k in range(total_steps))
            self.step_weights = [(k + 1) / denom for k in range(total_steps)]

        self.logging_children.append(self.loss_fn)

    @property
    def is_normalized(self) -> bool:
        """Output is always in [0, 1] due to clamping."""
        return True

    @overload
    def normalize(self, y: None) -> None:
        ...

    @overload
    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        ...

    def normalize(self, y: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        # Output is already normalized (clamped to [0, 1]), so identity.
        return y

    def parameters_with_model_mode(
        self, mode: ModelMode
    ) -> Iterator[torch.nn.Parameter]:
        yield from self.inference_nn.parameters()
        yield from self.refinement_net.parameters()

    @property
    def different_training_and_eval(self) -> bool:
        return False

    def forward(
        self,
        x: Any,
        labels: Optional[torch.Tensor],  # shape (batch, ...)
        buffer: Dict,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, input_dim) for multi-label classification.
            labels: Ground truth labels of shape (batch, num_labels), or None at test time.
            buffer: Dictionary with additional context.

        Returns:
            y_final: Final refined predictions, shape (batch, 1, num_labels), values in [0, 1].
            y_final: Same as first return (no separate cost-augmented output).
            loss: Weighted sum of per-step losses, or None if labels is None.
        """
        # Step 1: Initial prediction from inference network
        y_0 = torch.sigmoid(
            self.inference_nn(x, buffer)
        ).unsqueeze(1)  # (batch, 1, num_labels)

        all_predictions = [y_0]
        y_k = y_0

        # Steps 2-3: Iterative refinement
        for k in range(self.num_refinement_steps):
            # 3a: Get vector energy from score network
            e_k = self.score_nn.compute_vector_energy(x, y_k, buffer)

            if e_k is None:
                # 3b: Fall back to autograd-based gradient computation
                y_k_ag = y_k.detach().requires_grad_(True)
                scalar = self.score_nn(x, y_k_ag, buffer)
                e_k = torch.autograd.grad(
                    scalar.sum(), y_k_ag, create_graph=True
                )[0]
            else:
                # 3c: Detach to prevent task_nn gradients flowing through score_nn
                e_k = e_k.detach()

            # 3d: Get delta from refinement network
            # x is the input features tensor for MLC
            delta = self.refinement_net(x, y_k, e_k, buffer)

            # 3e: Update and clamp
            y_k = torch.clamp(y_k + delta, 0, 1)

            # 3f: Collect predictions
            all_predictions.append(y_k)

        # Step 5: Compute loss (only during training when labels are available)
        if labels is not None:
            total_loss = torch.tensor(
                0.0, device=y_k.device, dtype=y_k.dtype
            )
            labels_unsqueezed = labels.unsqueeze(1)  # (batch, 1, num_labels)

            for step_idx, w_k in enumerate(self.step_weights):
                y_step = all_predictions[step_idx]
                step_loss = self.loss_fn(
                    x, labels_unsqueezed, y_step, y_step, buffer
                )
                total_loss = total_loss + w_k * step_loss
        else:
            total_loss = None

        # Step 6: Return final prediction
        y_final = all_predictions[-1]  # (batch, 1, num_labels)

        return y_final, y_final, total_loss
