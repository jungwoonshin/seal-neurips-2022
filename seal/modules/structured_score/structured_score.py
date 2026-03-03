from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.common.registrable import Registrable
import torch


class StructuredScore(torch.nn.Module, Registrable):
    """Base class for all structured energy terms like linear-chain,
    skip-chain and other higher order energies.

    Inheriting classes should override the `foward` method.
    """

    def forward(
        self,
        y: torch.Tensor,
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            y: Tensor of shape (batch, num_samples or 1, ...)

        Returns:
            scores of shape (batch, num_samples or 1)
        """
        raise NotImplementedError

    def compute_vector_energy(
        self,
        y: torch.Tensor,
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        """Returns per-label energy vector of shape (batch, num_samples, L).

        Default: distributes scalar energy uniformly across labels.
        Subclasses should override for natural decompositions.

        Invariant: compute_vector_energy(y, buffer).sum(dim=-1) == forward(y, buffer)
        """
        scalar = self.forward(y, buffer, **kwargs)  # (batch, num_samples)
        num_labels = y.shape[-1]
        return scalar.unsqueeze(-1) / num_labels  # (batch, num_samples, L)


@StructuredScore.register("structured-score-container")
class StructuredScoreContainer(StructuredScore):
    """A collection of different `StructuredScore` modules
    that will be added together to form the total energy"""

    def __init__(self, constituent_energies: List[StructuredScore]) -> None:
        super().__init__()
        self.constituent_energies = torch.nn.ModuleList(constituent_energies)
        assert len(self.constituent_energies) > 0

    def forward(
        self,
        y: torch.Tensor,
        buffer: Dict,
        **kwargs: Any,
    ) -> torch.Tensor:
        total_energy: torch.Tensor = self.constituent_energies[0](
            y, buffer, **kwargs
        )

        for energy in self.constituent_energies[1:]:
            total_energy = total_energy + energy(y, buffer, **kwargs)

        return total_energy

    def compute_vector_energy(
        self,
        y: torch.Tensor,
        buffer: Dict,
        **kwargs: Any,
    ) -> Optional[torch.Tensor]:
        """Sums constituent vector energies."""
        total: Optional[torch.Tensor] = None
        for energy in self.constituent_energies:
            vec = energy.compute_vector_energy(y, buffer, **kwargs)
            if vec is not None:
                total = vec if total is None else total + vec
        return total
