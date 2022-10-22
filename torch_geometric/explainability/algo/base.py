from abc import abstractmethod
from typing import Callable, Optional, Tuple

import torch

from torch_geometric.explainability.explanations import Explanation


class ExplainerAlgorithm(torch.nn.Module):
    r"""An abstract class for explainer algorithms."""
    def __init__(self) -> None:
        super().__init__()

        # TODO: check if this is good practice, took the idea from captum
        self.objective: Callable[
            [Tuple[torch.Tensor,
                   ...], torch.Tensor, torch.Tensor, int], torch.Tensor]
        r"""
        This method compute the loss to be used for the explanation algorithm.
        Subclasses should override this method to define their own loss.

        Args:
            inputs (Tuple[torch.Tensor, ...]): the inputs of the GNN.
            output (torch.Tensor): the output of the GNN.
            target (torch.Tensor): the target of the GNN.
            target_index (int): the index of the target to explain.
        """

    def set_objective(
        self,
        objective: Callable[
            [Tuple[torch.Tensor,
                   ...], torch.Tensor, torch.Tensor, int], torch.Tensor],
    ) -> None:
        """Sets the loss function to be used for the explanation algorithm.

        Args:
            objective (Callable): loss function.
        """
        self.objective = objective

    @abstractmethod
    def explain(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target: torch.Tensor,
        model: torch.nn.Module,
        target_index: Optional[int] = None,
        **kwargs,
    ) -> Explanation:
        r"""This method computes the explanation of the GNN.

        Args:
            inputs (Tuple[torch.Tensor, ...]): the inputs of the GNN.
            target (torch.Tensor): the target of the GNN.
            model (torch.nn.Module): the GNN to explain.
            name_inputs (List[str]): the name of the inputs of the GNN.
                (used to create the explanation object)
        """

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target: torch.Tensor,
        model: torch.nn.Module,
        target_index: Optional[int] = None,
        **kwargs,
    ) -> Explanation:
        return self.explain(x, edge_index, target, model, target_index,
                            **kwargs)

    @abstractmethod
    def supports(
        self,
        explanation_type: str,
        mask_type: str,
    ) -> bool:
        r"""Check if the explainer supports the user-defined settings.

        Returns true if the explainer supports the settings.

        Responsability of the children to exclude the settings that are not
        supported.

        Args:
            explanation_type (str): the type of explanation to compute.
            mask_type (str): the type of mask to use.
        """
        return True
