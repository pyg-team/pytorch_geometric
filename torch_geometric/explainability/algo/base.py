from abc import abstractmethod
from typing import Callable, Optional

import torch

from torch_geometric.data import Data
from torch_geometric.explainability.explanations import Explanation


class ExplainerAlgorithm(torch.nn.Module):
    r"""An abstract class for explainer algorithms."""
    def __init__(self) -> None:
        super().__init__()

        self.objective: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor]
        r"""
        This method compute the loss to be used for the explanation algorithm.
        Subclasses should override this method to define their own loss.

        Args:
            exp_output (torch.Tensor): the output of the explanation algorithm.
                (e.g. the forward pass of the model with the mask applied).
            gnn_output (torch.Tensor): the original output of the GNN.
            target (torch.Tensor): the target of the GNN.
            target_index (int): the index of the target to explain.
        """

    def set_objective(
        self,
        objective: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int],
                            torch.Tensor],
    ) -> None:
        """Sets the loss function to be used for the explanation algorithm.

        Args:
            objective (Callable): loss function.
        """
        self.objective = objective

    @abstractmethod
    def explain(
        self,
        g: Data,
        model: torch.nn.Module,
        target: torch.Tensor,
        target_index: Optional[int] = None,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Explanation:
        """This method computes the explanation of the GNN.

        Args:
            g (Data): input graph.
            model (torch.nn.Module): model to use in explanations.
            target (torch.Tensor): target of the GNN.
            target_index (int, optional): Index of the target to explain. Used
                in case of multi-outputs. Defaults to None.
            batch (torch.Tensor, optional): _description_. Defaults to None.

        Returns:
            Explanation: explanation of the GNN.
        """

    def forward(
        self,
        g: Data,
        model: torch.nn.Module,
        target: torch.Tensor,
        target_index: Optional[int] = None,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Explanation:
        return self.explain(g=g, model=model, target=target,
                            target_index=target_index, batch=batch, **kwargs)

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
                Should be in ["model", "phenomenon"]
            mask_type (str): the type of mask to use.
                Should be in ["node", "edge", "node_and_edge", "layers"]
        """
