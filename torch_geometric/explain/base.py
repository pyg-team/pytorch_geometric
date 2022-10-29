from abc import abstractmethod
from typing import Optional

import torch

from torch_geometric.data import Data
from torch_geometric.explain.explanations import Explanation


class ExplainerAlgorithm(torch.nn.Module):
    """Abstract class for explanation algorithms."""
    @abstractmethod
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        This method compute the loss to be used for the explanation algorithm.
        Subclasses should override this method to define their own loss.

        Args:
            y_hat (torch.Tensor): the output of the explanation algorithm.
                (e.g. the forward pass of the model with the mask applied).
            y (torch.Tensor): the reference output.
        """

    @torch.no_grad()
    def get_initial_prediction(
        self,
        model: torch.nn.Module,
        g: Data,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Return the initial prediction of the model.

        Args:
            model (torch.nn.Module): the model to explain.
            g (torch_geometric.data.Data): the input graph.
            batch (torch.Tensor, optional): batch indicator. Defaults to None.

        Returns:
            torch.Tensor: output of the underlying model.
        """
        out = model(
            g,
            **dict(batch=batch, **kwargs),
        )
        return out

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        model: torch.nn.Module,
        target: torch.Tensor,
        target_index: int = 0,
        batch: Optional[torch.Tensor] = None,
        task_level: str = "graph",
        return_type: str = "regression",
        **kwargs,
    ) -> Explanation:
        """Compute the explanation.

        Args:
            x (torch.Tensor): node features.
            edge_index (torch.Tensor): edge indices.
            model (torch.nn.Module): the model to explain.
            target (torch.Tensor): the target of the model.
            target_index (int): the index of the target to explain.
                By default suppose the target is a single value.
            batch (torch.Tensor, optional): batch indicator. Defaults to None.
            **kwargs: additional arguments to pass to the GNN.

        Returns:
            Explanation: the explanation.
        """

    @abstractmethod
    def explain(
        self,
        g: Data,
        model: torch.nn.Module,
        target: torch.Tensor,
        target_index: int = 0,
        batch: Optional[torch.Tensor] = None,
        task_level: str = "graph",
        return_type: str = "regression",
        **kwargs,
    ) -> Explanation:
        """This method computes the explanation of the GNN.

        This serves as a wrapper around the forward method, so that we can pass
        it a `Data` object.

        Args:
            g (Data): input graph.
            model (torch.nn.Module): model to use in explanations.
            target (torch.Tensor): target of the GNN.
            target_index (int): Index of the target to explain. Used
                in case of multi-outputs. Defaults to 0.
            batch (torch.Tensor, optional): batch tensor. Defaults to None.
            **kwargs: additional arguments to pass to the GNN.

        Returns:
            Explanation: explanation of the GNN.

        .. note: internally  should call the forward method by properly
                splitting the `Data` object into the arguments.
        """

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
