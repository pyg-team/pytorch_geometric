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

        This will be set by the Explainer class.

        Args:
            exp_output (torch.Tensor): the output of the explanation algorithm.
                (e.g. the forward pass of the model with the mask applied).
            gnn_output (torch.Tensor): the original output of the GNN.
            target (torch.Tensor): the target of the GNN.
            target_index (int): the index of the target to explain.
        """

    @abstractmethod
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        This method compute the loss to be used for the explanation algorithm.
        Subclasses should override this method to define their own loss.

        Args:
            y_hat (torch.Tensor): the output of the explanation algorithm.
                (e.g. the forward pass of the model with the mask applied).
            y (torch.Tensor): the target output.
        """

    @property
    def accept_new_loss(self) -> bool:
        """Whether the algorithm can accept a new loss function."""
        return True

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

    def _create_explanation_from_masks(self, g, attributions, mask_type):
        """Create explanation from masks.

        Args:
            g (Data): input graph.
            attributions (Tuple[torch.Tensor]): masks returned by captum.
            kwargs (dict): additional information to store in the explanation.

        Returns:
            Explanation: explanation object.
        """
        node_features_mask = None
        edge_mask = None
        if mask_type == "node":
            node_features_mask = attributions.squeeze(0)
        elif mask_type == "edge":
            edge_mask = attributions.squeeze(0)
        elif mask_type == "node_and_edge":
            node_features_mask = attributions[0].squeeze(0)
            edge_mask = attributions[1].squeeze(0)

        return Explanation(
            x=g.x,
            edge_index=g.edge_index,
            node_features_mask=node_features_mask,
            edge_mask=edge_mask,
        )
