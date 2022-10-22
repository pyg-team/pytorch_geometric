from abc import abstractmethod
from typing import Callable, Tuple

import torch

from torch_geometric.explainability.explanations import Explanation


class ExplainerAlgorithm(torch.nn.Module):
    r"""An abstract class for explainer algorithms."""
    def __init__(self) -> None:
        # TODO: check how to initialize... I don't like the fact that
        # an explainer algorithm "has" a model
        # it should be decoupled no ?
        super().__init__()

        # TODO: check if this is good practice, took the idea from captum
        objective: Callable[
            [Tuple[torch.Tensor,
                   ...], torch.Tensor, torch.Tensor], torch.Tensor]
        r"""
        This method compute the loss to be used for the explanation algorithm.
        Subclasses should override this method to define their own loss.

        Args:
            inputs (Tuple[torch.Tensor, ...]): the inputs of the GNN.
            output (torch.Tensor): the output of the GNN.
            target (torch.Tensor): the target of the GNN.
        """

    def set_objective(
        self,
        objective: Callable[
            [Tuple[torch.Tensor,
                   ...], torch.Tensor, torch.Tensor], torch.Tensor],
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
        **kwargs,
    ) -> Explanation:
        r"""This method computes the explanation of the GNN and return
        a list of masks.

        Args:
            inputs (Tuple[torch.Tensor, ...]): the inputs of the GNN.
            target (torch.Tensor): the target of the GNN.
            model (torch.nn.Module): the GNN to explain.
            name_inputs (List[str]): the name of the inputs of the GNN.
                (used to create the explanation object)
        """
        pass

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

        """
        return True


class RandomExplainer(ExplainerAlgorithm):
    """Dummy explainer."""
    def __init__(self) -> None:
        super().__init__()

    def explain(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        target: torch.Tensor,
        model: torch.nn.Module,
        **kwargs,
    ) -> Explanation:

        node_features_mask = torch.rand(x.shape)
        node_mask = torch.rand(x.shape[0])
        edge_mask = torch.rand(edge_index.shape[1])
        if "edge_attr" in kwargs:
            edge_features_mask = torch.rand(kwargs["edge_attr"].shape)
        return Explanation(
            x=x,
            edge_index=edge_index,
            node_features_mask=node_features_mask,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_features_mask=edge_features_mask,
            **kwargs,
        )

    def supports(
        self,
        explanation_type: str,
        mask_type: str,
    ) -> bool:
        return mask_type != "layers"
