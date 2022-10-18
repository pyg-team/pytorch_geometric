from abc import abstractmethod
from typing import Callable, Tuple

import torch


class ExplainerAlgorithm(torch.nn.Module):
    r"""An abstract class for explainer algorithms."""
    def __init__(self) -> None:
        # TODO: check how to initialize... I don't like the fact that
        # an explainer algorithm "has" a model
        # it should be decoupled no ?
        super().__init__()

        # TODO: check if this is good practice, took the idea from captum
        loss: Callable[[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
                       torch.Tensor]
        r"""
        This method compute the loss to be used for the explanation algorithm.
        Subclasses should override this method to define their own loss.

        Args:
            inputs (Tuple[torch.Tensor, ...]): the inputs of the GNN.
            output (torch.Tensor): the output of the GNN.
            target (torch.Tensor): the target of the GNN.
        """

    def _set_loss(
        self,
        loss: Callable[[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor],
                       torch.Tensor],
    ) -> None:
        """Sets the loss function to be used for the explanation algorithm.

        Args:
            loss (Callable): loss function.
        """
        self.loss = loss

    # TODO: how to work with the inputs for all kind of models?
    # (index, features...)
    @abstractmethod
    def explain(
        self,
        inputs: Tuple[torch.Tensor, ...],
        target: torch.Tensor,
        model: torch.nn.Module,
    ) -> Tuple[torch.Tensor, ...]:
        r"""This method computes the explanation of the GNN and return
        a list of masks.

        Args:
            inputs (Tuple[torch.Tensor, ...]): the inputs of the GNN.
            target (torch.Tensor): the target of the GNN.
            model (torch.nn.Module): the GNN to explain.
        """
        pass

    # TODO: improve on just passing a general dict
    @abstractmethod
    def supports(self, *args, **kwargs) -> bool:
        r"""Check if the explainer supports the user-defined settings.

        Returns true if the explainer supports the settings.

        Args:
            *args: the user-defined explanation settings.
            **kwargs: the user-defined explanation settings.
        """
        pass


class Saliency(ExplainerAlgorithm):
    """Saliency explainer."""
    def __init__(self) -> None:
        super().__init__()

    def explain(
        self,
        inputs: Tuple[torch.Tensor, ...],
        target: torch.Tensor,
        model: torch.nn.Module,
    ) -> Tuple[torch.Tensor, ...]:
        # TODO: implement it, check captum or do it manually.
        raise NotImplementedError()
