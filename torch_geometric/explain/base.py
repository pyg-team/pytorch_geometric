from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import torch

from torch_geometric.explain.explanations import Explanation


class ExplainerAlgorithm(torch.nn.Module):
    """Abstract class for explanation algorithms."""
    @abstractmethod
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        This method compute the loss to be used for the explanation algorithm.

        Args:
            y_hat (torch.Tensor): the output of the explanation algorithm.
                (e.g. the forward pass of the model with the mask applied).
            y (torch.Tensor): the reference output.
        """

    @torch.no_grad()
    def get_initial_prediction(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        model: torch.nn.Module,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Return the initial prediction of the model.

        Args:
            x (torch.Tensor): node features.
            edge_index (torch.Tensor): edge indices.
            model (torch.nn.Module): the model to explain.
            batch (torch.Tensor, optional): batch indicator.
            **kwargs: additional arguments to pass to the model.
        """
        return model(x=x, edge_index=edge_index, batch=batch, **kwargs)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        model: torch.nn.Module,
        target: torch.Tensor,
        target_index: Union[int, Tuple[int, ...], torch.Tensor,
                            List[Tuple[int, ...]], List[int]] = 0,
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
            target_index: TargetIndex
                Output indices to explain. If not provided, the explanation is
                computed for the first index of the target. (default: :obj:`0`)

                For general 1D outputs, targets can be either:

                    . a single integer or a tensor containing a single
                        integer, which is applied to all input examples

                    . a list of integers or a 1D tensor, with length matching
                        the number of examples (i.e number of unique values in
                        the batch vector). Each integer is applied as the
                        target for the corresponding element of the batch.

                For outputs with > 1 dimension, targets can be either:

                    . a single tuple, which contains (:obj:`target.dim()`)
                        elements. This target index is applied for all
                        elements of the batch.

                    . a list of tuples with length equal to the number of
                        examples in inputs, and each tuple containing
                        (:obj:`target.dim()`) elements. Each tuple is applied
                        as the target for the corresponding element of the
                        batch.

            batch (torch.Tensor, optional): batch indicator.
            **kwargs: additional arguments to pass to the GNN.
        """

    @abstractmethod
    def supports(
        self,
        explanation_type: str,
        mask_type: str,
    ) -> bool:
        """Check if the explainer supports the user-defined settings.

        Returns true if the explainer supports the settings.


        Args:
            explanation_type (str): the type of explanation to compute.
                Should be in :obj:`"model"`, or :obj:`"phenomenon"`
            mask_type (str): the type of mask to use.
                Should be in :obj:`"node"`, :obj:`"edge"`,
                :obj:`"node_and_edge"`, or :obj:`"layers"`.
        """
