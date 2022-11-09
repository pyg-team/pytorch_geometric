from abc import abstractmethod
from typing import List, Optional, Tuple, Union

import torch

from torch_geometric.explain import Explanation
from torch_geometric.explain.config import (
    ExplainerConfig,
    MaskType,
    ModelConfig,
    ModelReturnType,
    ModelTaskLevel,
)
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.subgraph import get_num_hops


class ExplainerAlgorithm(torch.nn.Module):
    r"""Abstract class for explanation algorithms."""
    @abstractmethod
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor,
             **kwargs) -> torch.Tensor:
        r"""
        This method computes the loss to be used for the explanation algorithm.

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
    ) -> torch.Tensor:
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
        task_level: ModelTaskLevel = ModelTaskLevel.graph,
        return_type: ModelReturnType = ModelReturnType.regression,
        node_mask_type: MaskType = MaskType.object,
        edge_mask_type: MaskType = MaskType.none,
        index: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs,
    ) -> Explanation:
        """Compute the explanation.

        Args:
            x (torch.Tensor): node features.
            edge_index (torch.Tensor): edge indices.
            model (torch.nn.Module): the model to explain.
            target (torch.Tensor): the target of the model.
            target_index (TargetIndex): Output indices to explain.
                If not provided, the explanation is computed for the first
                index of the target. (default: :obj:`0`)

                For general 1D outputs, targets can be either:

                - a single integer or a tensor containing a single
                  integer, which is applied to all input examples

                - a list of integers or a 1D tensor, with length matching
                  the number of examples (i.e number of unique values in
                  the batch vector). Each integer is applied as the
                  target for the corresponding element of the batch.

                For outputs with > 1 dimension, targets can be either:

                - a single tuple, which contains (:obj:`target.dim()`)
                  elements. This target index is applied for all
                  elements of the batch.

                - a list of tuples with length equal to the number of
                  examples in inputs, and each tuple containing
                  (:obj:`target.dim()`) elements. Each tuple is applied
                  as the target for the corresponding element of the
                  batch.

            batch (torch.Tensor, optional): batch indicator.
            node_mask_type (MaskType): the type of node mask to use.
            edge_mask_type (MaskType): the type of edge mask to use.
            index (Union[int, Tuple[int, ...]], optional): the node/edge index
                to explain. Can be a single index if no batch is provided, or a
                tuple of indices if a batch is provided. only used if the model
                task level is :obj:`"node"` or :obj:`"edge"`.
                (default: :obj:`None`)
            **kwargs: additional arguments to pass to the model.
        """

    @abstractmethod
    def supports(
        self,
        explanation_config: ExplainerConfig,
        model_config: ModelConfig,
    ) -> bool:
        """Check if the explainer supports the user-defined settings.

        Returns true if the explainer supports the settings (mainly the mask
        types), false otherwise. If the explainer does not support the
        settings, an error message explaining the reason is returned.

        Args:
            explanation_config (ExplainerConfig): the user-defined settings.
            model_config (ModelConfig): the model configuration.
        """
        pass

    def _flow(self, model: torch.nn.Module) -> str:
        for module in model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def subgraph(self, model: torch.nn.Module, node_idx: int, x: torch.Tensor,
                 edge_index: torch.Tensor, **kwargs):
        r"""Returns the subgraph of the given node.
        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.
        :rtype: (Tensor, Tensor, LongTensor, LongTensor, LongTensor, dict)
        """
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        num_hops = get_num_hops(model)
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, num_hops, edge_index, relabel_nodes=True,
            num_nodes=num_nodes, flow=self._flow(model))

        x = x[subset]
        kwargs_new = {}
        for key, value in kwargs.items():
            if torch.is_tensor(value) and value.size(0) == num_nodes:
                kwargs_new[key] = value[subset]
            elif torch.is_tensor(value) and value.size(0) == num_edges:
                kwargs_new[key] = value[edge_mask]
            else:
                kwargs_new[key] = value
        return x, edge_index, mapping, edge_mask, subset, kwargs_new
