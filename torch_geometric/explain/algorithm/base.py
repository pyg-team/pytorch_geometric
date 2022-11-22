from abc import abstractmethod
from typing import Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.config import ExplainerConfig, ModelConfig
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils.subgraph import get_num_hops


class ExplainerAlgorithm(torch.nn.Module):
    r"""Abstract base class for explainer algorithms."""
    @abstractmethod
    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
        target: Tensor,
        target_index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        r"""Computes the explanation.

        Args:
            model (torch.nn.Module): The model to explain.
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The input edge indices.
            explainer_config (ExplainerConfig): The explainer configuration.
            model_config (ModelConfig): the model configuration.
            target (torch.Tensor): the target of the model.
            target_index (int or torch.Tensor, optional): The target indices to
                explain. (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """

    @abstractmethod
    def loss(self, y_hat: Tensor, y: Tensor, **kwargs) -> Tensor:
        r"""Computes the loss to be used for the explanation algorithm.

        Args:
            y_hat (torch.Tensor): the output of the explanation algorithm.
                (*e.g.*, the forward pass of the model with the mask applied).
            y (torch.Tensor): the reference output.
        """

    @abstractmethod
    def supports(
        self,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
    ) -> bool:
        r"""Checks if the explainer supports the user-defined settings.

        Args:
            explainer_config (ExplainerConfig): The explainer configuration.
            model_config (ModelConfig): the model configuration.
        """
        pass

    ###########################################################################

    @torch.no_grad()
    def get_initial_prediction(self, model: torch.nn.Module, *args,
                               **kwargs) -> Tensor:
        r"""Returns the initial prediction of the model.

        Args:
            model (torch.nn.Module): The model to explain.
            *args: Arguments passed to :obj:`model`.
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """
        return model(*args, **kwargs)

    def subgraph(
        self,
        model: torch.nn.Module,
        node_idx: Union[int, Tensor],
        x: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs,
    ):
        r"""Returns the subgraph for the given node(s).

        Args:
            model (torch.nn.Module): The model to explain.
            node_idx (int or torch.Tensor): The node(s) to explain.
            x (torch.Tensor): The input node feature matrix.
            edge_index (torch.LongTensor): The input edge indices.
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.

        :rtype: (Tensor, LongTensor, LongTensor, LongTensor, BoolTensor, dict)
        """
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=get_num_hops(model),
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes,
            flow=self._flow(model),
        )

        x = x[node_idx]
        for key, value in kwargs.items():
            if torch.is_tensor(value) and value.size(0) == num_nodes:
                kwargs[key] = value[node_idx]
            elif torch.is_tensor(value) and value.size(0) == num_edges:
                kwargs[key] = value[edge_mask]

        return x, edge_index, mapping, node_idx, edge_mask, kwargs

    # Helper functions ########################################################

    def _flow(self, model: torch.nn.Module) -> str:
        for module in model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'
