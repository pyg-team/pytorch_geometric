from abc import abstractmethod
from typing import Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.config import (
    ExplainerConfig,
    ModelConfig,
    ModelMode,
    ModelReturnType,
)
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
        index: Optional[Union[int, Tensor]] = None,
        target_index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:
        r"""Computes the explanation.

        Args:
            model (torch.nn.Module): The model to explain.
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The input edge indices.
            explainer_config (ExplainerConfig): The explainer configuration.
            model_config (ModelConfig): The model configuration.
            target (torch.Tensor): The target of the model.
            index (Union[int, Tensor], optional): The index of the model
                output to explain. Can be a single index or a tensor of
                indices. (default: :obj:`None`)
            target_index (int or torch.Tensor, optional): The target indices to
                explain in case targets are multi-dimensional.
                (default: :obj:`None`)
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
    def get_initial_prediction(
        self,
        model: torch.nn.Module,
        *args,
        model_mode: ModelMode,
        **kwargs,
    ) -> Tensor:
        r"""Returns the initial prediction of the model.

        If the model mode is :obj:`"regression"`, the prediction is returned as
        a scalar value.
        If the model mode is :obj:`"classification"`, the prediction is
        returned as the predicted class label.

        Args:
            model (torch.nn.Module): The model to explain.
            *args: Arguments passed to :obj:`model`.
            model_mode (ModelMode): The mode of the model.
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """
        out = model(*args, **kwargs)
        if model_mode == ModelMode.classification:
            out = out.argmax(dim=-1)
        return out

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
        r"""Determines the message passing flow of the :obj:`model`."""
        for module in model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def _to_log_prob(self, y: Tensor, return_type: ModelReturnType) -> Tensor:
        r"""Converts the model output to log-probabilities.

        Args:
            y (Tensor): The output of the model.
            return_type (ModelReturnType): The model return type.
        """
        if return_type == ModelReturnType.probs:
            return y.log()
        if return_type == ModelReturnType.raw:
            return y.log_softmax(dim=-1)
        if return_type == ModelReturnType.log_probs:
            return y
        raise NotImplementedError

    def _reshape_common_attributes(
        self,
        common_mask: Tensor,
        num_objects: int,
    ) -> Tensor:
        r"""Reshapes the common mask from shape :obj:`[1, F]` or `F` to
        :obj:`[N, F]` where :obj:`N` refers to the number of objects.

        Args:
            common_mask (Tensor): the common mask.
            number_object (int): the number of objects.
        """
        return torch.stack([common_mask.squeeze(0)] * num_objects, dim=0)
