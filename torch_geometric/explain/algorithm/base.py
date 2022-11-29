from abc import abstractmethod
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.config import (
    ExplainerConfig,
    ModelConfig,
    ModelReturnType,
)
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph


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
        target_index: Optional[int] = None,
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
            target_index (int, optional): The index of the model outputs to
                reference in case the model returns a list of tensors, *e.g.*,
                in a multi-task learning scenario. Should be kept to
                :obj:`None` in case the model only returns a single output
                tensor. (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
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

    # Helper functions ########################################################

    def _post_process_mask(
        self,
        mask: Optional[Tensor],
        num_elems: int,
        hard_mask: Optional[Tensor] = None,
        apply_sigmoid: bool = True,
    ) -> Optional[Tensor]:
        r""""Post processes any mask to not include any attributions of
        elements not involved during message passing."""
        if mask is None:
            return mask

        if mask.size(0) == 1:  # common_attributes:
            mask = mask.repeat(num_elems, 1)

        mask = mask.detach().squeeze(-1)

        if apply_sigmoid:
            mask = mask.sigmoid()

        if hard_mask is not None:
            mask[~hard_mask] = 0.

        return mask

    def _get_hard_masks(
        self,
        model: torch.nn.Module,
        index: Optional[Union[int, Tensor]],
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""Returns hard node and edge masks that only include the nodes and
        edges visited during message passing."""
        if index is None:
            return None, None  # Consider all nodes and edges.

        index, _, _, edge_mask = k_hop_subgraph(
            index,
            num_hops=self._num_hops(model),
            edge_index=edge_index,
            num_nodes=num_nodes,
            flow=self._flow(model),
        )

        node_mask = edge_index.new_zeros(num_nodes, dtype=torch.bool)
        node_mask[index] = True

        return node_mask, edge_mask

    @staticmethod
    def _num_hops(model: torch.nn.Module) -> int:
        r"""Returns the number of hops the :obj:`model` is aggregating
        information from.
        """
        num_hops = 0
        for module in model.modules():
            if isinstance(module, MessagePassing):
                num_hops += 1
        return num_hops

    @staticmethod
    def _flow(model: torch.nn.Module) -> str:
        r"""Determines the message passing flow of the :obj:`model`."""
        for module in model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    @staticmethod
    def _to_log_prob(y: Tensor, return_type: ModelReturnType) -> Tensor:
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
