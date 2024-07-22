from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.config import (
    ExplainerConfig,
    ModelConfig,
    ModelReturnType,
)
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import k_hop_subgraph


class ExplainerAlgorithm(torch.nn.Module):
    r"""An abstract base class for implementing explainer algorithms."""
    @abstractmethod
    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Union[Explanation, HeteroExplanation]:
        r"""Computes the explanation.

        Args:
            model (torch.nn.Module): The model to explain.
            x (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): The input
                node features of a homogeneous or heterogeneous graph.
            edge_index (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): The
                input edge indices of a homogeneous or heterogeneous graph.
            target (torch.Tensor): The target of the model.
            index (Union[int, Tensor], optional): The index of the model
                output to explain. Can be a single index or a tensor of
                indices. (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to
                :obj:`model`.
        """

    @abstractmethod
    def supports(self) -> bool:
        r"""Checks if the explainer supports the user-defined settings provided
        in :obj:`self.explainer_config`, :obj:`self.model_config`.
        """

    ###########################################################################

    @property
    def explainer_config(self) -> ExplainerConfig:
        r"""Returns the connected explainer configuration."""
        if not hasattr(self, '_explainer_config'):
            raise ValueError(
                f"The explanation algorithm '{self.__class__.__name__}' is "
                f"not yet connected to any explainer configuration. Please "
                f"call `{self.__class__.__name__}.connect(...)` before "
                f"proceeding.")
        return self._explainer_config

    @property
    def model_config(self) -> ModelConfig:
        r"""Returns the connected model configuration."""
        if not hasattr(self, '_model_config'):
            raise ValueError(
                f"The explanation algorithm '{self.__class__.__name__}' is "
                f"not yet connected to any model configuration. Please call "
                f"`{self.__class__.__name__}.connect(...)` before "
                f"proceeding.")
        return self._model_config

    def connect(
        self,
        explainer_config: ExplainerConfig,
        model_config: ModelConfig,
    ):
        r"""Connects an explainer and model configuration to the explainer
        algorithm.
        """
        self._explainer_config = ExplainerConfig.cast(explainer_config)
        self._model_config = ModelConfig.cast(model_config)

        if not self.supports():
            raise ValueError(
                f"The explanation algorithm '{self.__class__.__name__}' does "
                f"not support the given explanation settings.")

    # Helper functions ########################################################

    @staticmethod
    def _post_process_mask(
        mask: Optional[Tensor],
        hard_mask: Optional[Tensor] = None,
        apply_sigmoid: bool = True,
    ) -> Optional[Tensor]:
        r""""Post processes any mask to not include any attributions of
        elements not involved during message passing.
        """
        if mask is None:
            return mask

        mask = mask.detach()

        if apply_sigmoid:
            mask = mask.sigmoid()

        if hard_mask is not None and mask.size(0) == hard_mask.size(0):
            mask[~hard_mask] = 0.

        return mask

    @staticmethod
    def _get_hard_masks(
        model: torch.nn.Module,
        node_index: Optional[Union[int, Tensor]],
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        r"""Returns hard node and edge masks that only include the nodes and
        edges visited during message passing.
        """
        if node_index is None:
            return None, None  # Consider all nodes and edges.

        index, _, _, edge_mask = k_hop_subgraph(
            node_index,
            num_hops=ExplainerAlgorithm._num_hops(model),
            edge_index=edge_index,
            num_nodes=num_nodes,
            flow=ExplainerAlgorithm._flow(model),
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

    def _loss_binary_classification(self, y_hat: Tensor, y: Tensor) -> Tensor:
        if self.model_config.return_type == ModelReturnType.raw:
            loss_fn = F.binary_cross_entropy_with_logits
        elif self.model_config.return_type == ModelReturnType.probs:
            loss_fn = F.binary_cross_entropy
        else:
            assert False

        return loss_fn(y_hat.view_as(y), y.float())

    def _loss_multiclass_classification(
        self,
        y_hat: Tensor,
        y: Tensor,
    ) -> Tensor:
        if self.model_config.return_type == ModelReturnType.raw:
            loss_fn = F.cross_entropy
        elif self.model_config.return_type == ModelReturnType.probs:
            loss_fn = F.nll_loss
            y_hat = y_hat.log()
        elif self.model_config.return_type == ModelReturnType.log_probs:
            loss_fn = F.nll_loss
        else:
            assert False

        return loss_fn(y_hat, y)

    def _loss_regression(self, y_hat: Tensor, y: Tensor) -> Tensor:
        assert self.model_config.return_type == ModelReturnType.raw
        return F.mse_loss(y_hat, y)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
