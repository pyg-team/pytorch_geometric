import logging
from typing import Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm.utils import clear_masks
from torch_geometric.explain.config import MaskType, ModelTaskLevel
from torch_geometric.nn.conv.message_passing import MessagePassing

from .base import ExplainerAlgorithm


class AttentionExplainer(ExplainerAlgorithm):
    r"""Attention Explainer provides explanations for edges by aggregating attention scores over attention-based layers.

    The following configurations are currently supported:

    - :class:`torch_geometric.explain.config.ModelConfig`

        - :attr:`task_level`: :obj:`"node"`, :obj:`"edge"`, or :obj:`"graph"`

    - :class:`torch_geometric.explain.config.ExplainerConfig`

        - :attr:`edge_mask_type`: :obj:`"object"`

    Args:
        aggregate_fn (str, optional): The method to aggregate the attention scores .
            (default: :obj:`max`)
    """

    def __init__(self, aggregate_fn: str = "max"):
        super().__init__()
        self.edge_mask = None
        self.aggregate_fn = aggregate_fn

    def supports(self) -> bool:
        task_level = self.model_config.task_level
        if task_level not in [
            ModelTaskLevel.node,
            ModelTaskLevel.edge,
            ModelTaskLevel.graph,
        ]:
            logging.error(f"Task level '{task_level.value}' not supported")
            return False

        edge_mask_type = self.explainer_config.edge_mask_type
        if edge_mask_type != MaskType.object:
            logging.error(
                f"Edge mask type '{edge_mask_type.value}' not supported"
            )
            return False

        return True

    def forward(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
    ) -> Explanation:

        # Get attention scores per layer.
        attention_scores = [
            layer.get_alphas()
            for layer in model.modules()
            if isinstance(layer, MessagePassing)
            and layer.get_alphas() is not None
        ]

        if not attention_scores:
            logging.error("No Attention layers used")

        attention_scores = torch.cat(attention_scores, dim=-1)

        if self.aggregate_fn == "mean":
            edge_mask = torch.mean(attention_scores, dim=-1)
        elif self.aggregate_fn == "max":
            edge_mask = torch.max(attention_scores, dim=-1)
        else:
            raise NotImplementedError

        if self.model_config.task_level == ModelTaskLevel.node:
            # We need to compute hard masks to properly clean up edges and
            # nodes attributions not involved during message passing:
            hard_node_mask, hard_edge_mask = self._get_hard_masks(
                model, index, edge_index, num_nodes=x.size(0)
            )

        edge_mask = self._post_process_mask(
            self.edge_mask,
            edge_index.size(1),
            hard_edge_mask,
            apply_sigmoid=True,
        )

        self._clean_model(model)

        return Explanation(
            x=x,
            edge_index=edge_index,
            edge_mask=edge_mask,
            node_mask=None,
            node_feat_mask=None,
            edge_feat_mask=None,
        )

    def _clean_model(self, model):
        clear_masks(model)
        self.edge_mask = None
