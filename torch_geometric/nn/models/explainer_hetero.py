from typing import Dict

import torch
from torch import Tensor

from torch_geometric.nn import HeteroConv
from torch_geometric.nn.models.explainer import clear_masks, set_masks
from torch_geometric.typing import EdgeType


def set_masks_hetero(model: torch.nn.Module, mask_dict: Dict[EdgeType, Tensor],
                     edge_index_dict: Dict[EdgeType, Tensor],
                     apply_sigmoid: bool = True):
    """Apply masks to each `HeteroConv` layer used in `model`."""

    for module in model.modules():
        if isinstance(module, HeteroConv):
            # Set masks for the message passing layer of each
            # edge type.
            for edge_type, mask in mask_dict.items():
                # TODO(jinu) use common function get get
                # str_edge_type
                str_edge_type = '__'.join(edge_type)
                set_masks(
                    module.convs[str_edge_type],
                    mask,
                    edge_index_dict[edge_type],
                    apply_sigmoid=apply_sigmoid,
                )


def clear_masks_hetero(model: torch.nn.Module):
    """Clear masks of each `HeteroConv` layer used in `model`."""
    for module in model.modules():
        if isinstance(module, HeteroConv):
            for _, conv_module in module.convs.items():
                clear_masks(conv_module)
