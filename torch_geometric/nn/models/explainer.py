from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn import MessagePassing


# TODO: Use set_masks, clear_masks also for GNNExplainer
def set_masks(model: torch.nn.Module, mask: Tensor, edge_index: Tensor,
              apply_sigmoid: bool = True):
    """Apply mask to every graph layer in the model."""
    loop_mask = edge_index[0] != edge_index[1]

    # Loop over layers and set masks on MessagePassing layers
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = mask
            module.__loop_mask__ = loop_mask
            module.__apply_sigmoid__ = apply_sigmoid


def clear_masks(model: torch.nn.Module):
    """Clear all masks from the model."""
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None
            module.__loop_mask__ = None
            module.__apply_sigmoid__ = True


class CaptumModel(torch.nn.Module):
    r"""Model with forward function that can be easily used for
    explainability with `Captum.ai <https://captum.ai/>`_.

    Args:
        model (torch.nn.Module): Model to be explained.
        mask_type (str): Denotes the type of mask which is to be created with a
            captum explainer. Valid inputs are :obj:`'edge'`, :obj:`'node'`,
            and :obj:`'node_and_edge'`. (default: :obj:`'edge'`)
        node_idx (int, optional): Index of the node to be explained. With
            :obj:`'node_idx'` set, the forward function will return the output
            of the model for the node at the index specified.
            (default: :obj:`None`)
    """
    def __init__(self, model: torch.nn.Module, mask_type: str = "edge",
                 node_idx: Optional[int] = None):
        super().__init__()

        assert mask_type in ['edge', 'node', 'node_and_edge']

        self.mask_type = mask_type
        self.model = model
        self.node_idx = node_idx

    def forward(self, mask, *args):
        """"""
        # Set edge mask
        if self.mask_type == 'edge':
            set_masks(self.model, mask.squeeze(0), args[1],
                      apply_sigmoid=False)
        elif self.mask_type == 'node_and_edge':
            set_masks(self.model, args[0].squeeze(0), args[1],
                      apply_sigmoid=False)
            args = args[1:]

        # Edge mask
        if self.mask_type == 'edge':
            x = self.model(*args)

        # Node mask
        elif self.mask_type == 'node':
            x = self.model(mask.squeeze(0), *args)

        # Node and edge mask
        else:
            x = self.model(mask[0], *args)

        # Clear mask
        if self.mask_type in ['edge', 'node_and_edge']:
            clear_masks(self.model)

        if self.node_idx is not None:
            x = x[self.node_idx].unsqueeze(0)
        return x


def to_captum(model: torch.nn.Module, mask_type: str = "edge",
              node_idx: Optional[int] = None) -> torch.nn.Module:
    """Convert a model to a model that can be used for Captum explainers."""
    return CaptumModel(model, mask_type, node_idx)
