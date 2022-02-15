from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn import MessagePassing


def set_masks(model: torch.nn.Module, mask: Tensor, edge_index: Tensor,
              apply_sigmoid: bool = True):
    """Apply mask to every graph layer in the model."""
    loop_mask = edge_index[0] != edge_index[1]

    # Loop over layers and set masks on MessagePassing layers:
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
    return module


class CaptumModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mask_type: str = "edge",
                 output_idx: Optional[int] = None):
        super().__init__()
        assert mask_type in ['edge', 'node', 'node_and_edge']

        self.mask_type = mask_type
        self.model = model
        self.output_idx = output_idx

    def forward(self, mask, *args):
        """"""
        # The mask tensor, which comes from Captum's attribution methods,
        # contains the number of samples in dimension 0. Since we are
        # working with only one sample, we squeeze the tensors below.
        assert mask.shape[0] == 1, "Dimension 0 of input should be 1"
        if self.mask_type == "edge":
            assert len(args) >= 2, "Expects at least x and edge_index as args."
        if self.mask_type == "node":
            assert len(args) >= 1, "Expects at least edge_index as args."
        if self.mask_type == "node_and_edge":
            assert args[0].shape[0] == 1, "Dimension 0 of input should be 1"
            assert len(args[1:]) >= 1, "Expects at least edge_index as args."

        # Set edge mask:
        if self.mask_type == 'edge':
            set_masks(self.model, mask.squeeze(0), args[1],
                      apply_sigmoid=False)
        elif self.mask_type == 'node_and_edge':
            set_masks(self.model, args[0].squeeze(0), args[1],
                      apply_sigmoid=False)
            args = args[1:]

        if self.mask_type == 'edge':
            x = self.model(*args)

        elif self.mask_type == 'node':
            x = self.model(mask.squeeze(0), *args)

        else:
            x = self.model(mask[0], *args)

        # Clear mask:
        if self.mask_type in ['edge', 'node_and_edge']:
            clear_masks(self.model)

        if self.output_idx is not None:
            x = x[self.output_idx].unsqueeze(0)

        return x


def to_captum(model: torch.nn.Module, mask_type: str = "edge",
              output_idx: Optional[int] = None) -> torch.nn.Module:
    r"""Converts a model to a model that can be used for
    `Captum.ai <https://captum.ai/>`_ attribution methods.

    .. code-block:: python

        from captum.attr import IntegratedGradients
        from torch_geometric.nn import GCN, from_captum

        model = GCN(...)
        ...  # Train the model.

        # Explain predictions for node `10`:
        output_idx = 10

        captum_model = to_captum(model, mask_type="edge",
                                 output_idx=output_idx)
        edge_mask = torch.ones(num_edges, requires_grad=True, device=device)

        ig = IntegratedGradients(captum_model)
        ig_attr = ig.attribute(edge_mask.unsqueeze(0),
                               target=int(y[output_idx]),
                               additional_forward_args=(x, edge_index),
                               internal_batch_size=1)

    .. note::
        For an example of using a Captum attribution method within PyG, see
        `examples/captum_explainability.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        captum_explainability.py>`_.

    Args:
        model (torch.nn.Module): The model to be explained.
        mask_type (str, optional): Denotes the type of mask to be created with
            a Captum explainer. Valid inputs are :obj:`"edge"`, :obj:`"node"`,
            and :obj:`"node_and_edge"`:

            1. :obj:`"edge"`: The inputs to the forward function should be an
               edge mask tensor of shape :obj:`[1, num_edges]`, a regular
               :obj:`x` matrix and a regular :obj:`edge_index` matrix.

            2. :obj:`"node"`: The inputs to the forward function should be a
               node feature tensor of shape :obj:`[1, num_nodes, num_features]`
               and a regular :obj:`edge_index` matrix.

            3. :obj:`"node_and_edge"`: The inputs to the forward function
               should be a node feature tensor of shape
               :obj:`[1, num_nodes, num_features]`, an edge mask tensor of
               shape :obj:`[1, num_edges]` and a regular :obj:`edge_index`
               matrix.

            For all mask types, additional arguments can be passed to the
            forward function as long as the first arguments are set as
            described. (default: :obj:`"edge"`)
        output_idx (int, optional): Index of the output element (node or link
            index) to be explained. With :obj:`output_idx` set, the forward
            function will return the output of the model for the element at
            the index specified. (default: :obj:`None`)
    """
    return CaptumModel(model, mask_type, output_idx)
