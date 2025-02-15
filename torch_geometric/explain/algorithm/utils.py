from typing import Dict, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn import MessagePassing
from torch_geometric.typing import EdgeType


def set_masks(
    model: torch.nn.Module,
    mask: Union[Tensor, Parameter],
    edge_index: Tensor,
    apply_sigmoid: bool = True,
):
    r"""Apply mask to every graph layer in the :obj:`model`."""
    loop_mask = edge_index[0] != edge_index[1]

    # Loop over layers and set masks on MessagePassing layers:
    for module in model.modules():
        if isinstance(module, MessagePassing):
            # Skip layers that have been explicitly set to `False`:
            if module.explain is False:
                continue

            # Convert mask to a param if it was previously registered as one.
            # This is a workaround for the fact that PyTorch does not allow
            # assignments of pure tensors to parameter attributes:
            if (not isinstance(mask, Parameter)
                    and '_edge_mask' in module._parameters):
                mask = Parameter(mask)

            module.explain = True
            module._edge_mask = mask
            module._loop_mask = loop_mask
            module._apply_sigmoid = apply_sigmoid


def set_hetero_masks(
    model: torch.nn.Module,
    mask_dict: Dict[EdgeType, Union[Tensor, Parameter]],
    edge_index_dict: Dict[EdgeType, Tensor],
    apply_sigmoid: bool = True,
):
    r"""Apply masks to every heterogeneous graph layer in the :obj:`model`
    according to edge types.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleDict):
            for edge_type in mask_dict.keys():
                if edge_type in module:
                    edge_level_module = module[edge_type]
                elif '__'.join(edge_type) in module:
                    edge_level_module = module['__'.join(edge_type)]
                else:
                    continue

                set_masks(
                    edge_level_module,
                    mask_dict[edge_type],
                    edge_index_dict[edge_type],
                    apply_sigmoid=apply_sigmoid,
                )


def clear_masks(model: torch.nn.Module):
    r"""Clear all masks from the model."""
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if module.explain is True:
                module.explain = None
            module._edge_mask = None
            module._loop_mask = None
            module._apply_sigmoid = True
    return module
