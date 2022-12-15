import warnings
from typing import List

import torch
from torch import Tensor


def get_message_passing_embeddings(model: torch.nn.Module, *args,
                                   **kwargs) -> List[Tensor]:
    """Returns the output embeddings of all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers in
    :obj:`model`.

    Internally registers forward hooks on all :class:`MessagePassing`
    layers in :obj:`model` and runs a forward pass by calling
    `model(*args, **kwargs)`.

    Args:
        *args: Input to :obj:`model`.
        **kwargs: Input to :obj:`model`.
    """
    from torch_geometric.nn import MessagePassing
    intermediate_out = []

    def hook(model, input, output):
        # clone output in case it is modified inplace
        intermediate_out.append(output.clone())

    # Register forward hooks
    hook_handles = []
    for module in model.modules():
        if isinstance(module, MessagePassing):
            hook_handles.append(module.register_forward_hook(hook))

    # Run forward pass
    if len(hook_handles) == 0:
        warnings.warn("'model' does not have any 'MessagePassing' layers.")
    else:
        model_state = model.training
        model.eval()
        with torch.no_grad():
            model(*args, **kwargs)
        model.train(model_state)

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    return intermediate_out
