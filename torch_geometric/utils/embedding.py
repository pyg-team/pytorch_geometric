import warnings
from typing import Any, List

import torch
from torch import Tensor


def get_embeddings(
    model: torch.nn.Module,
    *args: Any,
    **kwargs: Any,
) -> List[Tensor]:
    """Returns the output embeddings of all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers in
    :obj:`model`.

    Internally, this method registers forward hooks on all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers of a :obj:`model`,
    and runs the forward pass of the :obj:`model` by calling
    :obj:`model(*args, **kwargs)`.

    Args:
        model (torch.nn.Module): The message passing model.
        *args: Arguments passed to the model.
        **kwargs (optional): Additional keyword arguments passed to the model.
    """
    from torch_geometric.nn import MessagePassing

    embeddings: List[Tensor] = []

    def hook(model: torch.nn.Module, inputs: Any, outputs: Any) -> None:
        # Clone output in case it will be later modified in-place:
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        assert isinstance(outputs, Tensor)
        embeddings.append(outputs.clone())

    hook_handles = []
    for module in model.modules():  # Register forward hooks:
        if isinstance(module, MessagePassing):
            hook_handles.append(module.register_forward_hook(hook))

    if len(hook_handles) == 0:
        warnings.warn("The 'model' does not have any 'MessagePassing' layers")

    training = model.training
    model.eval()
    with torch.no_grad():
        model(*args, **kwargs)
    model.train(training)

    for handle in hook_handles:  # Remove hooks:
        handle.remove()

    return embeddings
