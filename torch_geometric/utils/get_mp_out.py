import warnings
from typing import Callable, List

import torch
from torch import Tensor


def get_messagepassing_embeddings(model: torch.nn.Module,
                                  **kwargs) -> List[Tensor]:
    """Returns the output embeddings of all `MessagePassing` layers in
    :obj:`model`.

    Args:
        **kwargs: Input to :obj:`model`.
    """
    from torch_geometric.nn import MessagePassing
    intermediate_out = []

    def get_intermediate_output() -> Callable:
        def hook(model, input, output):
            intermediate_out.append(output)

        return hook

    # Register forward hooks
    hook_handles = []
    for module in model.modules():
        if isinstance(module, MessagePassing):
            hook_handles.append(
                module.register_forward_hook(get_intermediate_output()))

    # Run forward pass
    if len(hook_handles) == 0:
        warnings.warn("'model' does not have message passing layer.")
    else:
        model_state = model.training
        model.eval()
        with torch.no_grad():
            model(**kwargs)
        model.train(model_state)

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    return intermediate_out
