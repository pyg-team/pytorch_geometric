from typing import Callable, List

import torch
from torch import Tensor

from .subgraph import get_num_hops


def get_intermediate_messagepassing_embeddings(model: torch.nn.Module,
                                               **kwargs) -> List[Tensor]:
    """Get output of all `MessagePassing` layers in
    :obj:`model`.

    Args:
        **kwargs: Input to :obj:`model`.
    """
    from torch_geometric.nn import MessagePassing
    intermediate_out = {}

    def get_intermediate_output(name) -> Callable:
        def hook(model, input, output):
            intermediate_out[name] = output

        return hook

    num_mp_layers = get_num_hops(model)
    if num_mp_layers == 0:
        raise ValueError("'model' does not have message passing layer.")

    # Register forward hooks
    index = 0
    hook_handles = []
    for module in model.modules():
        if isinstance(module, MessagePassing):
            hook_handles.append(
                module.register_forward_hook(get_intermediate_output(index)))
            index += 1

    # Run forward pass
    model_state = model.training
    model.eval()
    _ = model(**kwargs)
    model.train(model_state)

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    return [intermediate_out[i] for i in range(num_mp_layers)]
