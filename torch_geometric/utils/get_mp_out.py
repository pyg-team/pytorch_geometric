from typing import Callable

import torch
from torch import Tensor

from .subgraph import get_num_hops


def intermediate_mp_output(i: int, model: torch.nn.Module, **kwargs) -> Tensor:
    """Get output of :obj:`i`th `MessagePassing` layer in
    :obj:`model`.

    Args:
        **kwargs: Input to :obj:`model`.
    """
    from torch_geometric.nn import MessagePassing
    intermediate_out = {}

    def get_intermediate_output() -> Callable:
        def hook(model, input, output):
            intermediate_out['feat'] = output

        return hook

    num_mp_layers = get_num_hops(model)
    mp_index = i if i >= 0 else num_mp_layers + i
    if num_mp_layers == 0:
        raise ValueError("'model' does not have message passing layer.")
    if num_mp_layers < mp_index:
        raise ValueError(
            "'i' has to take values in "
            f"range {[-num_mp_layers, num_mp_layers-1]} got ({i})")

    # Register forward hook
    mp_count = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            if mp_count == mp_index:
                module.register_forward_hook(get_intermediate_output())
                break
            mp_count += 1

    # Run forward pass
    model_state = model.training
    model.eval()
    _ = model(**kwargs)
    model.train(model_state)

    return intermediate_out['feat']
