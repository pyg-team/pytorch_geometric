from collections import defaultdict
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.jit import ScriptModule
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing


def summary(
    model: nn.Module,
    *inputs: Sequence[Any],
    max_depth: int = 3,
    leaf_module: Union[nn.Module, Tuple[nn.Module],
                       Optional[nn.Module]] = MessagePassing,
    **kwargs,
) -> str:
    r"""Summarizes the given PyTorch model. Summarized information includes
    (1) Layer names, (2) Input/output shapes, and (3) # of parameters.

    .. code-block:: python

        import torch
        from torch_geometric.nn import summary
        from torch_geometric.nn.models import GCN

        model = GCN(128, 64, 2, out_channels=32)
        x = torch.randn(100, 128)
        edge_index = torch.randint(100, size=(2, 20))
        print(summary(model, x, edge_index))

    Args:
        model (nn.Module): PyTorch model to summarize.
        inputs (Sequence[Any]): Arguments for the model's
            forward pass.
        max_depth (int): Depth of nested layers to display.
            Nested layers below this depth will not be displayed
            in the summary. (default: :obj:`"3"`)
        leaf_module (Union[nn.Module, Tuple[nn.Module], Optional[nn.Module]]):
            Modules to be treated as leaf ones, whose submodules are excluded
            from the the summarized information.
            (default: :obj:`"MessagePassing"`)
        **kwargs: Other arguments used in `model.forward` function.
    """
    def register_hook(info):
        def hook(module, input, output):
            info['input_shape'].append(get_shape(input))
            info['output_shape'].append(get_shape(output))

        return hook

    hooks = {}
    depth = 0
    stack = [(get_name(model), model, depth)]

    info_list = []
    input_shape = defaultdict(list)
    output_shape = defaultdict(list)
    while stack:
        var_name, module, depth = stack.pop()

        module_id = id(module)
        if module_id in hooks:
            hooks[module_id].remove()

        info = {}
        info['name'] = var_name
        info['input_shape'] = input_shape[module_id]
        info['output_shape'] = output_shape[module_id]
        info['depth'] = depth
        para = sum(p.numel() for p in module.parameters())
        info['#param'] = f"{para:,}" if para > 0 else "--"
        info_list.append(info)
        if not isinstance(module, ScriptModule):
            hooks[module_id] = module.register_forward_hook(
                register_hook(info))
        if (leaf_module is not None and isinstance(module, leaf_module)):
            continue
        module_items = reversed(module._modules.items())
        stack += [(f"({name}){get_name(mod)}", mod, depth + 1)
                  for name, mod in module_items if mod is not None]

    training = model.training
    model.eval()
    # make a forward pass
    with torch.no_grad():
        model(*inputs, **kwargs)
    model.train(training)

    # remove hooks
    for h in hooks.values():
        h.remove()

    info_list = postprocess(info_list)
    body = make_table(info_list, max_depth=max_depth)
    return body


def get_shape(input: Union[Any, Tuple[Any]]) -> str:
    if not isinstance(input, tuple):
        input = (input, )
    out = []
    for x in input:
        if isinstance(x, SparseTensor):
            out.append(str(list(x.sizes())))
        elif hasattr(x, 'size'):
            out.append(str(list(x.size())))
    out = ', '.join(out)
    return out


def get_name(module: nn.Module) -> str:
    return module.__class__.__name__


def postprocess(info_list: List[dict]) -> List[dict]:
    for idx, info in enumerate(info_list):
        depth = info['depth']
        if idx > 0:  # root module (0) is exclued
            if depth == 1:
                prefix = "├─"
            else:
                prefix = f"{'│    '*(depth-1)}└─"
            info['name'] = prefix + info['name']

        if info['input_shape']:
            info['input_shape'] = info['input_shape'].pop(0)
            info['output_shape'] = info['output_shape'].pop(0)
        else:
            info['input_shape'] = "--"
            info['output_shape'] = "--"
    return info_list


def make_table(info_list: List[dict], max_depth: int) -> str:
    from tabulate import tabulate
    content = [['Layer', 'Input Shape', 'Output Shape', '#Param']]
    for info in info_list:
        depth = info['depth']
        if depth > max_depth:
            continue
        row = [
            info['name'],
            info['input_shape'],
            info['output_shape'],
            info['#param'],
        ]
        content.append(row)

    return tabulate(content, headers='firstrow', tablefmt='psql')
