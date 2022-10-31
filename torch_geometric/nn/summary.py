from collections import defaultdict
from time import perf_counter
from typing import Any, Sequence

import torch
import torch.nn as nn


def summary(
    model: nn.Module,
    *inputs: Sequence[Any],
    device: torch.device = 'cpu',
    max_depth: int = 3,
) -> str:
    r"""Summarizes the given PyTorch model. Summarized information includes
    (1) Layer names, (2) Input/output shapes,
    (3) # of parameters, and (4) Excutation time.

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
        device (torch.device): Device for model and input_data.
            (default: :obj:`"cpu"`)
        max_depth (int): Depth of nested layers to display.
            Nested layers below this depth will not be displayed
            in the summary. (default: :obj:`"3"`)
    """
    def register_pre_hook(info):
        def hook(module, input):
            info['input_time'].append(perf_counter())

        return hook

    def register_hook(info):
        def hook(module, input, output):
            info['input_shape'].append(get_shape(input))
            info['output_shape'].append(get_shape(output))
            info['output_time'].append(perf_counter())

        return hook

    hooks = {}
    depth = 0
    stack = [(get_name(model), model, depth)]

    info_list = []
    input_shape = defaultdict(list)
    output_shape = defaultdict(list)
    intput_time = defaultdict(list)
    output_time = defaultdict(list)
    while stack:
        var_name, module, depth = stack.pop()
        module_id = id(module)
        if module_id in hooks:
            for hook in hooks[module_id]:
                hook.remove()

        info = {}
        info['name'] = var_name
        info['layer'] = module
        info['input_shape'] = input_shape[id(module)]
        info['output_shape'] = output_shape[id(module)]
        info['input_time'] = intput_time[id(module)]
        info['output_time'] = output_time[id(module)]
        info['depth'] = depth
        para = sum(p.numel() for p in module.parameters())
        info['#param'] = f"{para:,}" if para > 0 else "--"
        info_list.append(info)
        hooks[module_id] = (
            module.register_forward_pre_hook(register_pre_hook(info)),
            module.register_forward_hook(register_hook(info)),
        )
        module_items = reversed(module._modules.items())
        stack += [(f"({name}){get_name(mod)}", mod, depth + 1)
                  for name, mod in module_items if mod is not None]

    # set `eval` mode
    model.eval()
    # make a forward pass
    device = torch.device(device)
    with torch.no_grad():
        inputs = [x.to(device) if hasattr(x, 'to') else x for x in inputs]
        model.to(device)(*inputs)

    # remove hooks
    for hs in hooks.values():
        for h in hs:
            h.remove()

    for info in info_list:
        depth = info['depth']
        if info['layer'] is not model:
            if depth == 1:
                prefix = "├─"
            else:
                prefix = f"{'│    '*(depth-1)}└─"
            info['name'] = prefix + info['name']
            if info['input_shape']:
                info['input_shape'] = info['input_shape'].pop(0)
                info['output_shape'] = info['output_shape'].pop(0)
                duration = info['output_time'].pop(0) - info['input_time'].pop(
                    0)
                info['duration'] = f"{duration:.4f}"
            else:
                info['input_shape'] = "--"
                info['output_shape'] = "--"
                info['duration'] = "--"
        else:
            info['input_shape'] = info['input_shape'].pop()
            info['output_shape'] = info['output_shape'].pop()
            duration = info['output_time'].pop(0) - info['input_time'].pop(0)
            info['duration'] = f"{duration:.4f}"

    # make a table
    from tabulate import tabulate
    content = [['Layer', 'Input Shape', 'Output Shape', '#Param', 'Time']]
    for info in info_list:
        depth = info['depth']
        if depth > max_depth:
            continue
        row = [
            info['name'], info['input_shape'], info['output_shape'],
            info['#param'], info['duration']
        ]
        content.append(row)

    body = tabulate(content, headers='firstrow', tablefmt='psql')

    return body


def get_shape(input):
    if not isinstance(input, tuple):
        input = (input, )
    out = []
    for x in input:
        if hasattr(x, 'size'):
            out.append(str(list(x.size())))
    out = ', '.join(out)
    return out


def get_name(module):
    return str(module.__class__).split(".")[-1].split("'")[0]
