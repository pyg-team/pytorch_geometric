import os
import os.path as osp
from typing import Callable, List, Tuple, Union
from uuid import uuid1

import torch

from torch_geometric.nn.conv.utils.jit import class_from_module_repr


def Sequential(
    input_args: str,
    modules: List[Union[Tuple[Callable, str], Callable]],
) -> torch.nn.Module:
    r"""An extension of the :class:`torch.nn.Sequential` container in order to
    define a sequential GNN model.
    Since GNN operators take in multiple input arguments,
    :class:`torch_geometric.nn.Sequential` expects both global input
    arguments, and function header definitions of individual operators.
    If omitted, an intermediate module will operate on the *output* of its
    preceding module:

    .. code-block:: python

        from torch.nn import Linear, ReLU
        from torch_geometric.nn import Sequential, GCNConv

        model = Sequential('x, edge_index', [
            (GCNConv(in_channels, 64), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            ReLU(inplace=True),
            Linear(64, out_channels),
        ])

    where :obj:`'x, edge_index'` defines the input arguments of :obj:`model`,
    and :obj:`'x, edge_index -> x'` defines the function header, *i.e.* input
    arguments *and* return types, of :class:`~torch_geometric.nn.conv.GCNConv`.

    In particular, this also allows to create more sophisticated models,
    such as utilizing :class:`~torch_geometric.nn.models.JumpingKnowledge`:

    .. code-block:: python

        from torch.nn import Linear, ReLU, Dropout
        from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
        from torch_geometric.nn import global_mean_pool

        model = Sequential('x, edge_index, batch', [
            (Dropout(p=0.5), 'x -> x'),
            (GCNConv(dataset.num_features, 64), 'x, edge_index -> x1'),
            ReLU(inplace=True),
            (GCNConv(64, 64), 'x1, edge_index -> x2'),
            ReLU(inplace=True),
            (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
            (JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
            (global_mean_pool, 'x, batch -> x'),
            Linear(2 * 64, dataset.num_classes),
        ])

    Args:
        input_args (str): The input arguments of the model.
        modules ([(str, Callable) or Callable]): A list of modules (with
            optional function header definitions). Alternatively, an
            :obj:`OrderedDict` of modules (and function header definitions) can
            be passed.
    """
    try:
        from jinja2 import Template
    except ImportError:
        raise ModuleNotFoundError(
            "No module named 'jinja2' found on this machine. "
            "Run 'pip install jinja2' to install the library.")

    input_args = [x.strip() for x in input_args.split(',')]

    if not isinstance(modules, dict):
        modules = {f'module_{i}': module for i, module in enumerate(modules)}

    # We require the first entry of the input list to define arguments:
    assert len(modules) > 0
    first_module = list(modules.values())[0]
    assert isinstance(first_module, (tuple, list))

    # A list holding the callable function and the input and output names:
    calls: List[Tuple[str, Callable, List[str], List[str]]] = []

    for name, module in modules.items():
        if isinstance(module, (tuple, list)) and len(module) >= 2:
            module, desc = module[:2]
            in_desc, out_desc = parse_desc(desc)
        elif isinstance(module, (tuple, list)):
            module = module[0]
            in_desc = out_desc = calls[-1][-1]
        else:
            in_desc = out_desc = calls[-1][-1]

        calls.append((name, module, in_desc, out_desc))

    root = os.path.dirname(osp.realpath(__file__))
    with open(osp.join(root, 'sequential.jinja'), 'r') as f:
        template = Template(f.read())

    cls_name = f'Sequential_{uuid1().hex[:6]}'
    module_repr = template.render(
        cls_name=cls_name,
        input_args=input_args,
        calls=calls,
    )

    # Instantiate a class from the rendered module representation.
    module = class_from_module_repr(cls_name, module_repr)()
    module._names = list(modules.keys())
    for name, submodule, _, _ in calls:
        setattr(module, name, submodule)
    return module


def parse_desc(desc: str) -> Tuple[List[str], List[str]]:
    in_desc, out_desc = desc.split('->')
    in_desc = [x.strip() for x in in_desc.split(',')]
    out_desc = [x.strip() for x in out_desc.split(',')]
    return in_desc, out_desc
