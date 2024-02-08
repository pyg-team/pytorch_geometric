import os.path as osp
import random
from typing import Callable, List, NamedTuple, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.inspector import split, type_repr
from torch_geometric.template import module_from_template


class Child(NamedTuple):
    name: str
    module: Callable
    param_names: List[str]
    return_names: List[str]


def Sequential(
    input_args: str,
    modules: List[Union[Tuple[Callable, str], Callable]],
) -> torch.nn.Module:
    r"""An extension of the :class:`torch.nn.Sequential` container in order to
    define a sequential GNN model.

    Since GNN operators take in multiple input arguments,
    :class:`torch_geometric.nn.Sequential` additionally expects both global
    input arguments, and function header definitions of individual operators.
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

    Here, :obj:`'x, edge_index'` defines the input arguments of :obj:`model`,
    and :obj:`'x, edge_index -> x'` defines the function header, *i.e.* input
    arguments *and* return types of :class:`~torch_geometric.nn.conv.GCNConv`.

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
    signature = input_args.split('->')
    if len(signature) == 1:
        input_args = signature[0]
        return_type = type_repr(Tensor, globals())
    elif len(signature) == 2:
        input_args, return_type = signature[0], signature[1].strip()
    else:
        raise ValueError(f"Failed to parse arguments (got '{input_args}')")

    input_types = split(input_args, sep=',')
    if len(input_types) == 0:
        raise ValueError(f"Failed to parse arguments (got '{input_args}')")

    if not isinstance(modules, dict):
        modules = {f'module_{i}': module for i, module in enumerate(modules)}
    if len(modules) == 0:
        raise ValueError("'Sequential' expected a non-empty list of modules")

    children: List[Child] = []
    for i, (name, module) in enumerate(modules.items()):
        desc: Optional[str] = None
        if isinstance(module, (tuple, list)):
            if len(module) == 1:
                module = module[0]
            elif len(module) == 2:
                module, desc = module
            else:
                raise ValueError(f"Expected tuple of length 2 (got {module})")

        if i == 0 and desc is None:
            raise ValueError("Requires signature for first module")
        if not callable(module):
            raise ValueError(f"Expected callable module (got {module})")
        if desc is not None and not isinstance(desc, str):
            raise ValueError(f"Expected type hint representation (got {desc})")

        if desc is not None:
            signature = desc.split('->')
            if len(signature) != 2:
                raise ValueError(f"Failed to parse arguments (got '{desc}')")
            param_names = [v.strip() for v in signature[0].split(',')]
            return_names = [v.strip() for v in signature[1].split(',')]
            child = Child(name, module, param_names, return_names)
        else:
            param_names = children[-1].return_names
            child = Child(name, module, param_names, param_names)

        children.append(child)

    uid = '%06x' % random.randrange(16**6)
    root_dir = osp.dirname(osp.realpath(__file__))
    module = module_from_template(
        module_name=f'torch_geometric.nn.sequential_{uid}',
        template_path=osp.join(root_dir, 'sequential.jinja'),
        tmp_dirname='sequential',
        # Keyword arguments:
        input_types=input_types,
        return_type=return_type,
        children=children,
    )

    model = module.Sequential()
    model._module_names = [child.name for child in children]
    model._module_descs = [
        f"{', '.join(child.param_names)} -> {', '.join(child.return_names)}"
        for child in children
    ]
    for child in children:
        setattr(model, child.name, child.module)

    return model
