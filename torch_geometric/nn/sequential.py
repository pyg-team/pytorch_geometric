import copy
import inspect
import os.path as osp
import random
import sys
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import torch
from torch import Tensor

from torch_geometric.inspector import Parameter, Signature, eval_type, split
from torch_geometric.template import module_from_template


class Child(NamedTuple):
    name: str
    param_names: List[str]
    return_names: List[str]


class Sequential(torch.nn.Module):
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
        modules ([(Callable, str) or Callable]): A list of modules (with
            optional function header definitions). Alternatively, an
            :obj:`OrderedDict` of modules (and function header definitions) can
            be passed.
    """
    _children: List[Child]

    def __init__(
        self,
        input_args: str,
        modules: List[Union[Tuple[Callable, str], Callable]],
    ) -> None:
        super().__init__()

        caller_path = inspect.stack()[1].filename
        self._caller_module = osp.splitext(osp.basename(caller_path))[0]

        _globals = copy.copy(globals())
        _globals.update(sys.modules['__main__'].__dict__)
        if self._caller_module in sys.modules:
            _globals.update(sys.modules[self._caller_module].__dict__)

        signature = input_args.split('->')
        if len(signature) == 1:
            args_repr = signature[0]
            return_type_repr = 'Tensor'
            return_type = Tensor
        elif len(signature) == 2:
            args_repr = signature[0]
            return_type_repr = signature[1].strip()
            return_type = eval_type(return_type_repr, _globals)
        else:
            raise ValueError(f"Failed to parse arguments (got '{input_args}')")

        param_dict: Dict[str, Parameter] = {}
        for arg in split(args_repr, sep=','):
            signature = arg.split(':')
            if len(signature) == 1:
                name = signature[0].strip()
                param_dict[name] = Parameter(
                    name=name,
                    type=Tensor,
                    type_repr='Tensor',
                    default=inspect._empty,
                )
            elif len(signature) == 2:
                name = signature[0].strip()
                param_dict[name] = Parameter(
                    name=name,
                    type=eval_type(signature[1].strip(), _globals),
                    type_repr=signature[1].strip(),
                    default=inspect._empty,
                )
            else:
                raise ValueError(f"Failed to parse argument "
                                 f"(got '{arg.strip()}')")

        self.signature = Signature(param_dict, return_type, return_type_repr)

        if not isinstance(modules, dict):
            modules = {
                f'module_{i}': module
                for i, module in enumerate(modules)
            }
        if len(modules) == 0:
            raise ValueError(f"'{self.__class__.__name__}' expects a "
                             f"non-empty list of modules")

        self._children: List[Child] = []
        for i, (name, module) in enumerate(modules.items()):
            desc: Optional[str] = None
            if isinstance(module, (tuple, list)):
                if len(module) == 1:
                    module = module[0]
                elif len(module) == 2:
                    module, desc = module
                else:
                    raise ValueError(f"Expected tuple of length 2 "
                                     f"(got {module})")

            if i == 0 and desc is None:
                raise ValueError("Signature for first module required")
            if not callable(module):
                raise ValueError(f"Expected callable module (got {module})")
            if desc is not None and not isinstance(desc, str):
                raise ValueError(f"Expected type hint representation "
                                 f"(got {desc})")

            if desc is not None:
                signature = desc.split('->')
                if len(signature) != 2:
                    raise ValueError(
                        f"Failed to parse arguments (got '{desc}')")
                param_names = [v.strip() for v in signature[0].split(',')]
                return_names = [v.strip() for v in signature[1].split(',')]
                child = Child(name, param_names, return_names)
            else:
                param_names = self._children[-1].return_names
                child = Child(name, param_names, param_names)

            setattr(self, name, module)
            self._children.append(child)

        self._set_jittable_template()

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        for child in self._children:
            module = getattr(self, child.name)
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def __len__(self) -> int:
        return len(self._children)

    def __getitem__(self, idx: int) -> torch.nn.Module:
        return getattr(self, self._children[idx].name)

    def __setstate__(self, data: Dict[str, Any]) -> None:
        super().__setstate__(data)
        self._set_jittable_template()

    def __repr__(self) -> str:
        module_descs = [
            f"{', '.join(c.param_names)} -> {', '.join(c.return_names)}"
            for c in self._children
        ]
        module_reprs = [
            f'  ({i}) - {self[i]}: {module_descs[i]}' for i in range(len(self))
        ]
        return '{}(\n{}\n)'.format(
            self.__class__.__name__,
            '\n'.join(module_reprs),
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """"""  # noqa: D419
        value_dict = {
            name: arg
            for name, arg in zip(self.signature.param_dict.keys(), args)
        }
        for key, arg in kwargs.items():
            if key in value_dict:
                raise TypeError(f"'{self.__class__.__name__}' got multiple "
                                f"values for argument '{key}'")
            value_dict[key] = arg

        for child in self._children:
            args = [value_dict[name] for name in child.param_names]
            outs = getattr(self, child.name)(*args)
            if len(child.return_names) == 1:
                value_dict[child.return_names[0]] = outs
            else:
                for name, out in zip(child.return_names, outs):
                    value_dict[name] = out

        return outs

    # TorchScript Support #####################################################

    def _set_jittable_template(self, raise_on_error: bool = False) -> None:
        try:  # Optimize `forward()` via `*.jinja` templates:
            if ('forward' in self.__class__.__dict__ and
                    self.__class__.__dict__['forward'] != Sequential.forward):
                raise ValueError("Cannot compile custom 'forward' method")

            root_dir = osp.dirname(osp.realpath(__file__))
            uid = '%06x' % random.randrange(16**6)
            jinja_prefix = f'{self.__module__}_{self.__class__.__name__}_{uid}'
            module = module_from_template(
                module_name=jinja_prefix,
                template_path=osp.join(root_dir, 'sequential.jinja'),
                tmp_dirname='sequential',
                # Keyword arguments:
                modules=[self._caller_module],
                signature=self.signature,
                children=self._children,
            )

            self.forward = module.forward.__get__(self)

            # NOTE We override `forward` on the class level here in order to
            # support `torch.jit.trace` - this is generally dangerous to do,
            # and limits `torch.jit.trace` to a single `Sequential` module:
            self.__class__.forward = module.forward
        except Exception as e:  # pragma: no cover
            if raise_on_error:
                raise e

    def __prepare_scriptable__(self) -> 'Sequential':
        # Prevent type sharing when scripting `Sequential` modules:
        type_store = torch.jit._recursive.concrete_type_store.type_store
        type_store.pop(self.__class__, None)
        return self
