from typing import Tuple, List, Dict, Union, Optional, Any

import re
import copy
import inspect
import warnings
from itertools import chain

import torch
from torch.nn import Module
from torch import fx
from torch_geometric.nn import MessagePassing

NodeType = str
EdgeType = Tuple[str, str, str]
Metadata = Tuple[List[NodeType], List[EdgeType]]

# TODO:
# * Modules with multiple return statements => how to infer node or edge type
# * LazyInitialization - Bug: LazyLinear.weight does not support deepcopy yet
# * What are leaf modules?
# * Flexible args/kwargs replacement
# * What happens in ModuleLists and ModuleDicts?


def transform(module: Module, metadata: Metadata,
              input_map: Optional[Dict[str, str]] = None,
              debug: bool = True) -> Module:

    state = init_state(module, input_map)

    gm = fx.symbolic_trace(module)
    gm = duplicate_submodules(gm, metadata)

    if debug:
        gm.graph.print_tabular()
        print()
        print(gm.graph.python_code('self'))

    for node in list(gm.graph.nodes):
        print(node.op)
        if node.op == 'placeholder':
            unwrap_placeholder(gm.graph, node, state, metadata)
        elif node.op == 'output':
            unwrap_output(gm.graph, node, state, metadata)
        elif node.op == 'get_attr':
            raise NotImplementedError
        elif node.op == 'call_module':
            unwrap_call_module(gm.graph, node, state, metadata)
        elif node.op == 'call_method':
            unwrap_call_method(gm.graph, node, state, metadata)
        elif node.op == 'call_function':
            unwrap_call_function(gm.graph, node, state, metadata)
        else:
            raise NotImplementedError

    erase_unused_nodes(gm.graph)

    if debug:
        gm.graph.print_tabular()
        print()
        print(gm.graph.python_code('self'))

    gm.graph.lint()
    gm.recompile()

    return gm


def init_state(module: Module,
               input_map: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    out = {}
    for key in inspect.signature(module.forward).parameters.keys():
        if input_map is not None and key in input_map:
            out[key] = input_map[key]
            assert input_map[key] in ['node', 'edge']
        else:
            out[key] = 'edge' if bool(re.search('(edge|adj)', key)) else 'node'
    return out


def key2str(metatype: Union[str, Tuple[str, str, str]]) -> str:
    return '__'.join(metatype) if isinstance(metatype, tuple) else metatype


def duplicate_submodules(module: Module, metadata: Metadata) -> Module:
    # TODO: Handle `torch.nn.ModuleList` and `torch.nn.ModuleDict`
    metadata = metadata[:1] + ([key2str(t) for t in metadata[1]], )

    for name, submodule in dict(module.named_children()).items():
        module_dict = torch.nn.ModuleDict()

        is_message_passing = isinstance(submodule, MessagePassing)
        for metatype in metadata[1] if is_message_passing else metadata[0]:
            module_dict[metatype] = copy.deepcopy(submodule)
            if hasattr(submodule, 'reset_parameters'):
                module_dict[metatype].reset_parameters()
            else:
                warnings.warn((f"'{name}' will be duplicated, but its "
                               "parameters cannot be reset"))

        module._modules[name] = module_dict

    return module


def get_type(name: str, state: Dict[str, str]) -> Union[NodeType, EdgeType]:
    return NodeType if state[name] == 'node' else EdgeType


def find_node(graph: fx.Graph, name: str) -> Optional[fx.Node]:
    out: Optional[fx.Node] = None
    for node in graph.nodes:
        if node.name == name:
            out = node
    return out


def erase_unused_nodes(graph: fx.Graph) -> fx.Graph:
    for node in list(graph.nodes)[::-1]:  # Iterate in reverse-mode.
        try:
            if node.op not in ['placeholder', 'output']:
                graph.erase_node(node)
        except RuntimeError:
            pass
    return graph


def unwrap_placeholder(graph: fx.Graph, node: fx.Node, state: Dict[str, str],
                       metadata: Metadata) -> List[fx.Node]:

    outs: List[fx.Node] = []
    if node.type is not None:
        node.type = Dict[get_type(node.name, state), node.type]
    graph.inserting_after(node)
    for key in metadata[0] if state[node.name] == 'node' else metadata[1]:
        out = graph.create_node('call_method', 'get', args=(node, key),
                                name=f'{node.name}__{key2str(key)}')
        graph.inserting_after(out)
        outs.append(out)
    return outs


def unwrap_output(graph: fx.Graph, node: fx.Node, state: Dict[str, str],
                  metadata: Metadata):
    def _get_mapping(name):
        return {
            key: find_node(graph, f'{name}__{key2str(key)}')
            for key in (metadata[0] if state[name] == 'node' else metadata[1])
        }

    if isinstance(node.args[0], fx.Node):  # Single output value.
        if node.type is not None:
            node.type = Dict[get_type(node.args[0].name, state), node.type]
        node.args = (_get_mapping(node.args[0].name), )

    elif isinstance(node.args[0], tuple):  # Multiple output values.
        if node.type is not None:
            if (node.type.__dict__.get('__origin__', None) is not None
                    and issubclass(node.type.__dict__['__origin__'], tuple)):
                node.type.__dict__['__args__'] = tuple(
                    Dict[get_type(arg.name, state), t] for arg, t in zip(
                        node.args[0], node.type.__dict__['__args__']))
            else:
                node.type = None
        node.args = (tuple(_get_mapping(arg.name) for arg in node.args[0]), )

    else:
        raise NotImplementedError


def unwrap_call_module(graph: fx.Graph, node: fx.Node, state: Dict[str, str],
                       metadata: Metadata) -> List[fx.Node]:

    outs: List[fx.Node] = []
    graph.inserting_after(node)

    it = chain(node.args, node.kwargs.values())
    if all([state[v.name] == 'node' for v in it]):
        for key in metadata[0]:
            args = tuple(
                find_node(graph, f'{v.name}__{key}') for v in node.args)
            kwargs = {
                k: find_node(graph, f'{v.name}__{key}')
                for k, v in node.kwargs.items()
            }
            out = graph.create_node('call_module', f'{node.target}.{key}',
                                    args, kwargs, name=f'{node.name}__{key}')
            state[node.name] = 'node'
            graph.inserting_after(out)
            outs.append(out)

    elif all([state[v.name] == 'edge' for v in it]):
        raise NotImplementedError

    else:
        raise NotImplementedError

    return outs


def unwrap_call_method(graph: fx.Graph, node: fx.Node, state: Dict[str, str],
                       metadata: Metadata) -> List[fx.Node]:

    outs: List[fx.Node] = []
    graph.inserting_after(node)

    it = chain(node.args, node.kwargs.values())
    if all([state[v.name] == 'node' for v in it]):
        for key in metadata[0]:
            args = tuple(
                find_node(graph, f'{v.name}__{key}') for v in node.args)
            kwargs = {
                k: find_node(graph, f'{v.name}__{key}')
                for k, v in node.kwargs.items()
            }
            out = graph.create_node('call_method', node.target, args, kwargs,
                                    name=f'{node.name}__{key}')
            state[node.name] = 'node'
            graph.inserting_after(out)
            outs.append(out)

    elif all([state[v.name] == 'edge' for v in it]):
        raise NotImplementedError

    else:
        raise NotImplementedError

    return outs


def unwrap_call_function(graph: fx.Graph, node: fx.Node, state: Dict[str, str],
                         metadata: Metadata) -> List[fx.Node]:

    outs: List[fx.Node] = []
    graph.inserting_after(node)

    it = chain(node.args, node.kwargs.values())
    if all([state[v.name] == 'node' for v in it]):
        for key in metadata[0]:
            args = tuple(
                find_node(graph, f'{v.name}__{key}') for v in node.args)
            kwargs = {
                k: find_node(graph, f'{v.name}__{key}')
                for k, v in node.kwargs.items()
            }
            out = graph.create_node('call_function', node.target, args, kwargs,
                                    name=f'{node.name}__{key}')
            state[node.name] = 'node'
            graph.inserting_after(out)
            outs.append(out)

    elif all([state[v.name] == 'edge' for v in it]):
        raise NotImplementedError

    else:
        raise NotImplementedError

    return outs
