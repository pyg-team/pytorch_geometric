from typing import Tuple, Dict, Union, Optional, Any, OrderedDict
from torch_geometric.typing import NodeType, EdgeType, Metadata

import re
import copy
import warnings

import torch
from torch.nn import Module, ModuleList, ModuleDict, Sequential

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.fx import (symbolic_trace, get_submodule,
                                   find_node_by_name, find_node_by_target,
                                   erase_unused_nodes)

try:
    from torch.fx import GraphModule, Graph, Node
except ImportError:
    GraphModule = Graph = Node = None

# TODO:
# * LazyInitialization - Bug: LazyLinear.weight does not support deepcopy yet


def to_hetero(module: Module, metadata: Metadata, aggr: str = 'sum',
              input_map: Optional[Dict[str, str]] = None,
              debug: bool = False) -> Module:

    gm = symbolic_trace(module)

    if debug:
        gm.graph.print_tabular()
        print()
        print(gm.graph.python_code('self'))

    for node in list(gm.graph.nodes):
        if node.op == 'placeholder':
            unwrap_placeholder(gm.graph, node, metadata, input_map=input_map)
        elif node.op == 'get_attr':
            raise NotImplementedError
        elif (node.op == 'call_module' and isinstance(
                get_submodule(module, node.target), MessagePassing)):
            unwrap_call_mp_module(gm.graph, node, metadata, aggr)
        elif node.op == 'call_module':
            unwrap_call_module(gm.graph, node, metadata)
        elif node.op == 'call_method':
            unwrap_call_method(gm.graph, node, metadata)
        elif node.op == 'call_function':
            unwrap_call_function(gm.graph, node, metadata)
        elif node.op == 'output':
            unwrap_output(gm.graph, node, metadata)
        else:
            raise NotImplementedError

    erase_unused_nodes(gm.graph)

    if debug:
        gm.graph.print_tabular()
        print()
        print(gm.graph.python_code('self'))

    gm.graph.lint()
    gm.recompile()
    gm._modules = duplicate_submodules(module._modules, gm.graph, metadata)

    return gm


def unwrap_placeholder(graph: Graph, node: Node, metadata: Metadata,
                       input_map: Optional[Dict[str, str]] = None):

    input_type = (input_map or {}).get(node.name, None)
    if input_type is None and bool(re.search('(edge|adj)', node.name)):
        input_type = 'edge'
    is_edge_level_placeholder = input_type == 'edge'

    # if node.type is not None:
    #     Type = EdgeType if is_edge_level_placeholder else NodeType
    #     node.type = Dict[Type, node.type]

    graph.inserting_after(node)
    for key in metadata[int(is_edge_level_placeholder)]:
        out = graph.create_node('call_method', target='get', args=(node, key),
                                name=f'{node.name}__{key2str(key)}')
        graph.inserting_after(out)


def unwrap_call_mp_module(graph: Graph, node: Node, metadata: Metadata,
                          aggr: str):

    # Group edge-wise keys per destination:
    key_name, keys_per_dst = {}, {}
    for key in metadata[1]:
        keys_per_dst[key[-1]] = keys_per_dst.get(key[-1], []) + [key]
        key_name[key] = f'{node.name}__{key[-1]}{len(keys_per_dst[key[-1]])}'

    for dst, keys in copy.copy(keys_per_dst).items():
        if len(keys) == 1:
            # In case there is only a single edge-wise connection, there is no
            # need for any destination-wise aggregation, and we can already set
            # the intermediate variable name to the final output name.
            key_name[keys[0]] = f'{node.name}__{dst}'
            del keys_per_dst[dst]

    graph.inserting_after(node)
    for key in metadata[1]:
        args, kwargs = map_args(graph, node, key)
        out = graph.create_node('call_module',
                                target=f'{node.target}.{key2str(key)}',
                                args=args, kwargs=kwargs, name=key_name[key])
        graph.inserting_after(out)

    # Perform destination-wise aggregation.
    for dst, keys in keys_per_dst.items():
        keys = [key_name[key] for key in keys]
        i = len(keys) + 1
        while len(keys) >= 2:
            args = (find_node_by_name(graph, keys[-2]),
                    find_node_by_name(graph, keys[-1]))
            out = graph.create_node(
                'call_function', target=torch.add, args=args,
                name=f'{node.name}__{dst}{i if len(keys) > 2 else ""}')
            graph.inserting_after(out)
            keys = keys[:-2] + [f'{node.name}__{dst}{i}']
            i += 1


def unwrap_call_module(graph: Graph, node: Node, metadata: Metadata):
    graph.inserting_after(node)
    for key in metadata[int(has_edge_level_arg(graph, node, metadata))]:
        args, kwargs = map_args(graph, node, key)
        out = graph.create_node('call_module',
                                target=f'{node.target}.{key2str(key)}',
                                args=args, kwargs=kwargs,
                                name=f'{node.name}__{key2str(key)}')
        graph.inserting_after(out)


def unwrap_call_method(graph: Graph, node: Node, metadata: Metadata):
    graph.inserting_after(node)
    for key in metadata[int(has_edge_level_arg(graph, node, metadata))]:
        args, kwargs = map_args(graph, node, key)
        out = graph.create_node('call_method', target=node.target, args=args,
                                kwargs=kwargs,
                                name=f'{node.name}__{key2str(key)}')
        graph.inserting_after(out)


def unwrap_call_function(graph: Graph, node: Node, metadata: Metadata):
    graph.inserting_after(node)
    for key in metadata[int(has_edge_level_arg(graph, node, metadata))]:
        args, kwargs = map_args(graph, node, key)
        out = graph.create_node('call_function', target=node.target, args=args,
                                kwargs=kwargs,
                                name=f'{node.name}__{key2str(key)}')
        graph.inserting_after(out)


def unwrap_output(graph: Graph, node: Node, metadata: Metadata):
    def _unwrap_output(v: Any) -> Any:
        if isinstance(v, Node):
            return {
                k: find_node_by_name(graph, f'{v.name}__{key2str(k)}')
                for k in metadata[int(is_edge_level_node(graph, v, metadata))]
            }
        elif isinstance(v, dict):
            return {key: _unwrap_output(value) for key, value in v.items()}
        elif isinstance(v, list):
            return [_unwrap_output(value) for value in v]
        elif isinstance(v, tuple):
            return tuple(_unwrap_output(value) for value in v)
        else:
            return v

    # if node.type is not None and isinstance(node.args[0], Node):
    #     output = node.args[0]
    #     is_edge_level_output = is_edge_level_node(graph, output, metadata)
    #     Type = EdgeType if is_edge_level_output else NodeType
    #     node.type = Dict[Type, node.type]

    # elif (node.type is not None and isinstance(node.args[0], tuple)
    #       and hasattr(node.type, '__origin__')
    #       and issubclass(node.type.__origin__, tuple)):

    #     __args__ = []
    #     for output, _type in zip(node.args[0], node.type.__args__):
    #         is_edge_level_output = is_edge_level_node(graph, output, metadat)
    #         Type = EdgeType if is_edge_level_output else NodeType
    #         __args__ += [Dict[Type, _type]]
    #     node.type.__args__ = tuple(__args__)

    # else:
    #     node.type = None

    node.args = (_unwrap_output(node.args[0]), )


def duplicate_submodules(modules: OrderedDict[str, Module], graph: Graph,
                         metadata: Metadata):
    def _duplicate_submodule(module: Module, target: str):
        if isinstance(module, ModuleList) or isinstance(module, Sequential):
            return ModuleList([
                _duplicate_submodule(v, f'{target}.{i}')
                for i, v in enumerate(module)
            ])
        elif isinstance(module, ModuleDict):
            return ModuleDict({
                k: _duplicate_submodule(v, f'{target}.{k}')
                for k, v in module.items()
            })
        else:
            has_edge_level_target = bool(
                find_node_by_target(graph,
                                    f'{target}.{key2str(metadata[1][0])}'))

            module_dict = torch.nn.ModuleDict()
            for key in metadata[int(has_edge_level_target)]:
                module_dict[key2str(key)] = copy.deepcopy(module)
                if hasattr(module, 'reset_parameters'):
                    module_dict[key2str(key)].reset_parameters()
                else:
                    warnings.warn((f"'{target}' will be duplicated, but its "
                                   "parameters cannot be reset"))
            return module_dict

    for target, module in copy.copy(modules).items():
        modules[target] = _duplicate_submodule(module, target)
    return modules


# Helper function #############################################################


def key2str(metatype: Union[str, Tuple[str, str, str]]) -> str:
    return '__'.join(metatype) if isinstance(metatype, tuple) else metatype


def map_args(graph: Graph, node: Node, key: Union[NodeType, EdgeType]):
    def _map_arg(v: Any) -> Any:
        if isinstance(v, Node):
            out = find_node_by_name(graph, f'{v.name}__{key2str(key)}')
            if out is None and isinstance(key, tuple):
                out = (find_node_by_name(graph, f'{v.name}__{key[0]}'),
                       find_node_by_name(graph, f'{v.name}__{key[-1]}'))
            return out
        elif isinstance(v, dict):
            return {key: _map_arg(value) for key, value in v.items()}
        elif isinstance(v, list):
            return [_map_arg(value) for value in v]
        elif isinstance(v, tuple):
            return tuple(_map_arg(value) for value in v)
        else:
            return v

    args = tuple(_map_arg(v) for v in node.args)
    kwargs = {k: _map_arg(v) for k, v in node.kwargs.items()}
    return args, kwargs


def is_edge_level_node(graph: Graph, node: Node, metadata: Metadata) -> bool:
    return bool(
        find_node_by_name(graph, f'{node.name}__{key2str(metadata[1][0])}'))


def has_edge_level_arg(graph: Graph, node: Node, metadata: Metadata) -> bool:
    def _has_edge_level_arg(v: Any) -> bool:
        if isinstance(v, Node):
            return is_edge_level_node(graph, v, metadata)
        elif isinstance(v, dict):
            return any([_has_edge_level_arg(value) for value in v.values()])
        elif isinstance(v, (list, tuple)):
            return any([_has_edge_level_arg(value) for value in v])
        else:
            return False

    return (any([_has_edge_level_arg(v) for v in node.args])
            or any([_has_edge_level_arg(v) for v in node.kwargs.values()]))
