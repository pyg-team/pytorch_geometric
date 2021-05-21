from typing import Tuple, Dict, Union, Optional
from torch_geometric.typing import NodeType, EdgeType, Metadata

import copy
import warnings

import torch
from torch.nn import Module
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.fx import (symbolic_trace, set_node_hints_,
                                   find_node_by_name, erase_unused_nodes)

try:
    from torch.fx import Graph, Node
except ImportError:
    Graph = Node = None

# TODO:
# * Modules with multiple return statements => how to infer node or edge type
# * LazyInitialization - Bug: LazyLinear.weight does not support deepcopy yet
# * What happens in ModuleLists and ModuleDicts?


def to_hetero(module: Module, metadata: Metadata,
              input_map: Optional[Dict[str, str]] = None,
              debug: bool = False) -> Module:

    gm = symbolic_trace(module)
    set_node_hints_(gm.graph, input_map)
    duplicate_submodules_(gm, metadata)

    if debug:
        gm.graph.print_tabular()
        print(gm.graph.python_code('self'))

    for node in list(gm.graph.nodes):
        if node.op == 'placeholder':
            unwrap_placeholder(gm.graph, node, metadata)
        elif node.op == 'output':
            unwrap_output(gm.graph, node, metadata)
        elif node.op == 'get_attr':
            raise NotImplementedError
        elif node.op == 'call_module':
            unwrap_call_module(gm.graph, node, metadata)
        elif node.op == 'call_method':
            unwrap_call_method(gm.graph, node, metadata)
        elif node.op == 'call_function':
            unwrap_call_function(gm.graph, node, metadata)
        else:
            raise NotImplementedError

    erase_unused_nodes(gm.graph)

    if debug:
        gm.graph.print_tabular()
        print(gm.graph.python_code('self'))

    gm.graph.lint()
    gm.recompile()

    return gm


def key2str(metatype: Union[str, Tuple[str, str, str]]) -> str:
    return '__'.join(metatype) if isinstance(metatype, tuple) else metatype


def get_type_hint(node: Node) -> Union[NodeType, EdgeType]:
    return NodeType if node.hint == 'node' else EdgeType


def duplicate_submodules_(module: Module, metadata: Metadata):
    # TODO: Handle `torch.nn.ModuleList` and `torch.nn.ModuleDict`
    for name, submodule in dict(module.named_children()).items():
        module_dict = torch.nn.ModuleDict()

        is_message_passing = isinstance(submodule, MessagePassing)
        for metatype in metadata[1] if is_message_passing else metadata[0]:
            metatype = key2str(metatype)
            module_dict[metatype] = copy.deepcopy(submodule)
            if hasattr(submodule, 'reset_parameters'):
                module_dict[metatype].reset_parameters()
            else:
                warnings.warn((f"'{name}' will be duplicated, but its "
                               "parameters cannot be reset"))

        module._modules[name] = module_dict


def unwrap_placeholder(graph: Graph, node: Node, metadata: Metadata):
    if node.type is not None:
        node.type = Dict[get_type_hint(node), node.type]
    graph.inserting_after(node)
    for key in metadata[0] if node.hint == 'node' else metadata[1]:
        out = graph.create_node('call_method', target='get', args=(node, key),
                                name=f'{node.name}__{key2str(key)}')
        graph.inserting_after(out)


def unwrap_output(graph: Graph, node: Node, metadata: Metadata):
    def _unwrap_output(arg: Node) -> Dict[Union[NodeType, EdgeType], Node]:
        return {
            key: find_node_by_name(graph, f'{arg.name}__{key2str(key)}')
            for key in (metadata[0] if arg.hint == 'node' else metadata[1])
        }

    if isinstance(node.args[0], Node):  # Single output value.
        if node.type is not None:
            node.type = Dict[get_type_hint(node.args[0]), node.type]
        node.args = (_unwrap_output(node.args[0]), )

    elif isinstance(node.args[0], tuple):  # Multiple output values.
        if node.type is not None:
            if (hasattr(node.type, '__origin__')
                    and issubclass(node.type.__origin__, tuple)):
                node.type.__args__ = tuple(
                    Dict[get_type_hint(v), t]
                    for v, t in zip(node.args[0], node.type.__args__))
            else:
                node.type = None
        node.args = (tuple(_unwrap_output(v) for v in node.args[0]), )

    else:
        raise NotImplementedError


def unwrap_call_module(graph: Graph, node: Node, metadata: Metadata):
    graph.inserting_after(node)

    it = list(node.args) + list(node.kwargs.values())
    node_hints = [v.hint == 'node' for v in it if isinstance(v, Node)]
    edge_hints = [v.hint == 'edge' for v in it if isinstance(v, Node)]

    if all(node_hints) or all(edge_hints):
        for key in metadata[0] if all(node_hints) else metadata[1]:
            args = tuple(
                find_node_by_name(graph, f'{v.name}__{key2str(key)}')
                for v in node.args)
            kwargs = {
                k: find_node_by_name(graph, f'{v.name}__{key2str(key)}')
                for k, v in node.kwargs.items()
            }
            out = graph.create_node('call_module',
                                    target=f'{node.target}.{key2str(key)}',
                                    args=args, kwargs=kwargs,
                                    name=f'{node.name}__{key2str(key)}')
            graph.inserting_after(out)

    else:  # Message passing.
        key2name, keys_per_dst = {}, {}
        for key in metadata[1]:
            key2name[key] = f'{node.name}__{key2str(key)}'
            keys_per_dst[key[-1]] = keys_per_dst.get(key[-1], []) + [key]

        for dst, keys in copy.copy(keys_per_dst).items():
            if len(keys) == 1:
                # In case there is only a single connection, there is no need
                # for any destination-wise aggregation, and we set the
                # resulting variable name to the final value.
                key2name[keys[0]] = f'{node.name}__{dst}'
                del keys_per_dst[dst]

        for key in metadata[1]:  # Add message passing call per edge type.
            args = ()
            for v in node.args:
                if v.hint == 'node':
                    w = (find_node_by_name(graph, f'{v.name}__{key[0]}'),
                         find_node_by_name(graph, f'{v.name}__{key[-1]}'))
                else:
                    w = find_node_by_name(graph, f'{v.name}__{key2str(key)}')
                args += (w, )

            kwargs = {}
            for k, v in node.kwargs.items():
                if v.hint == 'node':
                    w = (find_node_by_name(graph, f'{v.name}__{key[0]}'),
                         find_node_by_name(graph, f'{v.name}__{key[-1]}'))
                else:
                    w = find_node_by_name(graph, f'{v.name}__{key2str(key)}')
                kwargs[k] = w

            out = graph.create_node('call_module',
                                    target=f'{node.target}.{key2str(key)}',
                                    args=args, kwargs=kwargs,
                                    name=key2name[key])
            graph.inserting_after(out)

        # Perform destination-wise aggregation.
        for dst, keys in keys_per_dst.items():
            keys = [f'{node.name}__{key2str(key)}' for key in keys]
            while len(keys) >= 2:
                i = len(keys) - 2
                args = (find_node_by_name(graph, keys[-2]),
                        find_node_by_name(graph, keys[-1]))
                out = graph.create_node(
                    'call_function', target=torch.add, args=args,
                    name=f'{node.name}__{dst}{i if i > 0 else ""}')
                graph.inserting_after(out)
                keys = keys[:-2] + [f'{node.name}__{dst}{i}']


def unwrap_call_method(graph: Graph, node: Node, metadata: Metadata):
    graph.inserting_after(node)

    it = list(node.args) + list(node.kwargs.values())
    node_hints = [v.hint == 'node' for v in it if isinstance(v, Node)]
    edge_hints = [v.hint == 'edge' for v in it if isinstance(v, Node)]

    if all(node_hints) or all(edge_hints):
        for key in metadata[0] if all(node_hints) else metadata[1]:
            args = tuple(
                find_node_by_name(graph, f'{v.name}__{key2str(key)}')
                for v in node.args)
            kwargs = {
                k: find_node_by_name(graph, f'{v.name}__{key2str(key)}')
                for k, v in node.kwargs.items()
            }
            out = graph.create_node('call_method', target=node.target,
                                    args=args, kwargs=kwargs,
                                    name=f'{node.name}__{key2str(key)}')
            graph.inserting_after(out)

    else:
        raise NotImplementedError


def unwrap_call_function(graph: Graph, node: Node, metadata: Metadata):
    graph.inserting_after(node)

    it = list(node.args) + list(node.kwargs.values())
    node_hints = [v.hint == 'node' for v in it if isinstance(v, Node)]
    edge_hints = [v.hint == 'edge' for v in it if isinstance(v, Node)]

    if all(node_hints) or all(edge_hints):
        for key in metadata[0] if all(node_hints) else metadata[1]:
            args = tuple(
                find_node_by_name(graph, f'{v.name}__{key2str(key)}')
                for v in node.args)
            kwargs = {
                k: find_node_by_name(graph, f'{v.name}__{key2str(key)}')
                for k, v in node.kwargs.items()
            }
            out = graph.create_node('call_function', target=node.target,
                                    args=args, kwargs=kwargs,
                                    name=f'{node.name}__{key2str(key)}')
            graph.inserting_after(out)

    else:
        raise NotImplementedError
