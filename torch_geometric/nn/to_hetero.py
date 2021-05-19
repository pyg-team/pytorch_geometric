from typing import Tuple, List, Dict, Union, Optional

import copy
import warnings

import torch
from torch.nn import Module
from torch import fx
from torch_geometric.nn import MessagePassing

Metagraph = Tuple[List[str], List[Tuple[str, str, str]]]

# TODO:
# * LazyInitialization
# * What are leaf modules?
# * Flexible args/kwargs replacement
# * What happens in ModuleLists and ModuleDicts?


def metatype2str(metatype: Union[str, Tuple[str, str, str]]) -> str:
    return '__'.join(metatype) if isinstance(metatype, tuple) else metatype


def duplicate_submodules(module: Module, metagraph: Metagraph) -> Module:
    # TODO: Handle `torch.nn.ModuleList` and `torch.nn.ModuleDict`
    metagraph = metagraph[:1] + ([metatype2str(t) for t in metagraph[1]], )

    for name, submodule in dict(module.named_children()).items():
        module_dict = torch.nn.ModuleDict()

        is_message_passing = isinstance(submodule, MessagePassing)
        for metatype in metagraph[1] if is_message_passing else metagraph[0]:
            module_dict[metatype] = copy.deepcopy(submodule)
            if hasattr(submodule, 'reset_parameters'):
                module_dict[metatype].reset_parameters()
            else:
                warnings.warn((f"'{name}' will be duplicated, but its "
                               "parameters cannot be reset"))

        module._modules[name] = module_dict

    return module


def transform(module: Module, metagraph: Metagraph, input_map: Dict[str, str],
              debug: bool = True) -> Module:
    metagraph = metagraph[:1] + ([metatype2str(t) for t in metagraph[1]], )

    gm = fx.symbolic_trace(module)
    gm = duplicate_submodules(gm, metagraph)

    if debug:
        gm.graph.print_tabular()
        print()
        print(gm.graph.python_code('self'))

    for node in list(gm.graph.nodes):
        if node.op == 'placeholder' and input_map[node.name] == 'node':
            unwrap_placeholder(gm.graph, node, metagraph[0])
        elif node.op == 'placeholder' and input_map[node.name] == 'edge':
            unwrap_placeholder(gm.graph, node, metagraph[1])
        elif node.op == 'get_attr':
            raise NotImplementedError
        elif node.op == 'call_module':
            unwrap_call_node_module(gm.graph, node, metagraph[0])
        elif node.op == 'call_method':
            unwrap_call_method(gm.graph, node, metagraph[0])
        elif node.op == 'call_function':
            raise NotImplementedError
        elif node.op == 'output':
            unwrap_output(gm.graph, node, metagraph[0])
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


def find_node(graph: fx.Graph, name: str) -> Optional[fx.Node]:
    out: Optional[fx.Node] = None
    for node in graph.nodes:
        if node.name == name:
            out = node
    return out


def unwrap_placeholder(graph: fx.Graph, node: fx.Node,
                       metatypes: List[str]) -> List[fx.Node]:
    outs: List[fx.Node] = []
    node.type = Dict[str, node.type]  # Replace input type.
    graph.inserting_after(node)
    for metatype in metatypes:
        out = graph.create_node('call_method', 'get', args=(node, metatype),
                                name=f'{node.name}__{metatype}')
        graph.inserting_after(out)
        outs.append(out)
    return outs


def unwrap_call_node_module(graph: fx.Graph, node: fx.Node,
                            metatypes: List[str]) -> List[fx.Node]:
    outs: List[fx.Node] = []
    graph.inserting_after(node)
    for metatype in metatypes:
        input_node = find_node(graph, name=f'{node.args[0].name}__{metatype}')
        out = graph.create_node('call_module', f'{node.target}.{metatype}',
                                args=(input_node, ),
                                name=f'{node.name}__{metatype}')
        graph.inserting_after(out)
        outs.append(out)
    return outs


def unwrap_call_method(graph: fx.Graph, node: fx.Node,
                       metatypes: List[str]) -> List[fx.Node]:
    outs: List[fx.Node] = []
    graph.inserting_after(node)
    for metatype in metatypes:
        input_node = find_node(graph, name=f'{node.args[0].name}__{metatype}')
        out = graph.create_node('call_method', node.target,
                                args=(input_node, ),
                                name=f'{node.name}__{metatype}')
        graph.inserting_after(out)
        outs.append(out)
    return outs


def unwrap_output(graph: fx.Graph, node: fx.Node,
                  metatypes: List[str]) -> fx.Node:
    node.args = ({
        metatype: find_node(graph, f'{node.args[0].name}__{metatype}')
        for metatype in metatypes
    }, )
    node.type = Dict[str, node.type]  # Replace output type.
    return node


def erase_unused_nodes(graph: fx.Graph) -> fx.Graph:
    for node in list(graph.nodes)[::-1]:  # Iterate in reverse-mode.
        try:
            if node.op not in ['output']:
                graph.erase_node(node)
        except RuntimeError:
            pass
    return graph
