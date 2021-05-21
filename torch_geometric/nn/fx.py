from typing import Optional, Dict, Any

import re

from torch.nn import Module
from torch_geometric.nn import MessagePassing

try:
    from torch.fx import Tracer, GraphModule, Graph, Node
except ImportError:
    Tracer = GraphModule = Graph = Node = None


def symbolic_trace(
        module: Module,
        concrete_args: Optional[Dict[str, Any]] = None) -> GraphModule:
    class MyTracer(Tracer):
        def is_leaf_module(self, module: Module, *args, **kwargs) -> bool:
            return (isinstance(module, MessagePassing)
                    or super().is_leaf_module(module, *args, **kwargs))

    return GraphModule(module, MyTracer().trace(module, concrete_args))


def set_node_hints_(graph: Graph, input_map: Optional[Dict[str, str]] = None):
    input_map = input_map or {}
    for node in graph.nodes:
        if node.op == 'placeholder':
            node.hint = input_map.get(node.name, None)
            if node.hint is None and bool(re.search('(edge|adj)', node.name)):
                node.hint = 'edge'
            node.hint = node.hint or 'node'
            assert node.hint in ['node', 'edge']
        elif node.op == 'output':
            pass  # Nothing to do
        elif node.op == 'get_attr':
            raise NotImplementedError
        else:
            it = list(node.args) + list(node.kwargs.values())
            node_hints = [v.hint == 'node' for v in it if isinstance(v, Node)]
            node.hint = 'node' if any(node_hints) else 'edge'


def find_node_by_name(graph: Graph, name: str) -> Optional[Node]:
    for node in graph.nodes:
        if node.name == name:
            return node
    return None


def erase_unused_nodes(graph: Graph) -> Graph:
    for node in list(graph.nodes)[::-1]:
        try:
            if node.op not in ['placeholder', 'output']:
                graph.erase_node(node)
        except RuntimeError:
            pass
    return graph
