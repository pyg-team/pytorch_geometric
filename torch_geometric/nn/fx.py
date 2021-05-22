from typing import Optional, Dict, Any

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


def find_node_by_name(graph: Graph, name: str) -> Optional[Node]:
    for node in graph.nodes:
        if node.name == name:
            return node
    return None


def find_node_by_target(graph: Graph, target: str) -> Optional[Node]:
    for node in graph.nodes:
        if node.target == target:
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
