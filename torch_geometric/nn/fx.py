from typing import Optional, Dict, Any, List, Tuple

from torch.nn import Module
from torch_geometric.nn import MessagePassing

try:
    from torch.fx import GraphModule, Graph, Node, Tracer
except ImportError:
    GraphModule = Graph = Node = Target = Tracer = None


class Transformer(object):
    def __init__(
        self,
        graph_module: GraphModule,
        input_map: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ):
        self.graph_module = graph_module
        self.input_map = input_map
        self.debug = debug

    @property
    def graph(self) -> Graph:
        return self.graph_module.graph

    @property
    def nodes(self) -> List[Node]:
        return list(self.graph.nodes)

    def transform(self) -> GraphModule:
        if self.debug:
            self.graph.print_tabular()
            print()
            print(self.graph.python_code('self'))

        for node in self.nodes:
            if node.op == 'placeholder':
                pass
            else:
                raise NotImplementedError

        for node in reversed(self.nodes):
            try:
                if node.op not in ['placeholder', 'output']:
                    self.graph.erase_node(node)
            except RuntimeError:
                pass

        if self.debug:
            self.graph_module.graph.print_tabular()
            print()
            print(self.graph.python_code('self'))

        self.graph_module.graph.lint()
        self.graph_module.recompile()

        return self.graph_module

    def call_placeholder(self, node: Node, target: Any, args: Tuple,
                         kwargs: Dict, name: str):
        raise NotImplementedError


def symbolic_trace(
        module: Module,
        concrete_args: Optional[Dict[str, Any]] = None) -> GraphModule:
    class MyTracer(Tracer):
        def is_leaf_module(self, module: Module, *args, **kwargs) -> bool:
            return (isinstance(module, MessagePassing)
                    or super().is_leaf_module(module, *args, **kwargs))

    return GraphModule(module, MyTracer().trace(module, concrete_args))


def get_submodule(module: Module, target: str) -> Module:
    out = module
    for attr in target.split('.'):
        out = getattr(out, attr)
    return out


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
