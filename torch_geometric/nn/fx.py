from typing import Optional, Dict, Any

from torch.nn import Module, ModuleList, ModuleDict, Sequential
from torch_geometric.nn.conv import MessagePassing

try:
    from torch.fx import GraphModule, Graph, Node, Tracer
except ImportError:
    GraphModule = Graph = Node = Target = Tracer = None


class Transformer(object):
    def __init__(
        self,
        module: Module,
        debug: bool = False,
    ):
        self.module = module
        self.gm = symbolic_trace(module)
        self.debug = debug

    @property
    def graph(self) -> Graph:
        return self.gm.graph

    def transform(self) -> GraphModule:
        if self.debug:
            self.graph.print_tabular()
            print()
            print(self.graph.python_code('self'))

        for node in list(self.graph.nodes):
            if (node.op == 'call_module' and isinstance(
                    get_submodule(self.module, node.target), MessagePassing)):
                self.call_message_passing_module(node, node.target, node.name)
            else:
                getattr(self, node.op)(node, node.target, node.name)

        # Remove all unused nodes:
        for node in reversed(list(self.graph.nodes)):
            try:
                if node.op not in ['placeholder', 'output']:
                    self.graph.erase_node(node)
            except RuntimeError:
                pass

        if self.debug:
            self.gm.graph.print_tabular()
            print()
            print(self.graph.python_code('self'))

        for target, submodule in dict(self.module._modules).items():
            self.gm._modules[target] = self._init_submodule(submodule, target)

        self.gm.graph.lint()
        self.gm.recompile()

        return self.gm

    def placeholder(self, node: Node, target: Any, name: str):
        pass

    def get_attr(self, node: Node, target: Any, name: str):
        pass

    def call_message_passing_module(self, node: Node, target: Any, name: str):
        pass

    def call_module(self, node: Node, target: Any, name: str):
        pass

    def call_method(self, node: Node, target: Any, name: str):
        pass

    def call_function(self, node: Node, target: Any, name: str):
        pass

    def output(self, node: Node, target: Any, name: str):
        pass

    def init_submodule(self, module: Module, target: str) -> Module:
        return module

    def _init_submodule(self, module: Module, target: str) -> Module:
        if isinstance(module, ModuleList) or isinstance(module, Sequential):
            return ModuleList([
                self._init_submodule(submodule, f'{target}.{i}')
                for i, submodule in enumerate(module)
            ])
        elif isinstance(module, ModuleDict):
            return ModuleDict({
                key: self._init_submodule(submodule, f'{target}.{key}')
                for key, submodule in module.items()
            })
        else:
            return self.init_submodule(module, target)

    def find_by_name(self, name: str) -> Optional[Node]:
        for node in self.graph.nodes:
            if node.name == name:
                return node
        return None

    def find_by_target(self, target: Any) -> Optional[Node]:
        for node in self.graph.nodes:
            if node.target == target:
                return node
        return None


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
