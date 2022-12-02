from typing import Any

from torch.nn import Module

from torch_geometric.nn.fx import Transformer

try:
    from torch.fx import Graph, GraphModule, Node
except (ImportError, ModuleNotFoundError, AttributeError):
    GraphModule, Graph, Node = 'GraphModule', 'Graph', 'Node'


def to_fixed_size(module: Module, batch_size: int,
                  debug: bool = False) -> GraphModule:
    r"""Converts a model and injects a pre-computed and fixed batch size to all
    global pooling operators.

    Args:
        module (torch.nn.Module): The model to transform.
        batch_size (int): The fixed batch size used in global pooling modules.
        debug (bool, optional): If set to :obj:`True`, will perform
            transformation in debug mode. (default: :obj:`False`)
    """
    transformer = ToFixedSizeTransformer(module, batch_size, debug)
    return transformer.transform()


class ToFixedSizeTransformer(Transformer):
    def __init__(self, module: Module, batch_size: int, debug: bool = False):
        super().__init__(module, debug=debug)
        self.batch_size = batch_size

    def call_global_pooling_module(self, node: Node, target: Any, name: str):
        kwargs = node.kwargs.copy()
        kwargs['dim_size'] = self.batch_size
        node.kwargs = kwargs
