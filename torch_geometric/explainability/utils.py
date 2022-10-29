from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.data import Data


def create_graph_data_from_args(
    x: Optional[Tensor] = None,
    edge_index: Optional[Tensor] = None,
    edge_attr: Optional[Tensor] = None,
    **kwargs: Tensor,
) -> Data:
    r"""Create a :class:`torch_geometric.data.Data.

    Serves as a interface between method that feed arguments to a model and
    the model itself. The model is expecting :class:`torch_geometric.data.Data`
    but some methods are expecting the arguments separately.

    Args:
        x (Tensor, optional): The node feature matrix.
        edge_index (Tensor, optional): Edge indices.
        edge_attr (Tensor, optional): Edge attributes.

    rtype :class:`torch_geometric.data.Data`
    """
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            data[key] = value
    return data


def get_args_from_graph(g: Data) -> dict[str, Tensor]:
    return {key: value for key, value in g.items() if torch.is_tensor(value)}


class Interface:
    """Interface class to convert between graph and model inputs.

    Useful when the model must respect __call__ signature
    with :class:`torch_geometric.data.Data` but the implemented model takes
    arguments separately.

    note::
        - the only strong constraint is subclassing are the inputs type of
        :py:meth:`convert` and :py:meth:`revert` methods.
    """
    def __init__(
        self, graph_to_inputs: Optional[Callable[[Data], dict[str,
                                                              Tensor]]] = None,
        inputs_to_graph: Optional[Callable[[Tuple[Tensor, ...]], Data]] = None
    ) -> None:
        if graph_to_inputs is None:
            graph_to_inputs = get_args_from_graph
        if inputs_to_graph is None:
            inputs_to_graph = create_graph_data_from_args
        self.graph_to_inputs_func = graph_to_inputs
        self.inputs_to_graph_func = inputs_to_graph

    def graph_to_inputs(self, graph: Data, **kwargs) -> dict[str, Tensor]:
        """Convert graph to inputs that can be passed to the model.

        The kwargs are just added to the dict output."""
        return dict({**self.graph_to_inputs_func(graph), **kwargs})

    def inputs_to_graph(self, x: Optional[Tensor] = None,
                        edge_index: Optional[Tensor] = None,
                        edge_attr: Optional[Tensor] = None, **kwargs) -> Data:
        """Convert inputs to graph."""
        return self.inputs_to_graph_func(x, edge_index, edge_attr, **kwargs)

    def set_graph_to_inpputs(
            self, graph_to_inputs: Callable[[Data], dict[str,
                                                         Tensor]]) -> None:
        self.graph_to_inputs_func = graph_to_inputs

    def set_inputs_to_graph(
            self, inputs_to_graph: Callable[[Tuple[Tensor, ...]],
                                            Data]) -> None:
        self.inputs_to_graph_func = inputs_to_graph
