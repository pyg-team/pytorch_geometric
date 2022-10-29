from typing import Optional

import torch
from torch import Tensor

from torch_geometric.data import Data


def create_graph_data_from_args(
    x: Optional[Tensor] = None,
    edge_index: Optional[Tensor] = None,
    edge_attr: Optional[Tensor] = None,
    **kwargs,
):
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
