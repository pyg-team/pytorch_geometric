from torch_geometric.typing import Adj

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean


@torch.jit._overload
def homophily_ratio(adj_t, y):
    # type: (SparseTensor, Tensor) -> float
    pass


@torch.jit._overload
def homophily_ratio(edge_index, y):
    # type: (Tensor, Tensor) -> float
    pass


def homophily_ratio(edge_index, y):
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index

    return int((y[row] == y[col]).sum()) / row.size(0)

def homophily(edge_index: Adj, y: Tensor):
    r"""The homophily of a graph characterizes how likely nodes with the same label
    are near each other in the graph. We adopt the measure proposed in
    the `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://arxiv.org/abs/2002.05287>`_ paper

    .. math::
        \text{homophily} = \frac{1}{\vert V \vert} \sum_{v \in V}{
        \frac{ \text{Number of }v\text{'s neighbors who have the same label as }v}
             { \text{Number of }v\text{'s neighbors} }
        }
        
    where :math:`V` designates the set of nodes.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        y (Tensor): The labels.
    """
    
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index
    
    sim = torch.zeros_like(row, dtype=float)
    sim[ (y[row] == y[col]) ] = 1
    mean_value = scatter_mean(sim, col, 0, dim_size=y.size(0))
    
    return mean_value.sum().item() / y.size(0)
