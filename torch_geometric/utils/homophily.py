from torch_geometric.typing import Adj

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean


@torch.jit._overload
def homophily(adj_t, y, method):
    # type: (SparseTensor, Tensor) -> float
    pass


@torch.jit._overload
def homophily(edge_index, y, method):
    # type: (Tensor, Tensor) -> float
    pass


def homophily(edge_index: Adj, y: Tensor, method: str = "edge"):
    r"""The homophily of a graph characterizes how likely nodes with the same label
    are near each other in the graph. There are many measures of homophily
    that fits this definition. In particular:
    
    - In the `"Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs"
      <https://arxiv.org/abs/2006.11468>`_ paper, the
      homophily is the fraction of edges in a graph which connects nodes
      that have the same class label:
    
      .. math::
        \text{homophily} = \frac{\vert (u,v) : (u,v) \in V \ \wedge \ y_u=y_v \vert }
                                {\vert V \vert} 
    
      That measure is called the *edge homophily ratio* in the paper.
    
    - In the `"Geom-GCN: Geometric Graph Convolutional Networks"
      <https://arxiv.org/abs/2002.05287>`_ paper, we find

      .. math::
        \text{homophily} = \frac{1}{\vert V \vert} \sum_{v \in V}{
        \frac{ \text{Number of }v\text{'s neighbors who have the same label as }v}
             { \text{Number of }v\text{'s neighbors} }
        }
    
    In both formulas, :math:`V` designates the set of nodes.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        y (Tensor): The labels.
        method (str, optional): The method used to calculate the homophily,
            either :obj:`"edge"` (first formula above) or :obj:`"node"`
            (second formula above). (default: :obj:`"edge"`)
    """
    
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index
    
    if method == "edge":
        return int((y[row] == y[col]).sum()) / row.size(0)
    else:
        sim = torch.zeros_like(row, dtype=float)
        sim[ (y[row] == y[col]) ] = 1
        mean_value = scatter_mean(sim, col, 0, dim_size=y.size(0))
        return mean_value.sum().item() / y.size(0)
