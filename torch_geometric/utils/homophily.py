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


def homophily(edge_index: Adj, y: Tensor, method: str = 'edge'):
    r"""The homophily of a graph characterizes how likely nodes with the same
    label are near each other in a graph.
    There are many measures of homophily that fits this definition.
    In particular:

    - In the `"Beyond Homophily in Graph Neural Networks: Current Limitations
      and Effective Designs" <https://arxiv.org/abs/2006.11468>`_ paper, the
      homophily is the fraction of edges in a graph which connects nodes
      that have the same class label:

      .. math::
        \text{homophily} = \frac{| \{ (v,w) : (v,w) \in \mathcal{E} \wedge
        y_v = y_w \} | } {|\mathcal{E}|}

      That measure is called the *edge homophily ratio*.

    - In the `"Geom-GCN: Geometric Graph Convolutional Networks"
      <https://arxiv.org/abs/2002.05287>`_ paper, edge homophily is normalized
      across neighborhoods:

      .. math::
        \text{homophily} = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}}
        \frac{ | \{ (w,v) : w \in \mathcal{N}(v) \wedge y_v = y_w \} |  }
        { |\mathcal{N}(v)| }

      That measure is called the *node homophily ratio*.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        y (Tensor): The labels.
        method (str, optional): The method used to calculate the homophily,
            either :obj:`"edge"` (first formula) or :obj:`"node"`
            (second formula). (default: :obj:`"edge"`)
    """
    assert method in ['edge', 'node']
    y = y.squeeze(-1) if y.dim() > 1 else y

    if isinstance(edge_index, SparseTensor):
        col, row, _ = edge_index.coo()
    else:
        row, col = edge_index

    if method == 'edge':
        return int((y[row] == y[col]).sum()) / row.size(0)
    else:
        out = torch.zeros_like(row, dtype=float)
        out[y[row] == y[col]] = 1.
        out = scatter_mean(out, col, 0, dim_size=y.size(0))
        return float(out.mean())
