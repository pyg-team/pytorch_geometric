from typing import Union

import torch
from torch import Tensor
from torch_scatter import scatter_mean
from torch_sparse import SparseTensor

from torch_geometric.typing import Adj, OptTensor


def homophily(edge_index: Adj, y: Tensor, batch: OptTensor = None,
              method: str = 'edge') -> Union[float, Tensor]:
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

    - In the `"Large-scale learning on non-homophilous graphs: \
      New benchmarks and strong simple methods"
      <https://arxiv.org/abs/2110.14446v1>`_ paper, the class
      insensitive homophily metric better captures the presence or
      absence of homophily:

      .. math::
        \text{homophily} = \frac{1}{C-1}\sum_{k=0}^{C-1}\begin{bmatrix}
        {h_k - \frac{\lvert C_k \rvert}{n}}\end{bmatrix}_+,
      .. math::
        h_k = \frac{\sum_{u \in C_k}d_u^{(k_u)}}{\sum_{u \in C_k}d_u}

      where :math:`C_k` is the number of samples of class :math:`k`,
      :math:`d_u` is the number of neighbors of node :math:`u`,
      and :math:`d^{(k_u)}_u` is the number of neighbors of u that have
      the same class label. That measure is called the *class
      insensitive edge homophily ratio*.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        y (Tensor): The labels.
        batch (LongTensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns
        each node to a specific example. (default: :obj:`None`)
        method (str, optional): The method used to calculate the homophily,
        either :obj:`"edge"` (first formula), :obj:`"node"`
        (second formula) or :obj:`"edge-insensitive"`
        (third formula). (default: :obj:`"edge"`)
    """
    assert method in ['edge', 'node', 'edge-insensitive']
    y = y.squeeze(-1) if y.dim() > 1 else y

    if isinstance(edge_index, SparseTensor):
        col, row, _ = edge_index.coo()
    else:
        row, col = edge_index

    if method == 'edge':
        out = torch.zeros(row.size(0), device=row.device)
        out[y[row] == y[col]] = 1.
        if batch is None:
            return float(out.mean())
        else:
            return scatter_mean(out, batch[col], dim=0)

    elif method == 'node':
        out = torch.zeros(row.size(0), device=row.device)
        out[y[row] == y[col]] = 1.
        out = scatter_mean(out, col, 0, dim_size=y.size(0))
        if batch is None:
            return float(out.mean())
        else:
            return scatter_mean(out, batch, dim=0)

    else:
        c = y.squeeze().max() + 1
        counts = y.unique(return_counts=True)[1]
        proportions = counts.float() / y.shape[0]
        h = homophily(edge_index, y, batch=y, method='edge')

        out = 0
        for k in range(c):
            class_add = torch.clamp(h[k] - proportions[k], min=0)
            if not torch.isnan(class_add):
                out += class_add

        out /= c - 1
        return out
