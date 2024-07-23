from typing import Union, overload

import torch
from torch import Tensor

from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import degree, scatter


@overload
def homophily(
    edge_index: Adj,
    y: Tensor,
    batch: None = ...,
    method: str = ...,
) -> float:
    pass


@overload
def homophily(
    edge_index: Adj,
    y: Tensor,
    batch: Tensor,
    method: str = ...,
) -> Tensor:
    pass


def homophily(
    edge_index: Adj,
    y: Tensor,
    batch: OptTensor = None,
    method: str = 'edge',
) -> Union[float, Tensor]:
    r"""The homophily of a graph characterizes how likely nodes with the same
    label are near each other in a graph.

    There are many measures of homophily that fits this definition.
    In particular:

    - In the `"Beyond Homophily in Graph Neural Networks: Current Limitations
      and Effective Designs" <https://arxiv.org/abs/2006.11468>`_ paper, the
      homophily is the fraction of edges in a graph which connects nodes
      that have the same class label:

      .. math::
        \frac{| \{ (v,w) : (v,w) \in \mathcal{E} \wedge y_v = y_w \} | }
        {|\mathcal{E}|}

      That measure is called the *edge homophily ratio*.

    - In the `"Geom-GCN: Geometric Graph Convolutional Networks"
      <https://arxiv.org/abs/2002.05287>`_ paper, edge homophily is normalized
      across neighborhoods:

      .. math::
        \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \frac{ | \{ (w,v) : w
        \in \mathcal{N}(v) \wedge y_v = y_w \} |  } { |\mathcal{N}(v)| }

      That measure is called the *node homophily ratio*.

    - In the `"Large-Scale Learning on Non-Homophilous Graphs: New Benchmarks
      and Strong Simple Methods" <https://arxiv.org/abs/2110.14446>`_ paper,
      edge homophily is modified to be insensitive to the number of classes
      and size of each class:

      .. math::
        \frac{1}{C-1} \sum_{k=1}^{C} \max \left(0, h_k - \frac{|\mathcal{C}_k|}
        {|\mathcal{V}|} \right),

      where :math:`C` denotes the number of classes, :math:`|\mathcal{C}_k|`
      denotes the number of nodes of class :math:`k`, and :math:`h_k` denotes
      the edge homophily ratio of nodes of class :math:`k`.

      Thus, that measure is called the *class insensitive edge homophily
      ratio*.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        y (Tensor): The labels.
        batch (LongTensor, optional): Batch vector\
            :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns
            each node to a specific example. (default: :obj:`None`)
        method (str, optional): The method used to calculate the homophily,
            either :obj:`"edge"` (first formula), :obj:`"node"` (second
            formula) or :obj:`"edge_insensitive"` (third formula).
            (default: :obj:`"edge"`)

    Examples:
        >>> edge_index = torch.tensor([[0, 1, 2, 3],
        ...                            [1, 2, 0, 4]])
        >>> y = torch.tensor([0, 0, 0, 0, 1])
        >>> # Edge homophily ratio
        >>> homophily(edge_index, y, method='edge')
        0.75

        >>> # Node homophily ratio
        >>> homophily(edge_index, y, method='node')
        0.6000000238418579

        >>> # Class insensitive edge homophily ratio
        >>> homophily(edge_index, y, method='edge_insensitive')
        0.19999998807907104
    """
    assert method in {'edge', 'node', 'edge_insensitive'}
    y = y.squeeze(-1) if y.dim() > 1 else y

    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
    else:
        row, col = edge_index

    if method == 'edge':
        out = torch.zeros(row.size(0), device=row.device)
        out[y[row] == y[col]] = 1.
        if batch is None:
            return float(out.mean())
        else:
            dim_size = int(batch.max()) + 1
            return scatter(out, batch[col], 0, dim_size, reduce='mean')

    elif method == 'node':
        out = torch.zeros(row.size(0), device=row.device)
        out[y[row] == y[col]] = 1.
        out = scatter(out, col, 0, dim_size=y.size(0), reduce='mean')
        if batch is None:
            return float(out.mean())
        else:
            return scatter(out, batch, dim=0, reduce='mean')

    elif method == 'edge_insensitive':
        assert y.dim() == 1
        num_classes = int(y.max()) + 1
        assert num_classes >= 2
        batch = torch.zeros_like(y) if batch is None else batch
        num_nodes = degree(batch, dtype=torch.int64)
        num_graphs = num_nodes.numel()
        batch = num_classes * batch + y

        h = homophily(edge_index, y, batch, method='edge')
        h = h.view(num_graphs, num_classes)

        counts = batch.bincount(minlength=num_classes * num_graphs)
        counts = counts.view(num_graphs, num_classes)
        proportions = counts / num_nodes.view(-1, 1)

        out = (h - proportions).clamp_(min=0).sum(dim=-1)
        out /= num_classes - 1
        return out if out.numel() > 1 else float(out)

    else:
        raise NotImplementedError
