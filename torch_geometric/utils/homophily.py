from typing import Union

import torch
from torch import Tensor

from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import degree, scatter, unbatch, unbatch_edge_index


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
        \frac{| \{ (v,w) : (v,w) \in \mathcal{E} \wedge y_v = y_w \} | }
        {|\mathcal{E}|}

      This measure is called the *edge homophily ratio*.

    - In the `"Geom-GCN: Geometric Graph Convolutional Networks"
      <https://arxiv.org/abs/2002.05287>`_ paper, edge homophily is normalized
      across neighborhoods:

      .. math::
        \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \frac{ | \{ (w,v) : w
        \in \mathcal{N}(v) \wedge y_v = y_w \} |  } { |\mathcal{N}(v)| }

      This measure is called the *node homophily ratio*.

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

      Thus, this measure is called the *class insensitive edge homophily
      ratio*.

    - In the `"Characterizing Graph Datasets for Node Classification:
      Homophily-Heterophily Dichotomy and Beyond"
      <https://arxiv.org/abs/2209.06177>`_ paper, edge homophily is adjusted
      for the expected number of edges connecting nodes with the same class
      label (taking into account the number of classes, their sizes, and the
      distribution of node degrees among them):

      .. math::
        \frac{h_{edge} - \sum_{k=1}^C \bar{p}(k)^2}
        {1 - \sum_{k=1}^C \bar{p}(k)^2},

      where :math:`h_{edge}` denotes edge homophily, :math:`C` denotes the
      number of classes, and :math:`\bar{p}(\cdot)` is the empirical
      degree-weighted distribution of classes:
      :math:`\bar{p}(k) = \frac{\sum_{v\,:\,y_v = k} d(v)}{2|E|}`,
      where :math:`d(v)` is the degree of node :math:`v`.

      This measure is called *adjusted homophily ratio*.

      It has been shown that adjusted homophily satisifes more desirable
      properties than other homophily measures, which makes it appropriate for
      comparing the levels of homophily across datasets with different number
      of classes, different class sizes, andd different degree distributions
      among classes.

      Adjusted homophily can be negative. If adjusted homophily is zero, then
      the edge pattern in the graph is independent of node class labels. If it
      is positive, then the nodes in the graph tend to connect to nodes of the
      same class more often, and if it is negative, than the nodes in the graph
      tend to connect to nodes of different classes more often (compared to the
      null model where edges are independent of node class labels).

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        y (Tensor): The labels.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns
            each node to a specific example. (default: :obj:`None`)
        method (str, optional): The method used to calculate the homophily,
            either :obj:`"edge"` (first formula), :obj:`"node"` (second
            formula), :obj:`"edge_insensitive"` (third formula) or
            :obj:`"adjusted"` (fourth formula).
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

        >>> # Adjusted homophily ratio
        >>> homophily(edge_index, y, method='adjusted')
        -0.14285714285714285
    """
    assert method in {'edge', 'node', 'edge_insensitive', 'adjusted'}
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

    elif method == 'adjusted':
        import networkx as nx
        import numpy as np

        if isinstance(edge_index, SparseTensor):
            edge_index = torch.vstack([row, col])

        device = edge_index.device

        if batch is None:
            edge_index_list = [edge_index]
            y_list = [y]
        else:
            edge_index_list = unbatch_edge_index(edge_index, batch)
            y_list = unbatch(y, batch)

        h_adj_list = []
        for edge_index, y in zip(edge_index_list, y_list):
            graph = nx.Graph()
            graph.add_nodes_from(range(edge_index.max().item()))
            graph.add_edges_from(edge_index.cpu().T.numpy())

            labels = y.cpu().numpy()

            # Convert labels to consecutive integers
            unique_labels = np.unique(labels)
            labels_map = {label: i for i, label in enumerate(unique_labels)}
            labels = np.array([labels_map[label] for label in labels])

            num_classes = len(unique_labels)
            if num_classes == 1:
                h_adj_list.append(1)
                continue

            edges_with_same_label = 0
            for u, v in graph.edges:
                if labels[u] == labels[v]:
                    edges_with_same_label += 1

            h_edge = edges_with_same_label / len(graph.edges)

            degree_sums = np.zeros((num_classes, ))
            for u in graph.nodes:
                label = labels[u]
                degree_sums[label] += graph.degree(u)

            adjust = (degree_sums**2 / (len(graph.edges) * 2)**2).sum()

            h_adj = (h_edge - adjust) / (1 - adjust)

            h_adj_list.append(h_adj)

        if batch is None:
            return h_adj_list[0]
        else:
            return torch.tensor(h_adj_list, dtype=torch.float32, device=device)

    else:
        raise NotImplementedError
