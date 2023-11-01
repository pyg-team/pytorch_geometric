from typing import Union

import torch
from torch import Tensor

from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import unbatch, unbatch_edge_index


def label_informativeness(edge_index: Adj, y: Tensor, batch: OptTensor = None,
                          method: str = 'edge',
                          eps: float = 1e-8) -> Union[float, Tensor]:
    r"""Label informativeness (:math:`\mathrm{LI}`) is a characteristic of
    labeled graphs proposed in the `"Characterizing Graph Datasets for Node
    Classification: Homophily-Heterophily Dichotomy and Beyond"
    <https://arxiv.org/abs/2209.06177>`_ paper. It shows how much information
    about a node's label we get from knowing its neighbor's label. Formally,
    assume that we sample an edge :math:`(\xi,\eta) \in E`. The class labels of
    nodes :math:`\xi` and :math:`\eta` are then random variables :math:`y_\xi`
    and :math:`y_\eta`. We want to measure the amount of knowledge the label
    :math:`y_\eta` gives for predicting :math:`y_\xi`. The entropy
    :math:`H(y_\xi)` measures the `hardness' of predicting the label of
    :math:`\xi` without knowing :math:`y_\eta`. Given :math:`y_\eta`, this
    value is reduced to the conditional entropy :math:`H(y_\xi|y_\eta)`. In
    other words, :math:`y_\eta` reveals
    :math:`I(y_\xi,y_\eta) = H(y_\xi) - H(y_\xi|y_\eta)` information about the
    label. To make the obtained quantity comparable across different datasets,
    label informativeness is defined as the normalized mutual information of
    :math:`y_{\xi}` and :math:`y_{\eta}`:

    .. math::
      \mathrm{LI} = \frac{I(y_\xi,y_\eta)}{H(y_\xi)}

    Depending on the distribution used for sampling an edge
    :math:`(\xi, \eta)`, several variants of label informativeness can be
    obtained. Two of them are particularly intuitive: in edge label
    informativeness (:math:`\mathrm{LI}_{edge}`), edges are sampled uniformly
    at random, and in node label informativeness (:math:`\mathrm{LI}_{node}`),
    first a node is sampled uniformly at random and then an edge incident to it
    is sampled uniformly at random. These two versions of label informativeness
    differ in how they weight high/low-degree nodes. In edge label
    informativeness, averaging is over the edges, thus high-degree nodes are
    given more weight. In node label informativeness, averaging is over the
    nodes, so all nodes are weighted equally.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        y (Tensor): The labels.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns
            each node to a specific example. (default: :obj:`None`)
        method (str, optional): The edge sampling method used to calculate
            label informativeness, either :obj:`"edge"` or :obj:`"node"`.
            (default: :obj:`"edge"`)
        eps (float, optional): A small constant for numerical stability.
            (default: :obj:`1e-8`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 2, 2, 3, 4],
        ...                            [1, 2, 0, 3, 4, 5]])
        >>> y = torch.tensor([0, 0, 0, 0, 1, 1])
        >>> # Edge label informativeness
        >>> label_informativeness(edge_index, y, method='edge')
        0.2517760099956565

        >>> # Node label informativeness
        >>> label_informativeness(edge_index, y, method='node')
        0.3381873621927896
    """
    import numpy as np
    import networkx as nx

    assert method in {'edge', 'node'}
    y = y.squeeze(-1) if y.dim() > 1 else y

    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        edge_index = torch.vstack([row, col])

    if batch is None:
        edge_index_list = [edge_index]
        y_list = [y]
    else:
        edge_index_list = unbatch_edge_index(edge_index, batch)
        y_list = unbatch(y, batch)

    li_list = []
    for edge_index, y in zip(edge_index_list, y_list):
        graph = nx.Graph()
        graph.add_nodes_from(range(edge_index.max().item()))
        graph.add_edges_from(edge_index.T.numpy())

        labels = y.numpy()

        # Convert labels to consecutive integers
        unique_labels = np.unique(labels)
        labels_map = {label: i for i, label in enumerate(unique_labels)}
        labels = np.array([labels_map[label] for label in labels])

        num_classes = len(unique_labels)

        if method == 'edge':
            class_degree_weighted_probs = np.array(
                [0 for _ in range(num_classes)], dtype=float
            )
            for u in graph.nodes:
                label = labels[u]
                class_degree_weighted_probs[label] += graph.degree(u)

            class_degree_weighted_probs /= class_degree_weighted_probs.sum()

            edge_probs = np.zeros((num_classes, num_classes))
            for u, v in graph.edges:
                label_u = labels[u]
                label_v = labels[v]
                edge_probs[label_u, label_v] += 1
                edge_probs[label_v, label_u] += 1

            edge_probs /= edge_probs.sum()

            edge_probs += eps

            numerator = (edge_probs * np.log(edge_probs)).sum()
            denominator = (class_degree_weighted_probs *
                           np.log(class_degree_weighted_probs)).sum()
            li_edge = 2 - numerator / denominator

            li_list.append(li_edge)

        elif method == 'node':
            class_probs = np.array(
                [0 for _ in range(num_classes)], dtype=float
            )
            class_degree_weighted_probs = np.array(
                [0 for _ in range(num_classes)], dtype=float
            )
            num_zero_degree_nodes = 0
            for u in graph.nodes:
                if graph.degree(u) == 0:
                    num_zero_degree_nodes += 1
                    continue

                label = labels[u]
                class_probs[label] += 1
                class_degree_weighted_probs[label] += graph.degree(u)

            class_probs /= class_probs.sum()
            class_degree_weighted_probs /= class_degree_weighted_probs.sum()
            num_nonzero_degree_nodes = len(graph.nodes) - num_zero_degree_nodes

            edge_probs = np.zeros((num_classes, num_classes))
            for u, v in graph.edges:
                label_u = labels[u]
                label_v = labels[v]
                edge_probs[label_u, label_v] += \
                    1 / (num_nonzero_degree_nodes * graph.degree(u))
                edge_probs[label_v, label_u] += \
                    1 / (num_nonzero_degree_nodes * graph.degree(v))

            edge_probs += eps

            log = np.log(edge_probs /
                         (class_probs.reshape(-1, 1) *
                          class_degree_weighted_probs.reshape(1, -1)))
            numerator = (edge_probs * log).sum()
            denominator = (class_probs * np.log(class_probs)).sum()
            li_node = - numerator / denominator

            li_list.append(li_node)

        else:
            raise NotImplementedError

    if batch is None:
        return li_list[0]
    else:
        return torch.tensor(li_list, dtype=torch.float32)
