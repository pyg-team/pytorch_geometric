import torch
import numpy as np
from torch_geometric.utils import to_undirected, remove_self_loops


def erdos_renyi_graph(num_nodes, edge_prob, directed=False):
    r"""Returns the :obj:`edge_index` of a random Erdos-Renyi graph.

    Args:
        num_nodes (int): The number of nodes.
        edge_prob (float): Probability of an edge.
        directed (bool, optional): If set to :obj:`True`, will return a
            directed graph. (default: :obj:`False`)
    """

    if directed:
        idx = torch.arange((num_nodes - 1) * num_nodes)
        idx = idx.view(num_nodes - 1, num_nodes)
        idx = idx + torch.arange(1, num_nodes).view(-1, 1)
        idx = idx.view(-1)
    else:
        idx = torch.combinations(torch.arange(num_nodes), r=2)

    # Filter edges.
    mask = torch.rand(idx.size(0)) < edge_prob
    idx = idx[mask]

    if directed:
        row = idx.div(num_nodes, rounding_mode='floor')
        col = idx % num_nodes
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = to_undirected(idx.t(), num_nodes=num_nodes)

    return edge_index


def stochastic_blockmodel_graph(block_sizes, edge_probs, directed=False):
    r"""Returns the :obj:`edge_index` of a stochastic blockmodel graph.

    Args:
        block_sizes ([int] or LongTensor): The sizes of blocks.
        edge_probs ([[float]] or FloatTensor): The density of edges going
        from each block to each other block. Must be symmetric if the graph is
            undirected.
        directed (bool, optional): If set to :obj:`True`, will return a
            directed graph. (default: :obj:`False`)
    """

    size, prob = block_sizes, edge_probs

    if not torch.is_tensor(size):
        size = torch.tensor(size, dtype=torch.long)
    if not torch.is_tensor(prob):
        prob = torch.tensor(prob, dtype=torch.float)

    assert size.dim() == 1
    assert prob.dim() == 2 and prob.size(0) == prob.size(1)
    assert size.size(0) == prob.size(0)
    if not directed:
        assert torch.allclose(prob, prob.t())

    node_idx = torch.cat([size.new_full((b, ), i) for i, b in enumerate(size)])
    num_nodes = node_idx.size(0)

    if directed:
        idx = torch.arange((num_nodes - 1) * num_nodes)
        idx = idx.view(num_nodes - 1, num_nodes)
        idx = idx + torch.arange(1, num_nodes).view(-1, 1)
        idx = idx.view(-1)
        row = idx.div(num_nodes, rounding_mode='floor')
        col = idx % num_nodes
    else:
        row, col = torch.combinations(torch.arange(num_nodes), r=2).t()

    mask = torch.bernoulli(prob[node_idx[row], node_idx[col]]).to(torch.bool)
    edge_index = torch.stack([row[mask], col[mask]], dim=0)

    if not directed:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    return edge_index


def barabasi_albert_graph(num_nodes, num_edges):
    r"""Returns the :obj:`edge_index` of a Barabasi-Albert preferential
    attachment model, where a graph of :obj:`num_nodes` nodes grows by
    attaching new nodes with :obj:`num_edges` edges that are preferentially
    attached to existing nodes with high degree.

    Args:
        num_nodes (int): The number of nodes.
        num_edges (int): The number of edges from a new node to existing nodes.
    """

    assert num_edges > 0 and num_edges < num_nodes

    row, col = torch.arange(num_edges), torch.randperm(num_edges)

    for i in range(num_edges, num_nodes):
        row = torch.cat([row, torch.full((num_edges, ), i, dtype=torch.long)])
        choice = np.random.choice(torch.cat([row, col]).numpy(), num_edges)
        col = torch.cat([col, torch.from_numpy(choice)])

    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    return edge_index
