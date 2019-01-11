import torch
import scipy.sparse
import networkx as nx

from .num_nodes import maybe_num_nodes


def to_scipy_sparse_matrix(edge_index, edge_attr=None, num_nodes=None):
    r"""Converts a graph given by edge indices and edge attributes to a scipy
    sparse matrix.

    Args:
        edge_index (LongTensor): The edge_indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    row, col = edge_index.cpu()

    if edge_attr is None:
        edge_attr = torch.ones(row.size(0))
    else:
        edge_attr = edge_attr.view(-1).cpu()
        assert edge_attr.size(0) == row.size(0)

    N = maybe_num_nodes(edge_index, num_nodes)
    out = scipy.sparse.coo_matrix((edge_attr, (row, col)), (N, N))
    return out


def to_networkx(edge_index, x=None, edge_attr=None, pos=None, num_nodes=None):
    r"""Converts a graph given by edge indices, node features, node positions
    and edge attributes to a networkx graph.

    Args:
        edge_index (LongTensor): The edge_indices.
        x (Tensor, optional): The node feature matrix. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        pos (Tensor, optional): The node position matrix.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    num_nodes = num_nodes if x is None else x.size(0)
    num_nodes = num_nodes if pos is None else pos.size(0)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    G = nx.DiGraph()

    for i in range(num_nodes):
        G.add_node(i)
        if x is not None:
            G.nodes[i]['x'] = x[i].cpu().numpy()
        if pos is not None:
            G.nodes[i]['pos'] = pos[i].cpu().numpy()

    for i in range(edge_index.size(1)):
        source, target = edge_index[0][i].item(), edge_index[1][i].item()
        G.add_edge(source, target)
        if edge_attr is not None:
            if edge_attr.numel() == edge_attr.size(0):
                G[source][target]['weight'] = edge_attr[i].item()
            else:
                G[source][target]['weight'] = edge_attr[i].cpu().numpy()

    return G
