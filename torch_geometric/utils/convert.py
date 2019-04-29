import torch
import scipy.sparse
import networkx as nx
import torch_geometric

from .num_nodes import maybe_num_nodes


def to_scipy_sparse_matrix(edge_index, edge_attr=None, num_nodes=None):
    r"""Converts a graph given by edge indices and edge attributes to a scipy
    sparse matrix.

    Args:
        edge_index (LongTensor): The edge indices.
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


def from_scipy_sparse_matrix(A):
    r""""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.
    """
    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight


def to_networkx(data, node_attrs=None, edge_attrs=None):
    r"""Converts a data object graph to a networkx graph.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
    """

    G = nx.DiGraph()
    G.add_nodes_from(range(data.num_nodes))

    values = {key: data[key].squeeze().tolist() for key in data.keys}

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = data[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G


def from_networkx(G):
    r""""Converts a networkx graph to a data object graph.

    Args:
        G (networkx.Graph): A networkx graph.
    """

    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    keys = []
    keys += list(G.nodes(data=True)[0].keys())
    keys += list(list(G.edges(data=True))[0][2].keys())
    data = {key: [] for key in keys}

    for _, feat_dict in G.nodes(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)

    for _, _, feat_dict in G.edges(data=True):
        for key, value in feat_dict.items():
            data[key].append(value)

    for key, item in data.items():
        data[key] = torch.tensor(item)
    data['edge_index'] = edge_index

    return torch_geometric.data.Data.from_dict(data)
