import torch
import scipy.sparse
import networkx as nx

import torch_geometric.data

try:
    import trimesh
except ImportError:
    trimesh = None

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
    r"""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.
    """
    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight


def to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.DiGraph` if :attr:`to_undirected` is set to :obj:`True`, or
    an undirected :obj:`networkx.Graph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    values = {}
    for key, item in data:
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G


def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data


def to_trimesh(data):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`trimesh.Trimesh`.

    Args:
        data (torch_geometric.data.Data): The data object.
    """

    if trimesh is None:
        raise ImportError('Package `trimesh` could not be found.')

    return trimesh.Trimesh(vertices=data.pos.detach().cpu().numpy(),
                           faces=data.face.detach().t().cpu().numpy(),
                           process=False)


def from_trimesh(mesh):
    r"""Converts a :obj:`trimesh.Trimesh` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mesh (trimesh.Trimesh): A :obj:`trimesh` mesh.
    """

    if trimesh is None:
        raise ImportError('Package `trimesh` could not be found.')

    pos = torch.from_numpy(mesh.vertices).to(torch.float)
    face = torch.from_numpy(mesh.faces).t().contiguous()

    return torch_geometric.data.Data(pos=pos, face=face)
