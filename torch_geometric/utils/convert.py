from collections import defaultdict
from typing import Any, Iterable, List, Optional, Tuple, Union

import scipy.sparse
import torch
from torch import Tensor
from torch.utils.dlpack import from_dlpack, to_dlpack

import torch_geometric

from .num_nodes import maybe_num_nodes


def to_scipy_sparse_matrix(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> scipy.sparse.coo_matrix:
    r"""Converts a graph given by edge indices and edge attributes to a scipy
    sparse matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    Examples:

        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> to_scipy_sparse_matrix(edge_index)
        <4x4 sparse matrix of type '<class 'numpy.float32'>'
            with 6 stored elements in COOrdinate format>
    """
    row, col = edge_index.cpu()

    if edge_attr is None:
        edge_attr = torch.ones(row.size(0))
    else:
        edge_attr = edge_attr.view(-1).cpu()
        assert edge_attr.size(0) == row.size(0)

    N = maybe_num_nodes(edge_index, num_nodes)
    out = scipy.sparse.coo_matrix(
        (edge_attr.numpy(), (row.numpy(), col.numpy())), (N, N))
    return out


def from_scipy_sparse_matrix(
        A: scipy.sparse.spmatrix) -> Tuple[Tensor, Tensor]:
    r"""Converts a scipy sparse matrix to edge indices and edge attributes.

    Args:
        A (scipy.sparse): A sparse matrix.

    Examples:

        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> adj = to_scipy_sparse_matrix(edge_index)
        >>> # `edge_index` and `edge_weight` are both returned
        >>> from_scipy_sparse_matrix(adj)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([1., 1., 1., 1., 1., 1.]))
    """
    A = A.tocoo()
    row = torch.from_numpy(A.row).to(torch.long)
    col = torch.from_numpy(A.col).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = torch.from_numpy(A.data)
    return edge_index, edge_weight


def to_networkx(
    data: 'torch_geometric.data.Data',
    node_attrs: Optional[Iterable[str]] = None,
    edge_attrs: Optional[Iterable[str]] = None,
    graph_attrs: Optional[Iterable[str]] = None,
    to_undirected: Optional[Union[bool, str]] = False,
    remove_self_loops: bool = False,
) -> Any:
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        graph_attrs (iterable of str, optional): The graph attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool or str, optional): If set to :obj:`True` or
            "upper", will return a :obj:`networkx.Graph` instead of a
            :obj:`networkx.DiGraph`. The undirected graph will correspond to
            the upper triangle of the corresponding adjacency matrix.
            Similarly, if set to "lower", the undirected graph will correspond
            to the lower triangle of the adjacency matrix. (default:
            :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)

    Examples:

        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Data(edge_index=edge_index, num_nodes=4)
        >>> to_networkx(data)
        <networkx.classes.digraph.DiGraph at 0x2713fdb40d0>

    """
    import networkx as nx

    G = nx.Graph() if to_undirected else nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    node_attrs = node_attrs or []
    edge_attrs = edge_attrs or []
    graph_attrs = graph_attrs or []

    values = {}
    for key, value in data(*(node_attrs + edge_attrs + graph_attrs)):
        if torch.is_tensor(value):
            value = value if value.dim() <= 1 else value.squeeze(-1)
            values[key] = value.tolist()
        else:
            values[key] = value

    to_undirected = "upper" if to_undirected is True else to_undirected
    to_undirected_upper = True if to_undirected == "upper" else False
    to_undirected_lower = True if to_undirected == "lower" else False

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected_upper and u > v:
            continue
        elif to_undirected_lower and u < v:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)

        for key in edge_attrs:
            G[u][v][key] = values[key][i]

    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    for key in graph_attrs:
        G.graph[key] = values[key]

    return G


def from_networkx(
    G: Any,
    group_node_attrs: Optional[Union[List[str], all]] = None,
    group_edge_attrs: Optional[Union[List[str], all]] = None,
) -> 'torch_geometric.data.Data':
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or all, optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or all, optional): The edge attributes to
            be concatenated and added to :obj:`data.edge_attr`.
            (default: :obj:`None`)

    .. note::

        All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
        be numeric.

    Examples:

        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Data(edge_index=edge_index, num_nodes=4)
        >>> g = to_networkx(data)
        >>> # A `Data` object is returned
        >>> from_networkx(g)
        Data(edge_index=[2, 6], num_nodes=4)
    """
    import networkx as nx

    from torch_geometric.data import Data

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G

    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        edges = list(G.edges(keys=False))
    else:
        edges = list(G.edges)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError):
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data


def to_trimesh(data):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`trimesh.Trimesh`.

    Args:
        data (torch_geometric.data.Data): The data object.

    Example:

        >>> pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        ...                    dtype=torch.float)
        >>> face = torch.tensor([[0, 1, 2], [1, 2, 3]]).t()

        >>> data = Data(pos=pos, face=face)
        >>> to_trimesh(data)
        <trimesh.Trimesh(vertices.shape=(4, 3), faces.shape=(2, 3))>
    """
    import trimesh
    return trimesh.Trimesh(vertices=data.pos.detach().cpu().numpy(),
                           faces=data.face.detach().t().cpu().numpy(),
                           process=False)


def from_trimesh(mesh):
    r"""Converts a :obj:`trimesh.Trimesh` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mesh (trimesh.Trimesh): A :obj:`trimesh` mesh.

Example:

    Example:

        >>> pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
        ...                    dtype=torch.float)
        >>> face = torch.tensor([[0, 1, 2], [1, 2, 3]]).t()

        >>> data = Data(pos=pos, face=face)
        >>> mesh = to_trimesh(data)
        >>> from_trimesh(mesh)
        Data(pos=[4, 3], face=[3, 2])
    """
    from torch_geometric.data import Data

    pos = torch.from_numpy(mesh.vertices).to(torch.float)
    face = torch.from_numpy(mesh.faces).t().contiguous()

    return Data(pos=pos, face=face)


def to_cugraph(edge_index: Tensor, edge_weight: Optional[Tensor] = None,
               relabel_nodes: bool = True):
    r"""Converts a graph given by :obj:`edge_index` and optional
    :obj:`edge_weight` into a :obj:`cugraph` graph object.

    Args:
        relabel_nodes (bool, optional): If set to :obj:`True`,
            :obj:`cugraph` will remove any isolated nodes, leading to a
            relabeling of nodes. (default: :obj:`True`)
    """
    import cudf
    import cugraph

    df = cudf.from_dlpack(to_dlpack(edge_index.t()))

    if edge_weight is not None:
        assert edge_weight.dim() == 1
        df[2] = cudf.from_dlpack(to_dlpack(edge_weight))

    return cugraph.from_cudf_edgelist(
        df, source=0, destination=1,
        edge_attr=2 if edge_weight is not None else None,
        renumber=relabel_nodes)


def from_cugraph(G) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Converts a :obj:`cugraph` graph object into :obj:`edge_index` and
    optional :obj:`edge_weight` tensors.
    """
    df = G.edgelist.edgelist_df

    src = from_dlpack(df['src'].to_dlpack()).long()
    dst = from_dlpack(df['dst'].to_dlpack()).long()
    edge_index = torch.stack([src, dst], dim=0)

    edge_weight = None
    if 'weights' in df:
        edge_weight = from_dlpack(df['weights'].to_dlpack())

    return edge_index, edge_weight
