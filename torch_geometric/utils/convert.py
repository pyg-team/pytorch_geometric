from collections import defaultdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.dlpack import from_dlpack, to_dlpack

import torch_geometric
from torch_geometric.utils.num_nodes import maybe_num_nodes


def to_scipy_sparse_matrix(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Any:
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
    import scipy.sparse as sp

    row, col = edge_index.cpu()

    if edge_attr is None:
        edge_attr = torch.ones(row.size(0), device="cpu")
    else:
        edge_attr = edge_attr.view(-1).cpu()
        assert edge_attr.size(0) == row.size(0)

    N = maybe_num_nodes(edge_index, num_nodes)
    out = sp.coo_matrix(  #
        (edge_attr.numpy(), (row.numpy(), col.numpy())), (N, N))
    return out


def from_scipy_sparse_matrix(A: Any) -> Tuple[Tensor, Tensor]:
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
    data: Union[
        'torch_geometric.data.Data',
        'torch_geometric.data.HeteroData',
    ],
    node_attrs: Optional[Iterable[str]] = None,
    edge_attrs: Optional[Iterable[str]] = None,
    graph_attrs: Optional[Iterable[str]] = None,
    to_undirected: Optional[Union[bool, str]] = False,
    to_multi: bool = False,
    remove_self_loops: bool = False,
) -> Any:
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData): A
            homogeneous or heterogeneous data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        graph_attrs (iterable of str, optional): The graph attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool or str, optional): If set to :obj:`True`, will
            return a :class:`networkx.Graph` instead of a
            :class:`networkx.DiGraph`.
            By default, will include all edges and make them undirected.
            If set to :obj:`"upper"`, the undirected graph will only correspond
            to the upper triangle of the input adjacency matrix.
            If set to :obj:`"lower"`, the undirected graph will only correspond
            to the lower triangle of the input adjacency matrix.
            Only applicable in case the :obj:`data` object holds a homogeneous
            graph. (default: :obj:`False`)
        to_multi (bool, optional): if set to :obj:`True`, will return a
            :class:`networkx.MultiGraph` or a :class:`networkx:MultiDiGraph`
            (depending on the :obj:`to_undirected` option), which will not drop
            duplicated edges that may exist in :obj:`data`.
            (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self-loops in the resulting graph. (default: :obj:`False`)

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

    from torch_geometric.data import HeteroData

    to_undirected_upper: bool = to_undirected == 'upper'
    to_undirected_lower: bool = to_undirected == 'lower'

    to_undirected = to_undirected is True
    to_undirected |= to_undirected_upper or to_undirected_lower
    assert isinstance(to_undirected, bool)

    if isinstance(data, HeteroData) and to_undirected:
        raise ValueError("'to_undirected' is not supported in "
                         "'to_networkx' for heterogeneous graphs")

    if to_undirected:
        G = nx.MultiGraph() if to_multi else nx.Graph()
    else:
        G = nx.MultiDiGraph() if to_multi else nx.DiGraph()

    def to_networkx_value(value: Any) -> Any:
        return value.tolist() if isinstance(value, Tensor) else value

    for key in graph_attrs or []:
        G.graph[key] = to_networkx_value(data[key])

    node_offsets = data.node_offsets
    for node_store in data.node_stores:
        start = node_offsets[node_store._key]
        assert node_store.num_nodes is not None
        for i in range(node_store.num_nodes):
            node_kwargs: Dict[str, Any] = {}
            if isinstance(data, HeteroData):
                node_kwargs['type'] = node_store._key
            for key in node_attrs or []:
                node_kwargs[key] = to_networkx_value(node_store[key][i])

            G.add_node(start + i, **node_kwargs)

    for edge_store in data.edge_stores:
        for i, (v, w) in enumerate(edge_store.edge_index.t().tolist()):
            if to_undirected_upper and v > w:
                continue
            elif to_undirected_lower and v < w:
                continue
            elif remove_self_loops and v == w and not edge_store.is_bipartite(
            ):
                continue

            edge_kwargs: Dict[str, Any] = {}
            if isinstance(data, HeteroData):
                v = v + node_offsets[edge_store._key[0]]
                w = w + node_offsets[edge_store._key[-1]]
                edge_kwargs['type'] = edge_store._key
            for key in edge_attrs or []:
                edge_kwargs[key] = to_networkx_value(edge_store[key][i])

            G.add_edge(v, w, **edge_kwargs)

    return G


def from_networkx(
    G: Any,
    group_node_attrs: Optional[Union[List[str], Literal['all']]] = None,
    group_edge_attrs: Optional[Union[List[str], Literal['all']]] = None,
) -> 'torch_geometric.data.Data':
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
        group_node_attrs (List[str] or "all", optional): The node attributes to
            be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
        group_edge_attrs (List[str] or "all", optional): The edge attributes to
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

    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data_dict: Dict[str, Any] = defaultdict(list)
    data_dict['edge_index'] = edge_index

    node_attrs: List[str] = []
    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())

    edge_attrs: List[str] = []
    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())

    if group_node_attrs is not None and not isinstance(group_node_attrs, list):
        group_node_attrs = node_attrs

    if group_edge_attrs is not None and not isinstance(group_edge_attrs, list):
        group_edge_attrs = edge_attrs

    for _, feat_dict in G.nodes(data=True):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data_dict[str(key)].append(value)

    for _, _, feat_dict in G.edges(data=True):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data_dict[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data_dict[str(key)] = value

    for key, value in data_dict.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data_dict[key] = torch.stack(value, dim=0)
        else:
            try:
                data_dict[key] = torch.as_tensor(value)
            except Exception:
                pass

    data = Data.from_dict(data_dict)

    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

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


def to_networkit(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    directed: bool = True,
) -> Any:
    r"""Converts a :obj:`(edge_index, edge_weight)` tuple to a
    :class:`networkit.Graph`.

    Args:
        edge_index (torch.Tensor): The edge indices of the graph.
        edge_weight (torch.Tensor, optional): The edge weights of the graph.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)
        directed (bool, optional): If set to :obj:`False`, the graph will be
            undirected. (default: :obj:`True`)
    """
    import networkit as nk

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    g = nk.graph.Graph(
        num_nodes,
        weighted=edge_weight is not None,
        directed=directed,
    )

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))

    if not directed:
        mask = edge_index[0] <= edge_index[1]
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]

    for (u, v), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        g.addEdge(u, v, w)

    return g


def from_networkit(g: Any) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Converts a :class:`networkit.Graph` to a
    :obj:`(edge_index, edge_weight)` tuple.
    If the :class:`networkit.Graph` is not weighted, the returned
    :obj:`edge_weight` will be :obj:`None`.

    Args:
        g (networkkit.graph.Graph): A :obj:`networkit` graph object.
    """
    is_directed = g.isDirected()
    is_weighted = g.isWeighted()

    edge_indices, edge_weights = [], []
    for u, v, w in g.iterEdgesWeights():
        edge_indices.append([u, v])
        edge_weights.append(w)
        if not is_directed:
            edge_indices.append([v, u])
            edge_weights.append(w)

    edge_index = torch.tensor(edge_indices).t().contiguous()
    edge_weight = torch.tensor(edge_weights) if is_weighted else None

    return edge_index, edge_weight


def to_trimesh(data: 'torch_geometric.data.Data') -> Any:
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

    assert data.pos is not None
    assert data.face is not None

    return trimesh.Trimesh(
        vertices=data.pos.detach().cpu().numpy(),
        faces=data.face.detach().t().cpu().numpy(),
        process=False,
    )


def from_trimesh(mesh: Any) -> 'torch_geometric.data.Data':
    r"""Converts a :obj:`trimesh.Trimesh` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        mesh (trimesh.Trimesh): A :obj:`trimesh` mesh.

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


def to_cugraph(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    relabel_nodes: bool = True,
    directed: bool = True,
) -> Any:
    r"""Converts a graph given by :obj:`edge_index` and optional
    :obj:`edge_weight` into a :obj:`cugraph` graph object.

    Args:
        edge_index (torch.Tensor): The edge indices of the graph.
        edge_weight (torch.Tensor, optional): The edge weights of the graph.
            (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`,
            :obj:`cugraph` will remove any isolated nodes, leading to a
            relabeling of nodes. (default: :obj:`True`)
        directed (bool, optional): If set to :obj:`False`, the graph will be
            undirected. (default: :obj:`True`)
    """
    import cudf
    import cugraph

    g = cugraph.Graph(directed=directed)
    df = cudf.from_dlpack(to_dlpack(edge_index.t()))

    df = cudf.DataFrame({
        'source':
        cudf.from_dlpack(to_dlpack(edge_index[0])),
        'destination':
        cudf.from_dlpack(to_dlpack(edge_index[1])),
    })

    if edge_weight is not None:
        assert edge_weight.dim() == 1
        df['weight'] = cudf.from_dlpack(to_dlpack(edge_weight))

    g.from_cudf_edgelist(
        df,
        source='source',
        destination='destination',
        edge_attr='weight' if edge_weight is not None else None,
        renumber=relabel_nodes,
    )

    return g


def from_cugraph(g: Any) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Converts a :obj:`cugraph` graph object into :obj:`edge_index` and
    optional :obj:`edge_weight` tensors.

    Args:
        g (cugraph.Graph): A :obj:`cugraph` graph object.
    """
    df = g.view_edge_list()

    src = from_dlpack(df[g.source_columns].to_dlpack()).long()
    dst = from_dlpack(df[g.destination_columns].to_dlpack()).long()
    edge_index = torch.stack([src, dst], dim=0)

    edge_weight = None
    if g.weight_column is not None:
        edge_weight = from_dlpack(df[g.weight_column].to_dlpack())

    return edge_index, edge_weight


def to_dgl(
    data: Union['torch_geometric.data.Data', 'torch_geometric.data.HeteroData']
) -> Any:
    r"""Converts a :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData` instance to a :obj:`dgl` graph
    object.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The data object.

    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 4, 4]])
        >>> x = torch.randn(5, 3)
        >>> edge_attr = torch.randn(6, 2)
        >>> data = Data(x=x, edge_index=edge_index, edge_attr=y)
        >>> g = to_dgl(data)
        >>> g
        Graph(num_nodes=5, num_edges=6,
            ndata_schemes={'x': Scheme(shape=(3,))}
            edata_schemes={'edge_attr': Scheme(shape=(2, ))})

        >>> data = HeteroData()
        >>> data['paper'].x = torch.randn(5, 3)
        >>> data['author'].x = torch.ones(5, 3)
        >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        >>> data['author', 'cites', 'paper'].edge_index = edge_index
        >>> g = to_dgl(data)
        >>> g
        Graph(num_nodes={'author': 5, 'paper': 5},
            num_edges={('author', 'cites', 'paper'): 5},
            metagraph=[('author', 'paper', 'cites')])
    """
    import dgl

    from torch_geometric.data import Data, HeteroData

    if isinstance(data, Data):
        if data.edge_index is not None:
            row, col = data.edge_index
        elif 'adj' in data:
            row, col, _ = data.adj.coo()
        elif 'adj_t' in data:
            row, col, _ = data.adj_t.t().coo()
        else:
            row, col = [], []

        g = dgl.graph((row, col), num_nodes=data.num_nodes)

        for attr in data.node_attrs():
            g.ndata[attr] = data[attr]
        for attr in data.edge_attrs():
            if attr in ['edge_index', 'adj_t']:
                continue
            g.edata[attr] = data[attr]

        return g

    if isinstance(data, HeteroData):
        data_dict = {}
        for edge_type, edge_store in data.edge_items():
            if edge_store.get('edge_index') is not None:
                row, col = edge_store.edge_index
            else:
                row, col, _ = edge_store['adj_t'].t().coo()

            data_dict[edge_type] = (row, col)

        g = dgl.heterograph(data_dict)

        for node_type, node_store in data.node_items():
            for attr, value in node_store.items():
                g.nodes[node_type].data[attr] = value

        for edge_type, edge_store in data.edge_items():
            for attr, value in edge_store.items():
                if attr in ['edge_index', 'adj_t']:
                    continue
                g.edges[edge_type].data[attr] = value

        return g

    raise ValueError(f"Invalid data type (got '{type(data)}')")


def from_dgl(
    g: Any,
) -> Union['torch_geometric.data.Data', 'torch_geometric.data.HeteroData']:
    r"""Converts a :obj:`dgl` graph object to a
    :class:`torch_geometric.data.Data` or
    :class:`torch_geometric.data.HeteroData` instance.

    Args:
        g (dgl.DGLGraph): The :obj:`dgl` graph object.

    Example:
        >>> g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))
        >>> g.ndata['x'] = torch.randn(g.num_nodes(), 3)
        >>> g.edata['edge_attr'] = torch.randn(g.num_edges(), 2)
        >>> data = from_dgl(g)
        >>> data
        Data(x=[6, 3], edge_attr=[4, 2], edge_index=[2, 4])

        >>> g = dgl.heterograph({
        >>> g = dgl.heterograph({
        ...     ('author', 'writes', 'paper'): ([0, 1, 1, 2, 3, 3, 4],
        ...                                     [0, 0, 1, 1, 1, 2, 2])})
        >>> g.nodes['author'].data['x'] = torch.randn(5, 3)
        >>> g.nodes['paper'].data['x'] = torch.randn(5, 3)
        >>> data = from_dgl(g)
        >>> data
        HeteroData(
        author={ x=[5, 3] },
        paper={ x=[3, 3] },
        (author, writes, paper)={ edge_index=[2, 7] }
        )
    """
    import dgl

    from torch_geometric.data import Data, HeteroData

    if not isinstance(g, dgl.DGLGraph):
        raise ValueError(f"Invalid data type (got '{type(g)}')")

    data: Union[Data, HeteroData]

    if g.is_homogeneous:
        data = Data()
        data.edge_index = torch.stack(g.edges(), dim=0)

        for attr, value in g.ndata.items():
            data[attr] = value
        for attr, value in g.edata.items():
            data[attr] = value

        return data

    data = HeteroData()

    for node_type in g.ntypes:
        for attr, value in g.nodes[node_type].data.items():
            data[node_type][attr] = value

    for edge_type in g.canonical_etypes:
        row, col = g.edges(form="uv", etype=edge_type)
        data[edge_type].edge_index = torch.stack([row, col], dim=0)
        for attr, value in g.edge_attr_schemes(edge_type).items():
            data[edge_type][attr] = value

    return data
