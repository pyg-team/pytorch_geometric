import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.testing import get_random_edge_index, withPackage
from torch_geometric.utils import (
    from_cugraph,
    from_dgl,
    from_hetero_networkx,
    from_networkit,
    from_networkx,
    from_scipy_sparse_matrix,
    from_trimesh,
    sort_edge_index,
    subgraph,
    to_cugraph,
    to_dgl,
    to_networkit,
    to_networkx,
    to_scipy_sparse_matrix,
    to_trimesh,
)


@withPackage('scipy')
def test_to_scipy_sparse_matrix():
    import scipy.sparse as sp

    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])

    adj = to_scipy_sparse_matrix(edge_index)
    assert isinstance(adj, sp.coo_matrix)
    assert adj.shape == (2, 2)
    assert adj.row.tolist() == edge_index[0].tolist()
    assert adj.col.tolist() == edge_index[1].tolist()
    assert adj.data.tolist() == [1, 1, 1]

    edge_attr = torch.tensor([1.0, 2.0, 3.0])
    adj = to_scipy_sparse_matrix(edge_index, edge_attr)
    assert isinstance(adj, sp.coo_matrix)
    assert adj.shape == (2, 2)
    assert adj.row.tolist() == edge_index[0].tolist()
    assert adj.col.tolist() == edge_index[1].tolist()
    assert adj.data.tolist() == edge_attr.tolist()


@withPackage('scipy')
def test_from_scipy_sparse_matrix():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    adj = to_scipy_sparse_matrix(edge_index)

    out = from_scipy_sparse_matrix(adj)
    assert out[0].tolist() == edge_index.tolist()
    assert out[1].tolist() == [1, 1, 1]


@withPackage('networkx')
def test_to_networkx():
    import networkx as nx

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pos = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_attr = torch.tensor([1.0, 2.0, 3.0])
    data = Data(x=x, pos=pos, edge_index=edge_index, weight=edge_attr)

    for remove_self_loops in [True, False]:
        G = to_networkx(data, node_attrs=['x', 'pos'], edge_attrs=['weight'],
                        remove_self_loops=remove_self_loops)

        assert G.nodes[0]['x'] == [1.0, 2.0]
        assert G.nodes[1]['x'] == [3.0, 4.0]
        assert G.nodes[0]['pos'] == [0.0, 0.0]
        assert G.nodes[1]['pos'] == [1.0, 1.0]

        if remove_self_loops:
            assert nx.to_numpy_array(G).tolist() == [[0.0, 1.0], [2.0, 0.0]]
        else:
            assert nx.to_numpy_array(G).tolist() == [[3.0, 1.0], [2.0, 0.0]]


@withPackage('networkx')
def test_from_networkx_set_node_attributes():
    import networkx as nx

    G = nx.path_graph(3)
    attrs = {
        0: {
            'x': torch.tensor([1, 0, 0])
        },
        1: {
            'x': torch.tensor([0, 1, 0])
        },
        2: {
            'x': torch.tensor([0, 0, 1])
        },
    }
    nx.set_node_attributes(G, attrs)

    assert from_networkx(G).x.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


@withPackage('networkx')
def test_to_networkx_undirected():
    import networkx as nx

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pos = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_attr = torch.tensor([1.0, 2.0, 3.0])
    data = Data(x=x, pos=pos, edge_index=edge_index, weight=edge_attr)

    for remove_self_loops in [True, False]:
        G = to_networkx(
            data,
            node_attrs=['x', 'pos'],
            edge_attrs=['weight'],
            remove_self_loops=remove_self_loops,
            to_undirected=True,
        )

        assert G.nodes[0]['x'] == [1, 2]
        assert G.nodes[1]['x'] == [3, 4]
        assert G.nodes[0]['pos'] == [0, 0]
        assert G.nodes[1]['pos'] == [1, 1]

        if remove_self_loops:
            assert nx.to_numpy_array(G).tolist() == [[0, 2], [2, 0]]
        else:
            assert nx.to_numpy_array(G).tolist() == [[3, 2], [2, 0]]

    G = to_networkx(data, edge_attrs=['weight'], to_undirected=False)
    assert nx.to_numpy_array(G).tolist() == [[3, 1], [2, 0]]

    G = to_networkx(data, edge_attrs=['weight'], to_undirected='upper')
    assert nx.to_numpy_array(G).tolist() == [[3, 1], [1, 0]]

    G = to_networkx(data, edge_attrs=['weight'], to_undirected='lower')
    assert nx.to_numpy_array(G).tolist() == [[3, 2], [2, 0]]


@withPackage('networkx')
def test_to_networkx_undirected_options():
    import networkx as nx
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 0]])
    data = Data(edge_index=edge_index, num_nodes=3)

    G = to_networkx(data, to_undirected=True)
    assert nx.to_numpy_array(G).tolist() == [[0, 1, 1], [1, 0, 1], [1, 1, 0]]

    G = to_networkx(data, to_undirected='upper')
    assert nx.to_numpy_array(G).tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    G = to_networkx(data, to_undirected='lower')
    assert nx.to_numpy_array(G).tolist() == [[0, 1, 1], [1, 0, 0], [1, 0, 0]]


@withPackage('networkx')
def test_to_networkx_hetero():
    edge_index = get_random_edge_index(5, 10, 20, coalesce=True)

    data = HeteroData()
    data['global_id'] = 0
    data['author'].x = torch.arange(5)
    data['paper'].x = torch.arange(10)
    data['author', 'paper'].edge_index = edge_index
    data['author', 'paper'].edge_attr = torch.arange(edge_index.size(1))

    G = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'],
                    graph_attrs=['global_id'])

    assert G.number_of_nodes() == 15
    assert G.number_of_edges() == edge_index.size(1)

    assert G.graph == {'global_id': 0}

    for i, (v, data) in enumerate(G.nodes(data=True)):
        assert i == v
        assert len(data) == 2
        if i < 5:
            assert data['x'] == i
            assert data['type'] == 'author'
        else:
            assert data['x'] == i - 5
            assert data['type'] == 'paper'

    for i, (v, w, data) in enumerate(G.edges(data=True)):
        assert v == int(edge_index[0, i])
        assert w == int(edge_index[1, i]) + 5
        assert len(data) == 2
        assert data['type'] == ('author', 'to', 'paper')
        assert data['edge_attr'] == i


@withPackage('networkx')
def test_from_networkx():
    x = torch.randn(2, 8)
    pos = torch.randn(2, 3)
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_attr = torch.randn(edge_index.size(1))
    perm = torch.tensor([0, 2, 1])
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    G = to_networkx(data, node_attrs=['x', 'pos'], edge_attrs=['edge_attr'])
    data = from_networkx(G)
    assert len(data) == 4
    assert data.x.tolist() == x.tolist()
    assert data.pos.tolist() == pos.tolist()
    assert data.edge_index.tolist() == edge_index[:, perm].tolist()
    assert data.edge_attr.tolist() == edge_attr[perm].tolist()


@withPackage('networkx')
def test_from_networkx_group_attrs():
    x = torch.randn(2, 2)
    x1 = torch.randn(2, 4)
    x2 = torch.randn(2, 8)
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_attr1 = torch.randn(edge_index.size(1))
    edge_attr2 = torch.randn(edge_index.size(1))
    perm = torch.tensor([0, 2, 1])
    data = Data(x=x, x1=x1, x2=x2, edge_index=edge_index,
                edge_attr1=edge_attr1, edge_attr2=edge_attr2)
    G = to_networkx(data, node_attrs=['x', 'x1', 'x2'],
                    edge_attrs=['edge_attr1', 'edge_attr2'])
    data = from_networkx(G, group_node_attrs=['x', 'x2'], group_edge_attrs=all)
    assert len(data) == 4
    assert data.x.tolist() == torch.cat([x, x2], dim=-1).tolist()
    assert data.x1.tolist() == x1.tolist()
    assert data.edge_index.tolist() == edge_index[:, perm].tolist()
    assert data.edge_attr.tolist() == torch.stack([edge_attr1, edge_attr2],
                                                  dim=-1)[perm].tolist()


@withPackage('networkx')
def test_networkx_vice_versa_convert():
    import networkx as nx

    G = nx.complete_graph(5)
    assert G.is_directed() is False
    data = from_networkx(G)
    assert data.is_directed() is False
    G = to_networkx(data)
    assert G.is_directed() is True
    G = nx.to_undirected(G)
    assert G.is_directed() is False


@withPackage('networkx')
def test_from_networkx_non_consecutive():
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(4)
    graph.add_node(2)
    graph.add_edge(4, 2)
    for node in graph.nodes():
        graph.nodes[node]['x'] = node

    data = from_networkx(graph)
    assert len(data) == 2
    assert data.x.tolist() == [4, 2]
    assert data.edge_index.tolist() == [[0, 1], [1, 0]]


@withPackage('networkx')
def test_from_networkx_inverse():
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(3)
    graph.add_node(2)
    graph.add_node(1)
    graph.add_node(0)
    graph.add_edge(3, 1)
    graph.add_edge(2, 1)
    graph.add_edge(1, 0)

    data = from_networkx(graph)
    assert len(data) == 2
    assert data.edge_index.tolist() == [[0, 1, 2, 2, 2, 3], [2, 2, 0, 1, 3, 2]]
    assert data.num_nodes == 4


@withPackage('networkx')
def test_from_networkx_non_numeric_labels():
    import networkx as nx

    graph = nx.Graph()
    graph.add_node('4')
    graph.add_node('2')
    graph.add_edge('4', '2')
    for node in graph.nodes():
        graph.nodes[node]['x'] = node
    data = from_networkx(graph)
    assert len(data) == 2
    assert data.x == ['4', '2']
    assert data.edge_index.tolist() == [[0, 1], [1, 0]]


@withPackage('networkx')
def test_from_networkx_without_edges():
    import networkx as nx

    graph = nx.Graph()
    graph.add_node(1)
    graph.add_node(2)
    data = from_networkx(graph)
    assert len(data) == 2
    assert data.edge_index.size() == (2, 0)
    assert data.num_nodes == 2


@withPackage('networkx')
def test_from_networkx_with_same_node_and_edge_attributes():
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from([(0, {'age': 1}), (1, {'age': 6}), (2, {'age': 5})])
    G.add_edges_from([(0, 1, {'age': 2}), (1, 2, {'age': 7})])

    data = from_networkx(G)
    assert len(data) == 4
    assert data.age.tolist() == [1, 6, 5]
    assert data.num_nodes == 3
    assert data.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert data.edge_age.tolist() == [2, 2, 7, 7]

    data = from_networkx(G, group_node_attrs=all, group_edge_attrs=all)
    assert len(data) == 3
    assert data.x.tolist() == [[1], [6], [5]]
    assert data.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert data.edge_attr.tolist() == [[2], [2], [7], [7]]


@withPackage('networkx')
def test_from_networkx_subgraph_convert():
    import networkx as nx

    G = nx.complete_graph(5)

    edge_index = from_networkx(G).edge_index
    sub_edge_index_1, _ = subgraph([0, 1, 3, 4], edge_index,
                                   relabel_nodes=True)

    sub_edge_index_2 = from_networkx(G.subgraph([0, 1, 3, 4])).edge_index

    assert sub_edge_index_1.tolist() == sub_edge_index_2.tolist()


@withPackage('networkx')
@pytest.mark.parametrize('n', [100])
@pytest.mark.parametrize('p', [0.8])
@pytest.mark.parametrize('q', [0.2])
def test_from_networkx_sbm(n, p, q):
    import networkx as nx
    G = nx.stochastic_block_model(
        sizes=[n // 2, n // 2],
        p=[[p, q], [q, p]],
        seed=0,
        directed=False,
    )

    data = from_networkx(G)
    assert data.num_nodes == 100
    assert torch.equal(data.block[:50], data.block.new_zeros(50))
    assert torch.equal(data.block[50:], data.block.new_ones(50))


@withPackage('networkit')
def test_to_networkit_vice_versa():
    edge_index = torch.tensor([[0, 1], [1, 0]])

    g = to_networkit(edge_index, directed=False)
    assert not g.isDirected()
    assert not g.isWeighted()

    edge_index, edge_weight = from_networkit(g)
    assert edge_index.tolist() == [[0, 1], [1, 0]]
    assert edge_weight is None


@withPackage('networkit')
@pytest.mark.parametrize('directed', [True, False])
@pytest.mark.parametrize('num_nodes', [None, 3])
@pytest.mark.parametrize('edge_weight', [None, torch.rand(3)])
def test_to_networkit(directed, edge_weight, num_nodes):
    import networkit

    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    g = to_networkit(edge_index, edge_weight, num_nodes, directed)

    assert isinstance(g, networkit.Graph)
    assert g.isDirected() == directed
    assert g.numberOfNodes() == 3

    if edge_weight is None:
        edge_weight = torch.tensor([1., 1., 1.])

    assert g.weight(0, 1) == float(edge_weight[0])
    assert g.weight(1, 2) == float(edge_weight[2])

    if directed:
        assert g.numberOfEdges() == 3
        assert g.weight(1, 0) == float(edge_weight[1])
    else:
        assert g.numberOfEdges() == 2


@pytest.mark.parametrize('directed', [True, False])
@pytest.mark.parametrize('weighted', [True, False])
@withPackage('networkit')
def test_from_networkit(directed, weighted):
    import networkit

    g = networkit.Graph(3, weighted=weighted, directed=directed)
    g.addEdge(0, 1)
    g.addEdge(1, 2)
    if directed:
        g.addEdge(1, 0)

    if weighted:
        for i, (u, v) in enumerate(g.iterEdges()):
            g.setWeight(u, v, i + 1)

    edge_index, edge_weight = from_networkit(g)

    if directed:
        assert edge_index.tolist() == [[0, 1, 1], [1, 2, 0]]
        if weighted:
            assert edge_weight.tolist() == [1, 2, 3]
        else:
            assert edge_weight is None
    else:
        assert edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
        if weighted:
            assert edge_weight.tolist() == [1, 1, 2, 2]
        else:
            assert edge_weight is None


@withPackage('trimesh')
def test_trimesh_vice_versa():
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                       dtype=torch.float)
    face = torch.tensor([[0, 1, 2], [1, 2, 3]]).t()

    data = Data(pos=pos, face=face)
    mesh = to_trimesh(data)
    data = from_trimesh(mesh)

    assert pos.tolist() == data.pos.tolist()
    assert face.tolist() == data.face.tolist()


@withPackage('trimesh')
def test_to_trimesh():
    import trimesh

    pos = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    face = torch.tensor([[0, 1, 2], [2, 1, 3]]).t()
    data = Data(pos=pos, face=face)

    obj = to_trimesh(data)

    assert isinstance(obj, trimesh.Trimesh)
    assert obj.vertices.shape == (4, 3)
    assert obj.faces.shape == (2, 3)
    assert obj.vertices.tolist() == data.pos.tolist()
    assert obj.faces.tolist() == data.face.t().contiguous().tolist()


@withPackage('trimesh')
def test_from_trimesh():
    import trimesh

    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    faces = [[0, 1, 2]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    data = from_trimesh(mesh)

    assert data.pos.tolist() == vertices
    assert data.face.t().contiguous().tolist() == faces


@withPackage('cudf', 'cugraph')
@pytest.mark.parametrize('edge_weight', [None, torch.rand(4)])
@pytest.mark.parametrize('relabel_nodes', [True, False])
@pytest.mark.parametrize('directed', [True, False])
def test_to_cugraph(edge_weight, directed, relabel_nodes):
    import cugraph

    if directed:
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    else:
        edge_index = torch.tensor([[0, 1], [1, 2]])

    if edge_weight is not None:
        edge_weight = edge_weight[:edge_index.size(1)]

    graph = to_cugraph(edge_index, edge_weight, relabel_nodes, directed)
    assert isinstance(graph, cugraph.Graph)
    assert graph.number_of_nodes() == 3

    edge_list = graph.view_edge_list()
    assert edge_list is not None

    edge_list = edge_list.sort_values(by=[0, 1])

    cu_edge_index = edge_list[[0, 1]].to_pandas().values
    cu_edge_index = torch.from_numpy(cu_edge_index).t()
    cu_edge_weight = None
    if edge_weight is not None:
        cu_edge_weight = edge_list['2'].to_pandas().values
        cu_edge_weight = torch.from_numpy(cu_edge_weight)

    cu_edge_index, cu_edge_weight = sort_edge_index(cu_edge_index,
                                                    cu_edge_weight)

    assert torch.equal(edge_index, cu_edge_index.cpu())
    if edge_weight is not None:
        assert torch.allclose(edge_weight, cu_edge_weight.cpu())


@withPackage('cudf', 'cugraph')
@pytest.mark.parametrize('edge_weight', [None, torch.randn(4)])
@pytest.mark.parametrize('directed', [True, False])
@pytest.mark.parametrize('relabel_nodes', [True, False])
def test_from_cugraph(edge_weight, directed, relabel_nodes):
    import cudf
    import cugraph
    from torch.utils.dlpack import to_dlpack

    if directed:
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    else:
        edge_index = torch.tensor([[0, 1], [1, 2]])

    if edge_weight is not None:
        edge_weight = edge_weight[:edge_index.size(1)]

    G = cugraph.Graph(directed=directed)
    df = cudf.from_dlpack(to_dlpack(edge_index.t()))
    if edge_weight is not None:
        df['2'] = cudf.from_dlpack(to_dlpack(edge_weight))

    G.from_cudf_edgelist(
        df,
        source=0,
        destination=1,
        edge_attr='2' if edge_weight is not None else None,
        renumber=relabel_nodes,
    )

    cu_edge_index, cu_edge_weight = from_cugraph(G)
    cu_edge_index, cu_edge_weight = sort_edge_index(cu_edge_index,
                                                    cu_edge_weight)

    assert torch.equal(edge_index, cu_edge_index.cpu())
    if edge_weight is not None:
        assert torch.allclose(edge_weight, cu_edge_weight.cpu())
    else:
        assert cu_edge_weight is None


@withPackage('dgl')
def test_to_dgl_graph():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 4, 4]])
    edge_attr = torch.randn(edge_index.size(1), 2)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    g = to_dgl(data)

    assert torch.equal(data.x, g.ndata['x'])
    row, col = g.edges()
    assert torch.equal(row, edge_index[0])
    assert torch.equal(col, edge_index[1])
    assert torch.equal(data.edge_attr, g.edata['edge_attr'])


@withPackage('dgl')
def test_to_dgl_hetero_graph():
    data = HeteroData()
    data['v1'].x = torch.randn(4, 3)
    data['v2'].x = torch.randn(4, 3)
    data['v1', 'v2'].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    data['v1', 'v2'].edge_attr = torch.randn(4, 2)

    g = to_dgl(data)

    assert data['v1', 'v2'].num_edges == g.num_edges(('v1', 'to', 'v2'))
    assert data['v1'].num_nodes == g.num_nodes('v1')
    assert data['v2'].num_nodes == g.num_nodes('v2')
    assert torch.equal(data['v1'].x, g.nodes['v1'].data['x'])
    assert torch.equal(data['v2'].x, g.nodes['v2'].data['x'])
    row, col = g.edges()
    assert torch.equal(row, data['v1', 'v2'].edge_index[0])
    assert torch.equal(col, data['v1', 'v2'].edge_index[1])
    assert torch.equal(g.edata['edge_attr'], data['v1', 'v2'].edge_attr)


@withPackage('dgl', 'torch_sparse')
def test_to_dgl_sparse():
    from torch_geometric.transforms import ToSparseTensor
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 4, 4]])
    edge_attr = torch.randn(edge_index.size(1), 2)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data = ToSparseTensor()(data)

    g = to_dgl(data)

    assert torch.equal(data.x, g.ndata["x"])
    pyg_row, pyg_col, _ = data.adj_t.t().coo()
    dgl_row, dgl_col = g.edges()
    assert torch.equal(pyg_row, dgl_row)
    assert torch.equal(pyg_col, dgl_col)
    assert torch.equal(data.edge_attr, g.edata['edge_attr'])


@withPackage('dgl')
def test_from_dgl_graph():
    import dgl
    g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))
    g.ndata['x'] = torch.randn(g.num_nodes(), 3)
    g.edata['edge_attr'] = torch.randn(g.num_edges())

    data = from_dgl(g)

    assert torch.equal(data.x, g.ndata['x'])
    row, col = g.edges()
    assert torch.equal(data.edge_index[0], row)
    assert torch.equal(data.edge_index[1], col)
    assert torch.equal(data.edge_attr, g.edata['edge_attr'])


@withPackage('dgl')
def test_from_dgl_hetero_graph():
    import dgl
    g = dgl.heterograph({
        ('v1', 'to', 'v2'): (
            [0, 1, 1, 2, 3, 3, 4],
            [0, 0, 1, 1, 1, 2, 2],
        )
    })
    g.nodes['v1'].data['x'] = torch.randn(5, 3)
    g.nodes['v2'].data['x'] = torch.randn(3, 3)

    data = from_dgl(g)

    assert data['v1', 'v2'].num_edges == g.num_edges(('v1', 'to', 'v2'))
    assert data['v1'].num_nodes == g.num_nodes('v1')
    assert data['v2'].num_nodes == g.num_nodes('v2')
    assert torch.equal(data['v1'].x, g.nodes['v1'].data['x'])
    assert torch.equal(data['v2'].x, g.nodes['v2'].data['x'])


@withPackage('networkx')
def test_from_hetero_networkx():
    author_paper_edge_index = get_random_edge_index(5, 10, 20, coalesce=True)
    paper_author_edge_index = get_random_edge_index(10, 5, 10, coalesce=True)
    author_instit_edge_index = torch.tensor([[0, 1], [2, 2]])
    author_instit_edge_index_bis = torch.tensor([[0, 1], [1, 0]])
    graph_x = [torch.tensor([0, 1, 2])]

    data = HeteroData()
    data['global_id'] = 0
    data['graph_x'] = graph_x
    data['author'].x = torch.arange(5)
    data['paper'].x = torch.arange(10)
    data['institution'].x = torch.arange(3)
    data['author', 'paper'].edge_index = author_paper_edge_index
    data['paper', 'author'].edge_index = paper_author_edge_index
    data['author', 'affiliated_with',
         'institution'].edge_index = author_instit_edge_index
    data['author', 'affiliated_with_bis',
         'institution'].edge_index = author_instit_edge_index_bis
    data['author',
         'paper'].edge_attr = torch.arange(author_paper_edge_index.size(1))
    data['paper',
         'author'].edge_attr = torch.arange(paper_author_edge_index.size(1))
    data['author', 'affiliated_with', 'institution'].edge_attr = torch.arange(
        author_instit_edge_index.size(1))
    data['author', 'affiliated_with_bis',
         'institution'].edge_attr = torch.arange(
             author_instit_edge_index_bis.size(1))

    G = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'],
                    graph_attrs=['global_id', 'graph_x'])

    data = from_hetero_networkx(G, node_type_attribute="type",
                                edge_type_attribute="type")

    assert data['graph_x'].tolist() == [x.tolist() for x in graph_x]
    assert data['author'].x.tolist() == torch.arange(5).tolist()
    assert data['paper'].x.tolist() == torch.arange(10).tolist()
    assert data[(
        'author', 'to',
        'paper')].edge_index.tolist() == author_paper_edge_index.tolist()
    assert data[(
        'paper', 'to',
        'author')].edge_index.tolist() == paper_author_edge_index.tolist()
    assert data[('author', 'affiliated_with', 'institution'
                 )].edge_index.tolist() == author_instit_edge_index.tolist()
    t = data[('author', 'affiliated_with_bis', 'institution')]
    assert t.edge_index.tolist() == author_instit_edge_index_bis.tolist()
    assert data[('author', 'to', 'paper')].edge_attr.tolist() == torch.arange(
        author_paper_edge_index.size(1)).tolist()


@withPackage('networkx')
def test_from_hetero_networkx_set_node_attributes():
    import networkx as nx

    G = nx.path_graph(3)
    attrs = {
        0: {
            'y': torch.tensor([1, 0, 0]),
            'type': 'A'
        },
        1: {
            'x': torch.tensor([0, 1, 0]),
            'type': 'B'
        },
        2: {
            'x': torch.tensor([0, 0, 1]),
            'type': 'B'
        },
    }
    nx.set_node_attributes(G, attrs)

    data = from_hetero_networkx(G, node_type_attribute="type",
                                edge_type_attribute=None)

    assert data['A'].y.tolist() == [[1, 0, 0]]
    assert data['B'].x.tolist() == [[0, 1, 0], [0, 0, 1]]


@withPackage('networkx')
def test_from_hetero_networkx_inverse():
    import networkx as nx

    G = nx.DiGraph()
    G.add_node(3, type='A')
    G.add_node(2, type='A')
    G.add_node(1, type='B')
    G.add_node(0, type='C')
    G.add_edge(3, 1, type=('A', 'to', 'B'))
    G.add_edge(2, 1, type=('A', 'to', 'B'))
    G.add_edge(1, 0, type=('B', 'to', 'C'))

    data = from_hetero_networkx(G, node_type_attribute="type",
                                edge_type_attribute="type")

    assert data[('A', 'to', 'B')].edge_index.tolist() == [[0, 1], [0, 0]]
    assert data[('B', 'to', 'C')].edge_index.tolist() == [[0], [0]]


@withPackage('networkx')
def test_from_hetero_networkx_non_consecutive():
    import networkx as nx

    G = nx.Graph()
    G.add_node(4, type='A')
    G.add_node(2, type='A')
    G.add_edge(4, 2)
    for node in G.nodes():
        G.nodes[node]['x'] = node

    data = from_hetero_networkx(G, node_type_attribute="type",
                                edge_type_attribute=None)

    assert data["A"].x.tolist() == [4, 2]
    assert data[("A", "to", "A")].edge_index.tolist() == [[0, 1], [1, 0]]


@withPackage('networkx')
def test_from_hetero_networkx_non_numeric_labels():
    import networkx as nx

    G = nx.Graph()
    G.add_node('4', type='A')
    G.add_node('2', type='A')
    G.add_edge('4', '2')
    for node in G.nodes():
        G.nodes[node]['x'] = node

    data = from_hetero_networkx(G, node_type_attribute="type",
                                edge_type_attribute=None)

    assert data["A"].x == ['4', '2']
    assert data[("A", "to", "A")].edge_index.tolist() == [[0, 1], [1, 0]]


@withPackage('networkx')
def test_from_hetero_networkx_with_same_node_and_edge_attributes():
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from([(0, {
        'age': 1,
        'type': 'A'
    }), (1, {
        'age': 6,
        'type': 'A'
    }), (2, {
        'age': 5,
        'type': 'A'
    })])
    G.add_edges_from([(0, 1, {'age': 2}), (1, 2, {'age': 7})])

    data = from_hetero_networkx(G, node_type_attribute="type",
                                edge_type_attribute=None)

    assert data['A'].age.tolist() == [1, 6, 5]
    assert data[('A', 'to', 'A')].edge_index.tolist() == [[0, 1, 1, 2],
                                                          [1, 0, 2, 1]]
    assert data[('A', 'to', 'A')].age.tolist() == [2, 2, 7, 7]


@withPackage('networkx')
def test_from_hetero_networkx_raise_missing_node_type_attribute():
    import networkx as nx

    G = nx.Graph()
    G.add_node(0, type='A')
    G.add_node(1)
    G.add_edge(0, 1)

    with pytest.raises(
            KeyError,
            match=r'Given node_type_attribute: .* missing from node .*') as _:
        from_hetero_networkx(G, node_type_attribute="type",
                             edge_type_attribute=None)


@withPackage('networkx')
def test_from_hetero_networkx_raise_missing_edge_type_attribute():
    import networkx as nx

    G = nx.Graph()
    G.add_node(0, type='A')
    G.add_node(1, type='A')
    G.add_edge(0, 1)

    with pytest.raises(
            KeyError,
            match=r'Given edge_type_attribute: .* missing from edge .*') as _:
        from_hetero_networkx(G, node_type_attribute="type",
                             edge_type_attribute="type")


@withPackage('networkx')
def test_from_hetero_networkx_raise_different_edge_attribute():
    import networkx as nx

    G = nx.Graph()
    G.add_node(0, type='A')
    G.add_node(1, type='A')
    G.add_node(2, type='A')
    G.add_edge(0, 1, a=1)
    G.add_edge(0, 2, b=2)

    with pytest.raises(
            ValueError,
            match='Not all edges contain the same attributes.') as _:
        from_hetero_networkx(G, node_type_attribute="type",
                             edge_type_attribute=None)


@withPackage('networkx')
def test_from_hetero_networkx_raise_different_node_attribute():
    import networkx as nx

    G = nx.Graph()
    G.add_node(0, type='A', a=1)
    G.add_node(1, type='A', b=1)

    with pytest.raises(
            ValueError,
            match='Not all nodes contain the same attributes.') as _:
        from_hetero_networkx(G, node_type_attribute="type",
                             edge_type_attribute=None)


@withPackage('networkx')
def test_from_hetero_networkx_works_with_named_edge_types():
    import networkx as nx

    G = nx.DiGraph()
    G.add_node(0, type="A", x=1)
    G.add_node(1, type="A", x=1)
    G.add_node(2, type="B", y=1)
    G.add_edge(0, 1, type=('A', 'towards', 'A'), x=1)
    G.add_edge(0, 2, type=('A', 'towards', 'B'))

    data = from_hetero_networkx(G, node_type_attribute="type",
                                edge_type_attribute="type")

    assert data[('A', 'towards', 'A')].edge_index.tolist() == [[0], [1]]
    assert data[('A', 'towards', 'B')].edge_index.tolist() == [[0], [0]]


@withPackage('networkx')
def test_from_hetero_networkx_works_with_non_string_types():
    import networkx as nx

    G = nx.Graph()
    G.add_node(0, type=1, x=1)
    G.add_node(1, type=False, x=1)

    data = from_hetero_networkx(G, node_type_attribute="type",
                                edge_type_attribute=None)

    assert data[str(1)].x.tolist() == [1]
    assert data[str(False)].x.tolist() == [1]


@withPackage('networkx')
def test_from_hetero_networkx_graph_attrs_selection():

    data = HeteroData()
    data['x'] = 0
    data['y'] = 0
    data['z'] = 0

    G = to_networkx(data, node_attrs=[], edge_attrs=[],
                    graph_attrs=['x', 'y', 'z'])

    data = from_hetero_networkx(G, node_type_attribute="type",
                                edge_type_attribute=None,
                                graph_attrs=['x', 'y'])

    assert data['x'] == 0
    assert data['y'] == 0
    assert 'z' not in data


@withPackage('networkx')
def test_from_hetero_networkx_graph_node_selection():
    import networkx as nx

    G = nx.DiGraph()
    G.add_node(0, type="A", x=0)
    G.add_node("node_1", type="B", y=1)
    G.add_node(2, type="B", x=1)

    data = from_hetero_networkx(G, node_type_attribute="type",
                                edge_type_attribute=None,
                                graph_attrs=['x', 'y'], nodes=[0, "node_1"])

    assert len(data['A'].x) == 1
    assert len(data['B'].y) == 1


@withPackage('networkx')
def test_from_hetero_networkx_attrs_selection():
    import networkx as nx

    G = nx.DiGraph()
    G.add_node(0, type="A", u=0, v=0, a=0)
    G.add_node(1, type="A", u=1, v=1, a=1)
    G.add_node(2, type="B", u=2, v=2, b=2)
    G.add_node(3, type="B", u=3, v=3, b=3)

    G.add_edge(0, 1, u=1, a=1)
    G.add_edge(2, 3, u=2, b=2)

    data = from_hetero_networkx(G, node_type_attribute="type",
                                group_node_attrs=["u",
                                                  "v"], group_edge_attrs=["u"])

    assert data['A']['x'].tolist() == [[0, 0], [1, 1]]
    assert data['B']['x'].tolist() == [[2, 2], [3, 3]]
    assert data[("A", "to", "A")]['edge_attr'].tolist() == [[1]]
    assert data[("B", "to", "B")]['edge_attr'].tolist() == [[2]]


@withPackage('networkx')
def test_from_hetero_networkx_missing_attrs():
    import networkx as nx

    G = nx.DiGraph()
    G.add_node(0, type="A", u=0, v=0)
    G.add_node(1, type="B", u=1)

    with pytest.raises(KeyError,
                       match=r'Missing required attribute in group: .*') as _:
        from_hetero_networkx(G, node_type_attribute="type",
                             group_node_attrs=["u", "v"])
