import pytest
import scipy.sparse
import torch
from torch_geometric.data import Data
from torch_geometric.testing import withPackage
from torch_geometric.utils import (
    from_networkit,
    from_networkx,
    from_scipy_sparse_matrix,
    from_trimesh,
    from_cugraph,
    subgraph,
    to_networkit,
    to_networkx,
    to_scipy_sparse_matrix,
    to_trimesh,
    to_cugraph
)
from torch.utils.dlpack import from_dlpack, to_dlpack

def test_to_scipy_sparse_matrix():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])

    adj = to_scipy_sparse_matrix(edge_index)
    assert isinstance(adj, scipy.sparse.coo_matrix) is True
    assert adj.shape == (2, 2)
    assert adj.row.tolist() == edge_index[0].tolist()
    assert adj.col.tolist() == edge_index[1].tolist()
    assert adj.data.tolist() == [1, 1, 1]

    edge_attr = torch.Tensor([1, 2, 3])
    adj = to_scipy_sparse_matrix(edge_index, edge_attr)
    assert isinstance(adj, scipy.sparse.coo_matrix) is True
    assert adj.shape == (2, 2)
    assert adj.row.tolist() == edge_index[0].tolist()
    assert adj.col.tolist() == edge_index[1].tolist()
    assert adj.data.tolist() == edge_attr.tolist()


def test_from_scipy_sparse_matrix():
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    adj = to_scipy_sparse_matrix(edge_index)

    out = from_scipy_sparse_matrix(adj)
    assert out[0].tolist() == edge_index.tolist()
    assert out[1].tolist() == [1, 1, 1]


@withPackage('networkx')
def test_to_networkx():
    import networkx as nx

    x = torch.Tensor([[1, 2], [3, 4]])
    pos = torch.Tensor([[0, 0], [1, 1]])
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_attr = torch.Tensor([1, 2, 3])
    data = Data(x=x, pos=pos, edge_index=edge_index, weight=edge_attr)

    for remove_self_loops in [True, False]:
        G = to_networkx(data, node_attrs=['x', 'pos'], edge_attrs=['weight'],
                        remove_self_loops=remove_self_loops)

        assert G.nodes[0]['x'] == [1, 2]
        assert G.nodes[1]['x'] == [3, 4]
        assert G.nodes[0]['pos'] == [0, 0]
        assert G.nodes[1]['pos'] == [1, 1]

        if remove_self_loops:
            assert nx.to_numpy_array(G).tolist() == [[0, 1], [2, 0]]
        else:
            assert nx.to_numpy_array(G).tolist() == [[3, 1], [2, 0]]


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

    x = torch.Tensor([[1, 2], [3, 4]])
    pos = torch.Tensor([[0, 0], [1, 1]])
    edge_index = torch.tensor([[0, 1, 0], [1, 0, 0]])
    edge_attr = torch.Tensor([1, 2, 3])
    data = Data(x=x, pos=pos, edge_index=edge_index, weight=edge_attr)

    for remove_self_loops in [True, False]:
        G = to_networkx(data, node_attrs=['x', 'pos'], edge_attrs=['weight'],
                        remove_self_loops=remove_self_loops,
                        to_undirected=True)

        assert G.nodes[0]['x'] == [1, 2]
        assert G.nodes[1]['x'] == [3, 4]
        assert G.nodes[0]['pos'] == [0, 0]
        assert G.nodes[1]['pos'] == [1, 1]

        if remove_self_loops:
            assert nx.to_numpy_array(G).tolist() == [[0, 1], [1, 0]]
        else:
            assert nx.to_numpy_array(G).tolist() == [[3, 1], [1, 0]]

    G = to_networkx(data, edge_attrs=['weight'], to_undirected=False)
    assert nx.to_numpy_array(G).tolist() == [[3, 1], [2, 0]]

    G = to_networkx(data, edge_attrs=['weight'], to_undirected="upper")
    assert nx.to_numpy_array(G).tolist() == [[3, 1], [1, 0]]

    G = to_networkx(data, edge_attrs=['weight'], to_undirected="lower")
    assert nx.to_numpy_array(G).tolist() == [[3, 2], [2, 0]]


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


@withPackage('networkit')
def test_to_networkit_vice_versa():
    edge_index = torch.tensor([[0, 1], [1, 0]])

    g = to_networkit(edge_index, directed=False)
    assert not g.isDirected()
    assert not g.isWeighted()

    edge_index, edge_weight = from_networkit(g)
    assert edge_index.tolist() == [[0, 1], [1, 0]]
    assert edge_weight is None


@withPackage('trimesh')
def test_trimesh():
    pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                       dtype=torch.float)
    face = torch.tensor([[0, 1, 2], [1, 2, 3]]).t()

    data = Data(pos=pos, face=face)
    mesh = to_trimesh(data)
    data = from_trimesh(mesh)

    assert pos.tolist() == data.pos.tolist()
    assert face.tolist() == data.face.tolist()


@withPackage("networkit")
@pytest.mark.parametrize('directed', [True, False])
@pytest.mark.parametrize('num_nodes', [None, 3])
@pytest.mark.parametrize('edge_weight', [None, torch.tensor([1, 2, 3], dtype=torch.float)])
def test_to_networkit(directed, edge_weight, num_nodes):
    from networkit import Graph
    print(directed, edge_weight)
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]], dtype=torch.long)
    g = to_networkit(edge_index, edge_weight, num_nodes, directed)

    assert isinstance(g, Graph)
    assert g.isDirected() == directed
    assert g.numberOfNodes() == 3

    # test case for graph with no edge weights
    if edge_weight is None:
        edge_weight = torch.tensor([1, 1, 1], dtype=torch.float)  # all set to one in networkit

    # checking number of edges and edges weights
    assert g.weight(0, 1) == edge_weight[0].item()
    assert g.weight(1, 2) == edge_weight[2].item()

    if directed:
        assert g.numberOfEdges() == 3
        assert g.weight(1, 0) == edge_weight[1].item()
    else:
        assert g.numberOfEdges() == 2


@pytest.mark.parametrize('directed', [True, False])
@pytest.mark.parametrize('weighted', [True, False])
@withPackage("networkit")
def test_from_networkit(directed, weighted):
    from networkit import Graph

    g = Graph(3, weighted=weighted, directed=directed)
    g.addEdge(0, 1)
    g.addEdge(1, 2)
    if directed:
        g.addEdge(1, 0)

    if weighted:
        for i, (u, v) in enumerate(g.iterEdges()):
            g.setWeight(u, v, i + 1)

    assert g.isWeighted() == weighted
    assert g.isDirected() == directed

    edge_index, edge_weight = from_networkit(g)
    expected_edge_index = [[0, 1, 1], [1, 2, 0]] if directed else [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert edge_index.tolist() == expected_edge_index
    if not weighted:
        assert edge_weight is None
    else:
        expected_edge_weight = [1, 2, 3] if directed else [1, 1, 2, 2]
        assert edge_weight.tolist() == expected_edge_weight


@withPackage("trimesh")
def test_to_trimesh():
    import trimesh

    pos = torch.tensor([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
    face = torch.tensor([[0, 1, 2], [2, 1, 3]]).t()
    data = Data(pos=pos, face=face)

    # Convert the data object to a trimesh object
    trimesh_object = to_trimesh(data)

    # Check if the trimesh object is created correctly
    assert isinstance(trimesh_object, trimesh.Trimesh)
    assert trimesh_object.vertices.shape == (4, 3)
    assert trimesh_object.faces.shape == (2, 3)
    assert trimesh_object.vertices.tolist() == data.pos.tolist()
    assert trimesh_object.faces.tolist() == data.face.t().contiguous().tolist()


@withPackage("trimesh")
def test_from_trimesh():

    import trimesh
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    faces = [[0, 1, 2]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    data = from_trimesh(mesh)

    assert data.pos.shape == (3, 3)
    assert data.face.shape == (3, 1)
    assert data.pos.tolist() == vertices
    assert data.face.t().contiguous().tolist() == faces


@withPackage("cugraph")
@pytest.mark.parametrize("edge_weight", [None, [1, 2, 3, 4]])
@pytest.mark.parametrize("relabel_nodes", [True, False])
@pytest.mark.parametrize("directed", [True, False])
def test_to_cugraph(edge_weight, directed, relabel_nodes):
    import cugraph

    edge_index = torch.tensor([[0, 1], [1, 2]])

    if directed:
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    if edge_weight is not None:
        edge_weight = torch.tensor(edge_weight)
        if not directed:
            edge_weight = edge_weight[:2]

    graph = to_cugraph(edge_index, edge_weight, relabel_nodes, directed)
    assert isinstance(graph, cugraph.Graph)
    assert graph.number_of_nodes() == 3

    edge_list = graph.view_edge_list()
    assert edge_list is not None

    cu_edge_index = edge_list[['src', 'dst']].to_pandas().values.tolist()
    assert edge_index.tolist() == torch.tensor(sorted(cu_edge_index)).t().contiguous().tolist()

    if edge_weight is not None:
        edge_list = edge_list.sort_values(by=['src', 'dst'])
        cu_edge_weight = edge_list['weights'].to_pandas().values

        assert edge_weight.tolist() == cu_edge_weight.tolist()


@withPackage("cugraph")
@pytest.mark.parametrize("edge_weight", [None, [1, 2, 7, 5]])
@pytest.mark.parametrize("directed", [True, False])
@pytest.mark.parametrize("relabel_nodes", [True, False])
def test_from_cugraph(edge_weight, directed, relabel_nodes):
    import cugraph
    import cudf

    edge_index = torch.tensor([[0, 1], [1, 2]])
    if directed:
        edge_index = torch.tensor([[1, 1, 2, 0], [0, 2, 0, 0]])

    if edge_weight is not None:
        edge_weight = torch.tensor(edge_weight)
        if not directed:
            edge_weight = edge_weight[:2]

    # creating cugraph
    G = cugraph.Graph(directed=directed)
    df = cudf.from_dlpack(to_dlpack(edge_index.t()))
    if edge_weight is not None:
        df['2'] = cudf.from_dlpack(to_dlpack(edge_weight))

    G.from_cudf_edgelist(
        df, source=0, destination=1,
        edge_attr='2' if edge_weight is not None else None,
        renumber=relabel_nodes)

    cu_edge_index, cu_edge_weight = from_cugraph(G)
    # NOTE: If the graph is of type Graph() then the displayed undirected edges are the same as displayed by networkx Graph(),
    # but the direction could be different i.e. an edge displayed by cugraph as (src, dst) could be displayed as (dst, src) by networkx.
    # cugraph.Graph stores symmetrized edgelist internally.
    assert sorted(cu_edge_index.t().tolist()) == [[0, 0], [1, 0], [1, 2], [2, 0]] if directed else [[1, 0], [2, 1]]

    if edge_weight is None:
        assert cu_edge_weight is None
    else:
        cu_edge_weight.tolist() == [1, 2, 7, 5] if directed else [2, 1]
