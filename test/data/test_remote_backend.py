import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.data.remote_backend_utils import num_nodes
from torch_geometric.testing.feature_store import MyFeatureStore
from torch_geometric.testing.graph_store import MyGraphStore


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


@pytest.mark.parametrize('FeatureStore', [MyFeatureStore, HeteroData])
@pytest.mark.parametrize('GraphStore', [MyGraphStore, HeteroData])
def test_num_nodes(FeatureStore, GraphStore):
    feature_store = FeatureStore()
    graph_store = GraphStore()

    # Graph:
    # X (100 nodes), Y (50 nodes), Z (20 nodes)
    # X -> Y, Y -> Z edges

    # Set up node features. We explicitly do not do this for 'z' since we
    # should be able to infer the size of 'z' from the yz edge:
    x = torch.arange(100)
    y = torch.arange(50)

    # Set up edges. We test both with and without the inclusion of 'yz', where
    # failing to include this edge should throw an error:
    xy = get_edge_index(100, 50, 20)
    yz = get_edge_index(50, 20, 20)

    feature_store.put_tensor(x, group_name='x', attr_name='x', index=None)
    feature_store.put_tensor(y, group_name='y', attr_name='x', index=None)

    graph_store.put_edge_index(xy, edge_type=('x', 'to', 'y'), layout='coo',
                               size=(100, 50))

    assert num_nodes(feature_store, graph_store, 'x') == 100
    assert num_nodes(feature_store, graph_store, 'y') == 50

    with pytest.raises(ValueError, match="Unable to accurately infer"):
        _ = num_nodes(feature_store, graph_store, 'z')

    graph_store.put_edge_index(yz, edge_type=('y', 'to', 'z'), layout='coo',
                               size=(50, 20))
    assert num_nodes(feature_store, graph_store, 'z') == 20
