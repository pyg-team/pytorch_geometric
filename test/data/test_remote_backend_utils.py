import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.data.remote_backend_utils import num_nodes, size
from torch_geometric.testing import MyFeatureStore, MyGraphStore


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


@pytest.mark.parametrize('FeatureStore', [MyFeatureStore, HeteroData])
@pytest.mark.parametrize('GraphStore', [MyGraphStore, HeteroData])
def test_num_nodes_size(FeatureStore, GraphStore):
    feature_store = FeatureStore()
    graph_store = GraphStore()

    # Infer num nodes from features:
    x = torch.arange(100)
    feature_store.put_tensor(x, group_name='x', attr_name='x', index=None)
    assert num_nodes(feature_store, graph_store, 'x') == 100

    # Infer num nodes and size from edges:
    xy = get_edge_index(100, 50, 20)
    graph_store.put_edge_index(xy, edge_type=('x', 'to', 'y'), layout='coo',
                               size=(100, 50))
    assert num_nodes(feature_store, graph_store, 'y') == 50
    assert size(feature_store, graph_store, ('x', 'to', 'y')) == (100, 50)

    # Throw an error if we cannot infer for an unknown node type:
    with pytest.raises(ValueError, match="Unable to accurately infer"):
        _ = num_nodes(feature_store, graph_store, 'z')
