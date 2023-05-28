import torch

from torch_geometric.distributed import LocalGraphStore
from torch_geometric.testing import get_random_edge_index


def test_local_graph_store():
    graph_store = LocalGraphStore()

    edge_index = get_random_edge_index(100, 100, 300)
    edge_id = torch.tensor([1, 2, 3, 5, 8, 4], dtype=torch.int64)

    graph_store.put_edge_index(edge_index, edge_type=None, layout='coo',
                               size=(100, 100))

    graph_store.put_edge_id(edge_id, edge_type=None, layout='coo',
                            size=(100, 100))

    assert len(graph_store.get_all_edge_attrs()) == 1
    edge_attr = graph_store.get_all_edge_attrs()[0]
    assert torch.equal(graph_store.get_edge_index(edge_attr), edge_index)
    assert torch.equal(graph_store.get_edge_id(edge_attr), edge_id)

    graph_store.remove_edge_index(edge_attr)
    graph_store.remove_edge_id(edge_attr)
    assert len(graph_store.get_all_edge_attrs()) == 0
