import torch

from torch_geometric.distributed import LocalGraphStore
from torch_geometric.testing import get_random_edge_index


def test_local_graph_store():
    graph_store = LocalGraphStore()

    edge_index = get_random_edge_index(100, 100, 300)
    edge_id = torch.tensor([1, 2, 3, 5, 8, 4])

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


def test_homogeneous_graph_store():
    edge_id = torch.randperm(300)
    edge_index = get_random_edge_index(100, 100, 300)

    graph_store = LocalGraphStore.from_data(edge_id, edge_index, num_nodes=100)

    assert len(graph_store.get_all_edge_attrs()) == 1
    edge_attr = graph_store.get_all_edge_attrs()[0]
    assert edge_attr.edge_type is None
    assert edge_attr.layout.value == 'coo'
    assert not edge_attr.is_sorted
    assert edge_attr.size == (100, 100)

    assert torch.equal(
        graph_store.get_edge_id(edge_type=None, layout='coo'),
        edge_id,
    )
    assert torch.equal(
        graph_store.get_edge_index(edge_type=None, layout='coo'),
        edge_index,
    )


def test_heterogeneous_graph_store():
    edge_type = ('paper', 'to', 'paper')
    edge_id_dict = {edge_type: torch.randperm(300)}
    edge_index_dict = {edge_type: get_random_edge_index(100, 100, 300)}

    graph_store = LocalGraphStore.from_hetero_data(
        edge_id_dict, edge_index_dict, num_nodes_dict={'paper': 100})

    assert len(graph_store.get_all_edge_attrs()) == 1
    edge_attr = graph_store.get_all_edge_attrs()[0]
    assert edge_attr.edge_type == edge_type
    assert edge_attr.layout.value == 'coo'
    assert not edge_attr.is_sorted
    assert edge_attr.size == (100, 100)

    assert torch.equal(
        graph_store.get_edge_id(edge_type, layout='coo'),
        edge_id_dict[edge_type],
    )
    assert torch.equal(
        graph_store.get_edge_index(edge_type, layout='coo'),
        edge_index_dict[edge_type],
    )
