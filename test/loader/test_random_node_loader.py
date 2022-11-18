import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import RandomNodeLoader


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def test_random_node_loader():
    data = Data()
    data.x = torch.randn(100, 128)
    data.node_id = torch.arange(100)
    data.edge_index = get_edge_index(100, 100, 500)
    data.edge_attr = torch.randn(500, 32)

    loader = RandomNodeLoader(data, num_parts=4, shuffle=True)
    assert len(loader) == 4

    for batch in loader:
        assert len(batch) == 4
        assert batch.node_id.min() >= 0
        assert batch.node_id.max() < 100
        assert batch.edge_index.size(1) == batch.edge_attr.size(0)
        assert torch.allclose(batch.x, data.x[batch.node_id])
        batch.validate()


def test_heterogeneous_random_node_loader():
    data = HeteroData()
    data['paper'].x = torch.randn(100, 128)
    data['paper'].node_id = torch.arange(100)
    data['author'].x = torch.randn(200, 128)
    data['author'].node_id = torch.arange(200)
    data['paper', 'author'].edge_index = get_edge_index(100, 200, 500)
    data['paper', 'author'].edge_attr = torch.randn(500, 32)
    data['author', 'paper'].edge_index = get_edge_index(200, 100, 400)
    data['author', 'paper'].edge_attr = torch.randn(400, 32)
    data['paper', 'paper'].edge_index = get_edge_index(100, 100, 600)
    data['paper', 'paper'].edge_attr = torch.randn(600, 32)

    loader = RandomNodeLoader(data, num_parts=4, shuffle=True)
    assert len(loader) == 4

    for batch in loader:
        assert len(batch) == 4
        assert batch.node_types == data.node_types
        assert batch.edge_types == data.edge_types
        batch.validate()
