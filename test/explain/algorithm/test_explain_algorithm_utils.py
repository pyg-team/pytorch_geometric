import torch

from torch_geometric.data import HeteroData
from torch_geometric.explain.algorithm.utils import (
    clear_masks,
    set_hetero_masks,
)
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv, to_hetero


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def get_hetero_data():
    data = HeteroData()
    data['paper'].x = torch.randn(8, 16)
    data['author'].x = torch.randn(10, 8)
    data['paper', 'paper'].edge_index = get_edge_index(8, 8, 10)
    data['author', 'paper'].edge_index = get_edge_index(10, 8, 10)
    data['paper', 'author'].edge_index = get_edge_index(8, 10, 10)
    return data


class HeteroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = HeteroConv({
            ('paper', 'to', 'paper'):
            GCNConv(-1, 32),
            ('author', 'to', 'paper'):
            SAGEConv((-1, -1), 32),
            ('paper', 'to', 'author'):
            SAGEConv((-1, -1), 32),
        })

        self.conv2 = HeteroConv({
            ('paper', 'to', 'paper'):
            GCNConv(-1, 32),
            ('author', 'to', 'paper'):
            SAGEConv((-1, -1), 32),
            ('paper', 'to', 'author'):
            SAGEConv((-1, -1), 32),
        })


class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 32)
        self.conv2 = SAGEConv((-1, -1), 32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


def test_set_clear_mask():
    data = get_hetero_data()
    edge_mask_dict = {
        ('paper', 'to', 'paper'): torch.ones(200),
        ('author', 'to', 'paper'): torch.ones(100),
        ('paper', 'to', 'author'): torch.ones(100),
    }

    model = HeteroModel()

    set_hetero_masks(model, edge_mask_dict, data.edge_index_dict)
    for edge_type in data.edge_types:  # Check that masks are correctly set:
        str_edge_type = '__'.join(edge_type)
        assert torch.allclose(model.conv1.convs[str_edge_type]._edge_mask,
                              edge_mask_dict[edge_type])
        assert model.conv1.convs[str_edge_type].explain

    clear_masks(model)
    for edge_type in data.edge_types:
        str_edge_type = '__'.join(edge_type)
        assert model.conv1.convs[str_edge_type]._edge_mask is None
        assert not model.conv1.convs[str_edge_type].explain

    model = GraphSAGE()
    model = to_hetero(GraphSAGE(), data.metadata(), debug=False)

    set_hetero_masks(model, edge_mask_dict, data.edge_index_dict)
    for edge_type in data.edge_types:  # Check that masks are correctly set:
        str_edge_type = '__'.join(edge_type)
        assert torch.allclose(model.conv1[str_edge_type]._edge_mask,
                              edge_mask_dict[edge_type])
        assert model.conv1[str_edge_type].explain

    clear_masks(model)
    for edge_type in data.edge_types:
        str_edge_type = '__'.join(edge_type)
        assert model.conv1[str_edge_type]._edge_mask is None
        assert not model.conv1[str_edge_type].explain
