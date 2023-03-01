import torch

from torch_geometric.explain.algorithm.utils import (
    clear_masks,
    set_hetero_masks,
)
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv, to_hetero


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


def test_set_clear_mask(hetero_data):
    edge_mask_dict = {
        ('paper', 'to', 'paper'): torch.ones(200),
        ('author', 'to', 'paper'): torch.ones(100),
        ('paper', 'to', 'author'): torch.ones(100),
    }

    model = HeteroModel()

    set_hetero_masks(model, edge_mask_dict, hetero_data.edge_index_dict)
    for edge_type in hetero_data.edge_types:
        # Check that masks are correctly set:
        str_edge_type = '__'.join(edge_type)
        assert torch.allclose(model.conv1.convs[str_edge_type]._edge_mask,
                              edge_mask_dict[edge_type])
        assert model.conv1.convs[str_edge_type].explain

    clear_masks(model)
    for edge_type in hetero_data.edge_types:
        str_edge_type = '__'.join(edge_type)
        assert model.conv1.convs[str_edge_type]._edge_mask is None
        assert not model.conv1.convs[str_edge_type].explain

    model = to_hetero(GraphSAGE(), hetero_data.metadata(), debug=False)

    set_hetero_masks(model, edge_mask_dict, hetero_data.edge_index_dict)
    for edge_type in hetero_data.edge_types:
        # Check that masks are correctly set:
        str_edge_type = '__'.join(edge_type)
        assert torch.allclose(model.conv1[str_edge_type]._edge_mask,
                              edge_mask_dict[edge_type])
        assert model.conv1[str_edge_type].explain

    clear_masks(model)
    for edge_type in hetero_data.edge_types:
        str_edge_type = '__'.join(edge_type)
        assert model.conv1[str_edge_type]._edge_mask is None
        assert not model.conv1[str_edge_type].explain
