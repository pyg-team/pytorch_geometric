import torch

from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv
from torch_geometric.nn.models.explainer_hetero import (
    clear_masks_hetero,
    set_masks_hetero,
)


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


class HetroNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_dummy = GCNConv(32, 64)
        self.conv1 = HeteroConv({
            ('paper', 'to', 'paper'):
            GCNConv(32, 64),
            ('author', 'to', 'paper'):
            SAGEConv((64, 32), 64),
        })

        self.conv2 = HeteroConv({
            ('paper', 'to', 'paper'):
            GCNConv(64, 64),
            ('author', 'to', 'paper'):
            SAGEConv((64, 64), 64),
        })

    def forward():
        pass


def test_set_clear_mask():
    data = HeteroData()
    data['paper'].x = torch.randn(50, 32)
    data['author'].x = torch.randn(30, 64)
    data['paper', 'paper'].edge_index = get_edge_index(50, 50, 200)
    data['author', 'paper'].edge_index = get_edge_index(30, 50, 100)
    data['paper', 'paper'].edge_weight = torch.rand(200)

    edge_mask_dict = {
        ('paper', 'to', 'paper'): torch.ones(200),
        ('author', 'to', 'paper'): torch.ones(100)
    }
    conv = HetroNet()
    set_masks_hetero(conv, edge_mask_dict, data.edge_index_dict)
    # Check that mask is set only for MessagePassing
    # layers in `HeteroConv` module
    assert conv.conv_dummy._edge_mask is None
    assert not conv.conv_dummy.explain
    for edge_type in data.edge_types:
        str_edge_type = '__'.join(edge_type)
        assert torch.allclose(conv.conv1.convs[str_edge_type]._edge_mask,
                              edge_mask_dict[edge_type])
        assert conv.conv1.convs[str_edge_type].explain

    clear_masks_hetero(conv)
    assert conv.conv_dummy._edge_mask is None
    assert not conv.conv_dummy.explain
    for edge_type in data.edge_types:
        str_edge_type = '__'.join(edge_type)
        assert conv.conv1.convs[str_edge_type]._edge_mask is None
        assert not conv.conv1.convs[str_edge_type].explain
