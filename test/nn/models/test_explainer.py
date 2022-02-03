import pytest
import torch
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv, to_captum


# TODO: Use same as in GNNExplainer Test
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 16, heads=2, concat=True)
        self.conv2 = GATConv(2 * 16, 7, heads=2, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


mask_type = ['edge', 'node_and_edge', 'node']


@pytest.mark.parametrize("mask_type", mask_type)
@pytest.mark.parametrize("model", [GCN(), GAT()])
@pytest.mark.parametrize("node_idx", [None, 1])
def test_to_captum(model, mask_type, node_idx):
    captum_model = to_captum(model, mask_type=mask_type, node_idx=node_idx)

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    pre_out = model(x, edge_index)

    captum_model = to_captum(model, mask_type=mask_type, node_idx=node_idx)

    if mask_type == 'node':
        mask = x * 0.0
        out = captum_model(mask.unsqueeze(0), edge_index)
    elif mask_type == 'edge':
        mask = torch.ones(edge_index.shape[1], dtype=torch.float) * 0.5
        out = captum_model(mask, x, edge_index)
    elif mask_type == 'node_and_edge':
        node_mask = x * 0.0
        edge_mask = torch.ones(edge_index.shape[1], dtype=torch.float) * 0.5
        out = captum_model(node_mask.unsqueeze(0), edge_mask, edge_index)

    if node_idx is not None:
        assert out.shape == (1, 7)
        assert torch.any(out != pre_out[[node_idx]])
    else:
        assert out.shape == (8, 7)
        assert torch.any(out != pre_out)
