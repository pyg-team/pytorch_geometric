import pytest
import torch

from torch_geometric.nn import GAT, GCN, to_captum

GCN = GCN(3, 16, 2, 7, dropout=0.5)
GAT = GAT(3, 16, 2, 7, heads=2, concat=False)

mask_type = ['edge', 'node_and_edge', 'node']


@pytest.mark.parametrize('mask_type', mask_type)
@pytest.mark.parametrize('model', [GCN, GAT])
@pytest.mark.parametrize('node_idx', [None, 1])
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
        out = captum_model(mask.unsqueeze(0), x, edge_index)
    elif mask_type == 'node_and_edge':
        node_mask = x * 0.0
        edge_mask = torch.ones(edge_index.shape[1], dtype=torch.float) * 0.5
        out = captum_model(node_mask.unsqueeze(0), edge_mask.unsqueeze(0),
                           edge_index)

    if node_idx is not None:
        assert out.shape == (1, 7)
        assert torch.any(out != pre_out[[node_idx]])
    else:
        assert out.shape == (8, 7)
        assert torch.any(out != pre_out)
