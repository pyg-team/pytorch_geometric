import torch

from torch_geometric.nn.models import PMLP


def test_pmlp():
    x = torch.randn(4, 16)

    pmlp = PMLP(in_channels=16, hidden_channels=32, out_channels=2,
                num_layers=4, aggr="sum")
    pmlp.training = True
    out = pmlp(x, edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
    assert out.size() == (4, 2)

    pmlp.training = False
    out = pmlp(x, edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
    assert out.size() == (4, 2)
