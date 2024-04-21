import pytest
import torch

from torch_geometric.nn import CMPNN


@pytest.mark.parametrize('communicator',
                         ['additive', 'inner_product', 'gru', 'mlp'])
def test_cmpnn(communicator):
    model = CMPNN(8, 16, 32, edge_dim=3, num_layers=2,
                  communicator=communicator)
    assert str(model) == ('CMPNN(in_channels=8, hidden_channels=16, '
                          'out_channels=32, edge_dim=3, num_layers=2, '
                          f'communicator={communicator})')

    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = torch.randn(edge_index.size(1), 3)
    batch = torch.tensor([0, 0, 0, 0])

    out = model(x, edge_index, edge_attr, batch)
    assert out.size() == (4, 32)
