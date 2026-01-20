import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.nn.models import AEROGNN
from torch_geometric.testing import withDevice

out_dims = [None, 8]
dropouts = [0.0, 0.5]
iterations_list = [1, 5, 10]
heads_list = [1, 2]
lambd_list = [0.5, 1.0, 2.0]
num_layers_list = [1, 2]


@pytest.mark.parametrize('out_dim', out_dims)
@pytest.mark.parametrize('dropout', dropouts)
@pytest.mark.parametrize('iterations', iterations_list)
@pytest.mark.parametrize('heads', heads_list)
@pytest.mark.parametrize('lambd', lambd_list)
@pytest.mark.parametrize('num_layers', num_layers_list)
def test_aero_gnn(out_dim, dropout, iterations, heads, lambd, num_layers):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    out_channels = 16 if out_dim is None else out_dim

    model = AEROGNN(
        in_channels=8,
        hidden_channels=16,
        num_layers=num_layers,
        out_channels=out_dim,
        iterations=iterations,
        heads=heads,
        lambd=lambd,
        dropout=dropout,
    )
    expected_str = (f'AEROGNN(8, 16, num_layers={num_layers}, '
                    f'out_channels={out_channels}, iterations={iterations}, '
                    f'heads={heads})')
    assert str(model) == expected_str
    assert model(x, edge_index).size() == (3, out_channels)


@pytest.mark.parametrize('add_dropout', [False, True])
def test_aero_gnn_add_dropout(add_dropout):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = AEROGNN(
        in_channels=8,
        hidden_channels=16,
        num_layers=1,
        out_channels=16,
        iterations=5,
        heads=1,
        add_dropout=add_dropout,
    )
    assert model(x, edge_index).size() == (3, 16)


def test_aero_gnn_reset_parameters():
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = AEROGNN(
        in_channels=8,
        hidden_channels=16,
        num_layers=2,
        out_channels=16,
        iterations=10,
        heads=2,
    )

    # Get output before reset
    out1 = model(x, edge_index)

    # Reset parameters
    model.reset_parameters()

    # Get output after reset (should be different due to random initialization)
    out2 = model(x, edge_index)

    # Outputs should be different (very unlikely to be the same after reset)
    assert not torch.allclose(out1, out2, atol=1e-6)


def test_aero_gnn_single_layer():
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = AEROGNN(
        in_channels=8,
        hidden_channels=16,
        num_layers=1,
        out_channels=16,
        iterations=5,
    )
    assert model(x, edge_index).size() == (3, 16)


def test_aero_gnn_multiple_iterations():
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = AEROGNN(
        in_channels=8,
        hidden_channels=16,
        num_layers=1,
        out_channels=16,
        iterations=20,
    )
    assert model(x, edge_index).size() == (3, 16)


def test_aero_gnn_multi_head():
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = AEROGNN(
        in_channels=8,
        hidden_channels=16,
        num_layers=1,
        out_channels=16,
        iterations=5,
        heads=4,
    )
    assert model(x, edge_index).size() == (3, 16)


def test_aero_gnn_with_sparse_tensor():
    from torch_geometric.typing import SparseTensor

    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(3, 3))

    model = AEROGNN(
        in_channels=8,
        hidden_channels=16,
        num_layers=1,
        out_channels=16,
        iterations=5,
    )
    assert model(x, adj).size() == (3, 16)


@withDevice
def test_aero_gnn_device(device):
    x = torch.randn(3, 8, device=device)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device)

    model = AEROGNN(
        in_channels=8,
        hidden_channels=16,
        num_layers=1,
        out_channels=16,
        iterations=5,
    ).to(device)

    assert model(x, edge_index).size() == (3, 16)
    assert model(x, edge_index).device == device


def test_aero_gnn_gradient():
    x = torch.randn(3, 8, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = AEROGNN(
        in_channels=8,
        hidden_channels=16,
        num_layers=1,
        out_channels=16,
        iterations=5,
    )

    out = model(x, edge_index)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    # Check that model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_aero_gnn_with_data():
    data = Data(
        x=torch.randn(3, 8),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
    )

    model = AEROGNN(
        in_channels=8,
        hidden_channels=16,
        num_layers=1,
        out_channels=16,
        iterations=5,
    )

    out = model(data.x, data.edge_index)
    assert out.size() == (3, 16)


def test_aero_gnn_num_nodes():
    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])

    model = AEROGNN(
        in_channels=8,
        hidden_channels=16,
        num_layers=1,
        out_channels=16,
        iterations=5,
    )

    # Test with explicit num_nodes
    out1 = model(x, edge_index, num_nodes=5)
    # Test without num_nodes (should infer)
    out2 = model(x, edge_index)

    assert out1.size() == (5, 16)
    assert out2.size() == (5, 16)
    assert torch.allclose(out1, out2, atol=1e-6)
