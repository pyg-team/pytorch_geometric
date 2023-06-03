import torch

from torch_geometric.nn import (
    LinkEncoding,
    PositionalEncoding,
    TemporalEncoding,
)
from torch_geometric.testing import withCUDA


@withCUDA
def test_positional_encoding(device):
    encoder = PositionalEncoding(64).to(device)
    assert str(encoder) == 'PositionalEncoding(64)'

    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    assert encoder(x).size() == (3, 64)


@withCUDA
def test_temporal_encoding(device):
    encoder = TemporalEncoding(64).to(device)
    assert str(encoder) == 'TemporalEncoding(64)'

    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    assert encoder(x).size() == (3, 64)


@withCUDA
def test_link_encoding(device):
    num_nodes = 3
    num_edges = 6
    num_edge_features = 10
    edge_attr = torch.rand((num_edges, num_edge_features), device=device)
    edge_index = torch.randint(low=0, high=num_nodes, size=(2, num_edges),
                               device=device)
    edge_time = torch.rand(num_edges, device=device)

    K = 3
    hidden_channels = 7
    out_channels = 11
    time_channels = 13

    encoder = LinkEncoding(
        K=K,
        in_channels=num_edge_features,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        time_channels=time_channels,
    ).to(device)
    assert str(encoder) == (f'LinkEncoding(K={K}, '
                            f'in_channels={num_edge_features}, '
                            f'hidden_channels={hidden_channels}, '
                            f'out_channels={out_channels}, '
                            f'time_channels={time_channels})')

    out = encoder(edge_attr, edge_time, edge_index)
    assert out.size() == (num_nodes, out_channels)
