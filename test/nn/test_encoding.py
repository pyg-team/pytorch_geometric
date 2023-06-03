import torch

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import (
    NodeEncoding,
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
def test_node_encoding(get_dataset, device):
    dataset = get_dataset(name='Cora')
    data = dataset[0]
    loader = NeighborLoader(
        data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=data.train_mask,
    )
    sampled_data = next(iter(loader))

    encoder = NodeEncoding().to(device)
    output = encoder(sampled_data)
    assert str(encoder) == 'NodeEncoding'
    assert output.size() == sampled_data.x.size()
