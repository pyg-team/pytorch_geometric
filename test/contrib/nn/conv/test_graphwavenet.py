import pytest
import torch

from torch_geometric.contrib.nn.conv.graphwavenet import GraphWaveNet


@pytest.mark.parametrize('num_nodes', [10, 20])
@pytest.mark.parametrize('in_channels', [32, 64])
@pytest.mark.parametrize('in_timesteps', [7, 14])
@pytest.mark.parametrize('out_channels', [8, 16])
@pytest.mark.parametrize('out_timesteps', [1, 10])
@pytest.mark.parametrize('dilations', [[1, 2, 3], [1, 2, 1, 2, 1, 2, 1, 2]])
@pytest.mark.parametrize('is_batched', [True, False])
def test_graphwavenet(num_nodes, in_channels, in_timesteps, out_channels,
                      out_timesteps, dilations, is_batched):

    model = GraphWaveNet(num_nodes, in_channels, out_channels, out_timesteps,
                         dilations=dilations)

    if is_batched:
        in_shape = (64, in_timesteps, num_nodes, in_channels)
        out_shape = (64, out_timesteps, num_nodes, out_channels)
    else:
        in_shape = (in_timesteps, num_nodes, in_channels)
        out_shape = (out_timesteps, num_nodes, out_channels)

    x = torch.randn(in_shape)
    edge_index = torch.tensor([[i for i in range(num_nodes - 1)],
                               [i + 1 for i in range(num_nodes - 1)]])
    edge_weight = torch.FloatTensor([i for i in range(len(edge_index[0]))])

    out = model(x, edge_index, edge_weight)

    assert out.shape == out_shape
