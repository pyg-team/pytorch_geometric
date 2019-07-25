import torch
from torch_geometric.nn import ChebConv


def test_cheb_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3],
                               [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))

    conv = ChebConv(in_channels, out_channels, K=3)
    assert conv.__repr__() == 'ChebConv(16, 32, K=3)'

    convs = (
        conv(x, edge_index),
        conv(x, edge_index, edge_weight),
        conv(x, edge_index, edge_weight, lambda_max=5.0),
        conv(x, edge_index, edge_weight, lambda_max=torch.tensor(7.0)),
        conv(x, edge_index, edge_weight, lambda_max=torch.tensor([9.0])),
    )
    for c in convs:
        assert c.size() == (num_nodes, out_channels)

    num_batches = 3
    batch_edge_index = edge_index.repeat(1, num_batches)
    batch_edge_weight = edge_weight.repeat(num_batches)
    batch_x = torch.randn((num_nodes * num_batches, in_channels))
    batch_lambda_max = torch.tensor([3.0 + n for n in range(num_batches)])
    batch = torch.tensor(range(num_batches))
    batch = batch.view(-1, 1).repeat(1, num_nodes).view(-1)
    conv_result = conv(x=batch_x,
                       edge_index=batch_edge_index,
                       edge_weight=batch_edge_weight,
                       batch=batch,
                       lambda_max=batch_lambda_max)
    assert conv_result.size() == (num_nodes * num_batches, out_channels)
