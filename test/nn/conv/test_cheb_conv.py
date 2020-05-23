import torch
from torch_geometric.nn import ChebConv


def test_cheb_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))

    conv = ChebConv(in_channels, out_channels, K=3)
    assert conv.__repr__() == 'ChebConv(16, 32, K=3, normalization=sym)'
    assert conv(x, edge_index).size() == (num_nodes, out_channels)
    assert conv(x, edge_index, edge_weight).size() == (num_nodes, out_channels)
    assert conv(x, edge_index, edge_weight,
                lambda_max=2.0).size() == (num_nodes, out_channels)
    jitcls = conv.jittable(x=x, edge_index=edge_index, full_eval=True)
    jitconv = jitcls(in_channels, out_channels, K=3)
    jitconv.load_state_dict(conv.state_dict())
    jittedconv = torch.jit.script(jitconv)
    conv.eval()
    jitconv.eval()
    jittedconv.eval()
    assert (torch.abs(conv(x, edge_index) -
            jitconv(x, edge_index)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index) -
            jittedconv(x, edge_index)) < 1e-6).all().item()

    batch = torch.tensor([0, 0, 1, 1])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    num_nodes = edge_index.max().item() + 1
    edge_weight = torch.rand(edge_index.size(1))
    x = torch.randn((num_nodes, in_channels))
    lambda_max = torch.tensor([2.0, 2.0])

    assert conv(x, edge_index, edge_weight, batch).size() == (num_nodes,
                                                              out_channels)
    assert (torch.abs(conv(x, edge_index, edge_weight, batch) -
            jitconv(x, edge_index, edge_weight, batch)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index, edge_weight, batch) -
            jittedconv(x, edge_index, edge_weight, batch)) < 1e-6).all().item()

    assert conv(x, edge_index, edge_weight, batch,
                lambda_max).size() == (num_nodes, out_channels)
    assert (torch.abs(conv(x, edge_index, edge_weight, batch, lambda_max) -
            jitconv(x, edge_index, edge_weight,
                    batch, lambda_max)) < 1e-6).all().item()
    assert (torch.abs(conv(x, edge_index, edge_weight, batch, lambda_max) -
            jittedconv(x, edge_index, edge_weight,
                       batch, lambda_max)) < 1e-6).all().item()
