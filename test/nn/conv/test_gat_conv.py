import torch
from torch_geometric.nn import GATConv


class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

    def forward(self, x, edge_index):
        return self.conv(x, edge_index, return_attention_weights=True)


def test_gat_conv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    conv = GATConv(in_channels, out_channels, heads=2, dropout=0.0)
    assert conv.__repr__() == 'GATConv(16, 32, heads=2)'
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, 2 * out_channels)
    assert conv((x, x), edge_index).tolist() == out.tolist()

    model = Net1()
    model.conv = conv.jittable()
    model = torch.jit.script(model)
    assert model(x, edge_index).tolist() == out.tolist()

    result = conv(x, edge_index, return_attention_weights=True)
    assert result[0].tolist() == out.tolist()
    assert result[1][0][:, :-num_nodes].tolist() == edge_index.tolist()
    assert result[1][1].size() == (edge_index.size(1) + num_nodes, 2)
    assert conv._alpha is None

    model = Net2()
    model.conv = conv.jittable()
    model = torch.jit.script(model)
    assert model(x, edge_index)[0].tolist() == result[0].tolist()
    assert model(x, edge_index)[1][0].tolist() == result[1][0].tolist()
    assert model(x, edge_index)[1][1].tolist() == result[1][1].tolist()

    conv = GATConv(in_channels, out_channels, heads=2, concat=False)
    out = conv(x, edge_index)
    assert out.size() == (num_nodes, out_channels)
    assert conv((x, x), edge_index).tolist() == out.tolist()

    model = Net1()
    model.conv = conv.jittable()
    model = torch.jit.script(model)
    assert model(x, edge_index).tolist() == out.tolist()

    result = conv(x, edge_index, return_attention_weights=True)
    assert result[0].tolist() == out.tolist()
    assert result[1][0][:, :-num_nodes].tolist() == edge_index.tolist()
    assert result[1][1].size() == (edge_index.size(1) + num_nodes, 2)
    assert conv._alpha is None

    model = Net2()
    model.conv = conv.jittable()
    model = torch.jit.script(model)
    assert model(x, edge_index)[0].tolist() == result[0].tolist()
    assert model(x, edge_index)[1][0].tolist() == result[1][0].tolist()
    assert model(x, edge_index)[1][1].tolist() == result[1][1].tolist()
