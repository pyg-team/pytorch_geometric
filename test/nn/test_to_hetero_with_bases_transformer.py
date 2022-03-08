from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential
from torch_sparse import SparseTensor

from torch_geometric.nn import (GINEConv, MessagePassing, RGCNConv, SAGEConv,
                                to_hetero_with_bases)


class Net1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(16, 32)
        self.lin2 = Linear(8, 16)

    def forward(self, x: Tensor, edge_attr: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.lin1(x)
        edge_attr = self.lin2(edge_attr)
        return x, edge_attr


class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(16, 16)
        self.conv1 = SAGEConv(16, 32)
        self.lin2 = Linear(32, 32)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.lin1(x).relu_()
        x = self.conv1(x, edge_index).relu_()
        x = self.lin2(x)
        return x


class Net3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(8, 16)
        self.conv1 = GINEConv(nn=Linear(16, 32))

    def forward(self, x: Tensor, edge_index: Tensor,
                edge_attr: Tensor) -> Tensor:
        x = self.conv1(x, edge_index, self.lin1(edge_attr))
        return x


class Net4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(16, 16)
        self.conv2 = SAGEConv(16, 16)
        self.lin1 = Linear(3 * 16, 32)

    def forward(self, x0: Tensor, edge_index: Tensor) -> Tensor:
        x1 = self.conv1(x0, edge_index).relu_()
        x2 = self.conv2(x1, edge_index).relu_()
        return self.lin1(torch.cat([x0, x1, x2], dim=-1))


class Net5(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(16, 16))
            self.convs.append(SAGEConv(16, 16))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for lin, conv in zip(self.lins, self.convs):
            x = (conv(x, edge_index) + lin(x))
        return x


class Net6(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.lins = torch.nn.ModuleDict()
        self.convs = torch.nn.ModuleDict()
        for i in range(num_layers):
            self.lins[str(i)] = Linear(16, 16)
            self.convs[str(i)] = SAGEConv(16, 16)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i in range(len(self.lins)):
            x = (self.convs[str(i)](x, edge_index) + self.lins[str(i)](x))
        return x


class Net7(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = Sequential(Linear(16, 16), ReLU(), Linear(16, 16))
        self.conv1 = SAGEConv(16, 32)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.mlp1(x)
        x = self.conv1(x, edge_index)
        return x


def test_to_hetero_with_bases():
    metadata = (['paper', 'author'], [('paper', 'cites', 'paper'),
                                      ('paper', 'written_by', 'author'),
                                      ('author', 'writes', 'paper')])

    x_dict = {'paper': torch.randn(100, 16), 'author': torch.randn(100, 8)}
    edge_index_dict = {
        ('paper', 'cites', 'paper'):
        torch.randint(100, (2, 200), dtype=torch.long),
        ('paper', 'written_by', 'author'):
        torch.randint(100, (2, 200), dtype=torch.long),
        ('author', 'writes', 'paper'):
        torch.randint(100, (2, 200), dtype=torch.long),
    }
    edge_attr_dict = {
        ('paper', 'cites', 'paper'): torch.randn(200, 8),
        ('paper', 'written_by', 'author'): torch.randn(200, 8),
        ('author', 'writes', 'paper'): torch.randn(200, 8),
    }

    model = Net1()
    in_channels = {'x': 16, 'edge_attr': 8}
    model = to_hetero_with_bases(model, metadata, num_bases=4,
                                 in_channels=in_channels, debug=False)
    out = model(x_dict, edge_attr_dict)
    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(out[0], dict) and len(out[0]) == 2
    assert out[0]['paper'].size() == (100, 32)
    assert out[0]['author'].size() == (100, 32)
    assert isinstance(out[1], dict) and len(out[1]) == 3
    assert out[1][('paper', 'cites', 'paper')].size() == (200, 16)
    assert out[1][('paper', 'written_by', 'author')].size() == (200, 16)
    assert out[1][('author', 'writes', 'paper')].size() == (200, 16)

    model = Net2()
    in_channels = {'x': 16}
    model = to_hetero_with_bases(model, metadata, num_bases=4,
                                 in_channels=in_channels, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 32)
    assert out['author'].size() == (100, 32)

    model = Net3()
    in_channels = {'x': 16, 'edge_attr': 8}
    model = to_hetero_with_bases(model, metadata, num_bases=4,
                                 in_channels=in_channels, debug=False)
    out = model(x_dict, edge_index_dict, edge_attr_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 32)
    assert out['author'].size() == (100, 32)

    model = Net4()
    in_channels = {'x': 16}
    model = to_hetero_with_bases(model, metadata, num_bases=4,
                                 in_channels=in_channels, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 32)
    assert out['author'].size() == (100, 32)

    model = Net5(num_layers=2)
    in_channels = {'x': 16}
    model = to_hetero_with_bases(model, metadata, num_bases=4,
                                 in_channels=in_channels, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 16)
    assert out['author'].size() == (100, 16)

    model = Net6(num_layers=2)
    in_channels = {'x': 16}
    model = to_hetero_with_bases(model, metadata, num_bases=4,
                                 in_channels=in_channels, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 16)
    assert out['author'].size() == (100, 16)

    model = Net7()
    in_channels = {'x': 16}
    model = to_hetero_with_bases(model, metadata, num_bases=4,
                                 in_channels=in_channels, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 32)
    assert out['author'].size() == (100, 32)


class GraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        if isinstance(x, Tensor):
            x = (x, x)
        return self.propagate(edge_index, x=(self.lin(x[0]), x[1]))


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)
        self.lin = Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        return self.lin(x) + self.conv(x, edge_index)


def test_to_hetero_with_bases_and_rgcn_equal_output():
    torch.manual_seed(1234)

    # Run `RGCN` with basis decomposition:
    x = torch.randn(10, 16)  # 6 paper nodes, 4 author nodes
    adj = (torch.rand(10, 10) > 0.5)
    adj[6:, 6:] = False
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    row, col = edge_index

    # # 0 = paper<->paper, 1 = author->paper, 2 = paper->author
    edge_type = torch.full((edge_index.size(1), ), -1, dtype=torch.long)
    edge_type[(row < 6) & (col < 6)] = 0
    edge_type[(row < 6) & (col >= 6)] = 1
    edge_type[(row >= 6) & (col < 6)] = 2
    assert edge_type.min() == 0

    num_bases = 4
    conv = RGCNConv(16, 32, num_relations=3, num_bases=num_bases, aggr='add')
    out1 = conv(x, edge_index, edge_type)

    # Run `to_hetero_with_bases`:
    x_dict = {
        'paper': x[:6],
        'author': x[6:],
    }
    edge_index_dict = {
        ('paper', '_', 'paper'):
        edge_index[:, edge_type == 0],
        ('paper', '_', 'author'):
        edge_index[:, edge_type == 1] - torch.tensor([[0], [6]]),
        ('author', '_', 'paper'):
        edge_index[:, edge_type == 2] - torch.tensor([[6], [0]]),
    }

    adj_t_dict = {
        key: SparseTensor.from_edge_index(edge_index).t()
        for key, edge_index in edge_index_dict.items()
    }

    metadata = (list(x_dict.keys()), list(edge_index_dict.keys()))
    model = to_hetero_with_bases(RGCN(16, 32), metadata, num_bases=num_bases,
                                 debug=False)

    # Set model weights:
    for i in range(num_bases):
        model.conv.convs[i].lin.weight.data = conv.weight[i].data.t()
        model.conv.convs[i].edge_type_weight.data = conv.comp[:, i].data.t()

    model.lin.weight.data = conv.root.data.t()
    model.lin.bias.data = conv.bias.data

    out2 = model(x_dict, edge_index_dict)
    out2 = torch.cat([out2['paper'], out2['author']], dim=0)
    assert torch.allclose(out1, out2, atol=1e-6)

    out3 = model(x_dict, adj_t_dict)
    out3 = torch.cat([out3['paper'], out3['author']], dim=0)
    assert torch.allclose(out1, out3, atol=1e-6)
