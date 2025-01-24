from typing import Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential

import torch_geometric.typing
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.nn import (
    GAT,
    BatchNorm,
    GATv2Conv,
    GCNConv,
    GINEConv,
    GraphSAGE,
)
from torch_geometric.nn import Linear as LazyLinear
from torch_geometric.nn import (
    MeanAggregation,
    MessagePassing,
    RGCNConv,
    SAGEConv,
    to_hetero,
)
from torch_geometric.testing import onlyCUDA
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import dropout_edge

torch.fx.wrap('dropout_edge')


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


class Net8(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = LazyLinear(-1, 32)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin1(x)
        return x


class Net9(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = BatchNorm(16)

    def forward(self, x: Tensor) -> Tensor:
        return self.batch_norm(x)


class Net10(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = SAGEConv(16, 32)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.dropout(x, p=0.5, training=self.training)
        edge_index, _ = dropout_edge(edge_index, p=0.5, training=self.training)
        return self.conv(x, edge_index)


class Net11(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = SAGEConv(16, 16)
        self.num_layers = 3

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        xs = [x]
        for _ in range(self.num_layers):
            xs.append(self.conv(xs[-1], edge_index))
        return torch.cat(xs, dim=-1)


class Net12(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = Net8()

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


def test_to_hetero_basic():
    x_dict = {
        'paper': torch.randn(100, 16),
        'author': torch.randn(100, 16),
    }
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

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj_t_dict = {}
        for edge_type, (row, col) in edge_index_dict.items():
            adj_t_dict[edge_type] = SparseTensor(
                row=col,
                col=row,
                sparse_sizes=(100, 100),
            )

    metadata = list(x_dict.keys()), list(edge_index_dict.keys())

    model = Net1()
    model = to_hetero(model, metadata, debug=False)
    out = model(x_dict, edge_attr_dict)
    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(out[0], dict) and len(out[0]) == 2
    assert out[0]['paper'].size() == (100, 32)
    assert out[0]['author'].size() == (100, 32)
    assert isinstance(out[1], dict) and len(out[1]) == 3
    assert out[1][('paper', 'cites', 'paper')].size() == (200, 16)
    assert out[1][('paper', 'written_by', 'author')].size() == (200, 16)
    assert out[1][('author', 'writes', 'paper')].size() == (200, 16)
    assert sum(p.numel() for p in model.parameters()) == 1520

    for aggr in ['sum', 'mean', 'min', 'max', 'mul']:
        model = Net2()
        model = to_hetero(model, metadata, aggr='mean', debug=False)
        assert sum(p.numel() for p in model.parameters()) == 5824

        out1 = model(x_dict, edge_index_dict)
        assert isinstance(out1, dict) and len(out1) == 2
        assert out1['paper'].size() == (100, 32)
        assert out1['author'].size() == (100, 32)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            out2 = model(x_dict, adj_t_dict)
            assert isinstance(out2, dict) and len(out2) == 2
            for key in x_dict.keys():
                assert torch.allclose(out1[key], out2[key], atol=1e-6)

    model = Net3()
    model = to_hetero(model, metadata, debug=False)
    out = model(x_dict, edge_index_dict, edge_attr_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 32)
    assert out['author'].size() == (100, 32)

    model = Net4()
    model = to_hetero(model, metadata, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 32)
    assert out['author'].size() == (100, 32)

    model = Net5(num_layers=2)
    model = to_hetero(model, metadata, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 16)
    assert out['author'].size() == (100, 16)

    model = Net6(num_layers=2)
    model = to_hetero(model, metadata, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 16)
    assert out['author'].size() == (100, 16)

    model = Net7()
    model = to_hetero(model, metadata, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 32)
    assert out['author'].size() == (100, 32)

    model = Net8()
    model = to_hetero(model, metadata, debug=False)
    out = model({'paper': torch.randn(4, 8), 'author': torch.randn(8, 16)})
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (4, 32)
    assert out['author'].size() == (8, 32)

    model = Net9()
    model = to_hetero(model, metadata, debug=False)
    out = model({'paper': torch.randn(4, 16), 'author': torch.randn(8, 16)})
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (4, 16)
    assert out['author'].size() == (8, 16)

    model = Net10()
    with pytest.warns(UserWarning, match="with keyword argument 'training'"):
        model = to_hetero(model, metadata, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 32)
    assert out['author'].size() == (100, 32)

    model = Net11()
    model = to_hetero(model, metadata, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (100, 64)
    assert out['author'].size() == (100, 64)

    model = Net12()
    with pytest.warns(UserWarning, match="parameters cannot be reset"):
        model = to_hetero(model, metadata, debug=False)
    out = model({'paper': torch.randn(4, 8), 'author': torch.randn(8, 16)})
    assert isinstance(out, dict) and len(out) == 2
    assert out['paper'].size() == (4, 32)
    assert out['author'].size() == (8, 32)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 64)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x


def test_to_hetero_with_gcn():
    x_dict = {
        'paper': torch.randn(100, 16),
    }
    edge_index_dict = {
        ('paper', 'cites', 'paper'): torch.randint(100, (2, 200)),
        ('paper', 'rev_cites', 'paper'): torch.randint(100, (2, 200)),
    }
    metadata = list(x_dict.keys()), list(edge_index_dict.keys())

    model = GCN()
    model = to_hetero(model, metadata, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 1
    assert out['paper'].size() == (100, 64)


def test_to_hetero_with_basic_model():
    x_dict = {
        'paper': torch.randn(100, 16),
        'author': torch.randn(100, 16),
    }
    edge_index_dict = {
        ('paper', 'cites', 'paper'):
        torch.randint(100, (2, 200), dtype=torch.long),
        ('paper', 'written_by', 'author'):
        torch.randint(100, (2, 200), dtype=torch.long),
        ('author', 'writes', 'paper'):
        torch.randint(100, (2, 200), dtype=torch.long),
    }

    metadata = list(x_dict.keys()), list(edge_index_dict.keys())

    model = GraphSAGE((-1, -1), 32, num_layers=3)
    model = to_hetero(model, metadata, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2

    model = GAT((-1, -1), 32, num_layers=3, add_self_loops=False)
    model = to_hetero(model, metadata, debug=False)
    out = model(x_dict, edge_index_dict)
    assert isinstance(out, dict) and len(out) == 2


class GraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='sum')
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


def test_to_hetero_and_rgcn_equal_output():
    torch.manual_seed(1234)

    # Run `RGCN`:
    x = torch.randn(10, 16)  # 6 paper nodes, 4 author nodes
    adj = (torch.rand(10, 10) > 0.5)
    adj[6:, 6:] = False
    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    row, col = edge_index

    # # 0 = paper<->paper, 1 = paper->author, 2 = author->paper
    edge_type = torch.full((edge_index.size(1), ), -1, dtype=torch.long)
    edge_type[(row < 6) & (col < 6)] = 0
    edge_type[(row < 6) & (col >= 6)] = 1
    edge_type[(row >= 6) & (col < 6)] = 2
    assert edge_type.min() == 0

    conv = RGCNConv(16, 32, num_relations=3, aggr='sum')
    out1 = conv(x, edge_index, edge_type)

    # Run `to_hetero`:
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

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj_t_dict = {
            key: SparseTensor.from_edge_index(edge_index).t()
            for key, edge_index in edge_index_dict.items()
        }

    node_types, edge_types = list(x_dict.keys()), list(edge_index_dict.keys())

    model = to_hetero(RGCN(16, 32), (node_types, edge_types))

    # Set model weights:
    for i, edge_type in enumerate(edge_types):
        weight = model.conv['__'.join(edge_type)].lin.weight
        weight.data = conv.weight[i].data.t()
    for i, node_type in enumerate(node_types):
        model.lin[node_type].weight.data = conv.root.data.t()
        model.lin[node_type].bias.data = conv.bias.data

    out2 = model(x_dict, edge_index_dict)
    out2 = torch.cat([out2['paper'], out2['author']], dim=0)
    assert torch.allclose(out1, out2, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        out3 = model(x_dict, adj_t_dict)
        out3 = torch.cat([out3['paper'], out3['author']], dim=0)
        assert torch.allclose(out1, out3, atol=1e-6)


class GraphLevelGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = SAGEConv(16, 32)
        self.pool = MeanAggregation()
        self.lin = Linear(32, 64)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = self.conv(x, edge_index)
        x = self.pool(x, batch)
        x = self.lin(x)
        return x


def test_graph_level_to_hetero():
    x_dict = {
        'paper': torch.randn(100, 16),
        'author': torch.randn(100, 16),
    }
    edge_index_dict = {
        ('paper', 'written_by', 'author'):
        torch.randint(100, (2, 200), dtype=torch.long),
        ('author', 'writes', 'paper'):
        torch.randint(100, (2, 200), dtype=torch.long),
    }
    batch_dict = {
        'paper': torch.zeros(100, dtype=torch.long),
        'author': torch.zeros(100, dtype=torch.long),
    }

    metadata = list(x_dict.keys()), list(edge_index_dict.keys())

    model = GraphLevelGNN()
    model = to_hetero(model, metadata, aggr='mean', debug=False)
    out = model(x_dict, edge_index_dict, batch_dict)
    assert out.size() == (1, 64)


class MessagePassingLoops(MessagePassing):
    def __init__(self):
        super().__init__()
        self.add_self_loops = True

    def forward(self, x):
        return x


class ModelLoops(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = MessagePassingLoops()

    def forward(self, x):
        return self.conv(x)


def test_hetero_transformer_self_loop_error():
    to_hetero(ModelLoops(), metadata=(['a'], [('a', 'to', 'a')]))
    with pytest.raises(ValueError, match="incorrect message passing"):
        to_hetero(ModelLoops(), metadata=(['a', 'b'], [('a', 'to', 'b'),
                                                       ('b', 'to', 'a')]))


def test_to_hetero_validate():
    model = Net1()
    metadata = (['my test'], [('my test', 'rel', 'my test')])

    with pytest.warns(UserWarning, match="letters, numbers and underscores"):
        model = to_hetero(model, metadata, debug=False)


def test_to_hetero_on_static_graphs():
    x_dict = {
        'paper': torch.randn(4, 100, 16),
        'author': torch.randn(4, 100, 16),
    }
    edge_index_dict = {
        ('paper', 'written_by', 'author'): torch.randint(100, (2, 200)),
        ('author', 'writes', 'paper'): torch.randint(100, (2, 200)),
    }

    metadata = list(x_dict.keys()), list(edge_index_dict.keys())
    model = to_hetero(Net4(), metadata, debug=False)
    out_dict = model(x_dict, edge_index_dict)

    assert len(out_dict) == 2
    assert out_dict['paper'].size() == (4, 100, 32)
    assert out_dict['author'].size() == (4, 100, 32)


@onlyCUDA
def test_to_hetero_lazy_cuda():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = GATv2Conv(
                (-1, -1),
                out_channels=2,
                add_self_loops=False,
                edge_dim=-1,
                heads=1,
            ).to('cuda')

        def forward(self, x, edge_index, edge_attr):
            return self.conv(x, edge_index, edge_attr)

    data = FakeHeteroDataset(edge_dim=10)[0].to('cuda')
    model = to_hetero(Model(), data.metadata())
    out_dict = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    assert len(out_dict) == len(data.node_types)
    for out in out_dict.values():
        assert out.is_cuda
        assert out.size(-1) == 2
