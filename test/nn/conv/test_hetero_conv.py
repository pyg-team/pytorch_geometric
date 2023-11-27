import random

import pytest
import torch

import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.nn import (
    GATConv,
    GCN2Conv,
    GCNConv,
    HeteroConv,
    Linear,
    MessagePassing,
    SAGEConv,
)
from torch_geometric.profile import benchmark
from torch_geometric.testing import (
    disableExtensions,
    get_random_edge_index,
    onlyLinux,
    withCUDA,
    withPackage,
)


@pytest.mark.parametrize('aggr', ['sum', 'mean', 'min', 'max', 'cat', None])
def test_hetero_conv(aggr):
    data = HeteroData()
    data['paper'].x = torch.randn(50, 32)
    data['author'].x = torch.randn(30, 64)
    data['paper', 'paper'].edge_index = get_random_edge_index(50, 50, 200)
    data['paper', 'author'].edge_index = get_random_edge_index(50, 30, 100)
    data['paper', 'author'].edge_attr = torch.randn(100, 3)
    data['author', 'paper'].edge_index = get_random_edge_index(30, 50, 100)
    data['paper', 'paper'].edge_weight = torch.rand(200)

    # Unspecified edge types should be ignored:
    data['author', 'author'].edge_index = get_random_edge_index(30, 30, 100)

    conv = HeteroConv(
        {
            ('paper', 'to', 'paper'):
            GCNConv(-1, 64),
            ('author', 'to', 'paper'):
            SAGEConv((-1, -1), 64),
            ('paper', 'to', 'author'):
            GATConv((-1, -1), 64, edge_dim=3, add_self_loops=False),
        },
        aggr=aggr,
    )

    assert len(list(conv.parameters())) > 0
    assert str(conv) == 'HeteroConv(num_relations=3)'

    out_dict = conv(
        data.x_dict,
        data.edge_index_dict,
        data.edge_attr_dict,
        edge_weight_dict=data.edge_weight_dict,
    )

    assert len(out_dict) == 2
    if aggr == 'cat':
        assert out_dict['paper'].size() == (50, 128)
        assert out_dict['author'].size() == (30, 64)
    elif aggr is not None:
        assert out_dict['paper'].size() == (50, 64)
        assert out_dict['author'].size() == (30, 64)
    else:
        assert out_dict['paper'].size() == (50, 2, 64)
        assert out_dict['author'].size() == (30, 1, 64)


def test_gcn2_hetero_conv():
    data = HeteroData()
    data['paper'].x = torch.randn(50, 32)
    data['author'].x = torch.randn(30, 64)
    data['paper', 'paper'].edge_index = get_random_edge_index(50, 50, 200)
    data['author', 'author'].edge_index = get_random_edge_index(30, 30, 100)
    data['paper', 'paper'].edge_weight = torch.rand(200)

    conv = HeteroConv({
        ('paper', 'to', 'paper'): GCN2Conv(32, alpha=0.1),
        ('author', 'to', 'author'): GCN2Conv(64, alpha=0.2),
    })

    out_dict = conv(
        data.x_dict,
        data.x_dict,
        data.edge_index_dict,
        edge_weight_dict=data.edge_weight_dict,
    )

    assert len(out_dict) == 2
    assert out_dict['paper'].size() == (50, 32)
    assert out_dict['author'].size() == (30, 64)


class CustomConv(MessagePassing):
    def __init__(self, out_channels):
        super().__init__(aggr='add')
        self.lin = Linear(-1, out_channels)

    def forward(self, x, edge_index, y, z):
        return self.propagate(edge_index, x=x, y=y, z=z)

    def message(self, x_j, y_j, z_j):
        return self.lin(torch.cat([x_j, y_j, z_j], dim=-1))


def test_hetero_conv_with_custom_conv():
    data = HeteroData()
    data['paper'].x = torch.randn(50, 32)
    data['paper'].y = torch.randn(50, 3)
    data['paper'].z = torch.randn(50, 3)
    data['author'].x = torch.randn(30, 64)
    data['author'].y = torch.randn(30, 3)
    data['author'].z = torch.randn(30, 3)
    data['paper', 'paper'].edge_index = get_random_edge_index(50, 50, 200)
    data['paper', 'author'].edge_index = get_random_edge_index(50, 30, 100)
    data['author', 'paper'].edge_index = get_random_edge_index(30, 50, 100)

    conv = HeteroConv({key: CustomConv(64) for key in data.edge_types})
    # Test node `args_dict` and `kwargs_dict` with `y_dict` and `z_dict`:
    out_dict = conv(
        data.x_dict,
        data.edge_index_dict,
        data.y_dict,
        z_dict=data.z_dict,
    )
    assert len(out_dict) == 2
    assert out_dict['paper'].size() == (50, 64)
    assert out_dict['author'].size() == (30, 64)


class MessagePassingLoops(MessagePassing):
    def __init__(self):
        super().__init__()
        self.add_self_loops = True


def test_hetero_conv_self_loop_error():
    HeteroConv({('a', 'to', 'a'): MessagePassingLoops()})
    with pytest.raises(ValueError, match="incorrect message passing"):
        HeteroConv({('a', 'to', 'b'): MessagePassingLoops()})


def test_hetero_conv_with_dot_syntax_node_types():
    data = HeteroData()
    data['src.paper'].x = torch.randn(50, 32)
    data['author'].x = torch.randn(30, 64)
    edge_index = get_random_edge_index(50, 50, 200)
    data['src.paper', 'src.paper'].edge_index = edge_index
    data['src.paper', 'author'].edge_index = get_random_edge_index(50, 30, 100)
    data['author', 'src.paper'].edge_index = get_random_edge_index(30, 50, 100)
    data['src.paper', 'src.paper'].edge_weight = torch.rand(200)

    conv = HeteroConv({
        ('src.paper', 'to', 'src.paper'):
        GCNConv(-1, 64),
        ('author', 'to', 'src.paper'):
        SAGEConv((-1, -1), 64),
        ('src.paper', 'to', 'author'):
        GATConv((-1, -1), 64, add_self_loops=False),
    })

    assert len(list(conv.parameters())) > 0
    assert str(conv) == 'HeteroConv(num_relations=3)'

    out_dict = conv(
        data.x_dict,
        data.edge_index_dict,
        edge_weight_dict=data.edge_weight_dict,
    )

    assert len(out_dict) == 2
    assert out_dict['src.paper'].size() == (50, 64)
    assert out_dict['author'].size() == (30, 64)


@withCUDA
@onlyLinux
@disableExtensions
@withPackage('torch>=2.1.0')
def test_compile_hetero_conv_graph_breaks(device):
    import torch._dynamo as dynamo

    data = HeteroData()
    data['a'].x = torch.randn(50, 16, device=device)
    data['b'].x = torch.randn(50, 16, device=device)
    edge_index = get_random_edge_index(50, 50, 100, device=device)
    data['a', 'to', 'b'].edge_index = edge_index
    data['b', 'to', 'a'].edge_index = edge_index.flip([0])

    conv = HeteroConv({
        ('a', 'to', 'b'): SAGEConv(16, 32).jittable(),
        ('b', 'to', 'a'): SAGEConv(16, 32).jittable(),
    }).to(device)

    explanation = dynamo.explain(conv)(data.x_dict, data.edge_index_dict)
    assert explanation.graph_break_count == 0

    compiled_conv = torch_geometric.compile(conv)

    expected = conv(data.x_dict, data.edge_index_dict)
    out = compiled_conv(data.x_dict, data.edge_index_dict)
    assert len(out) == len(expected)
    for key in expected.keys():
        assert torch.allclose(out[key], expected[key], atol=1e-6)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    dataset = FakeHeteroDataset(num_graphs=10).to(args.device)

    def gen_args():
        data = dataset[random.randrange(len(dataset))]
        return data.x_dict, data.edge_index_dict

    class HeteroGNN(torch.nn.Module):
        def __init__(self, channels: int = 32, num_layers: int = 2):
            super().__init__()
            self.convs = torch.nn.ModuleList()

            conv = HeteroConv({
                edge_type:
                SAGEConv(
                    in_channels=(
                        dataset.num_features[edge_type[0]],
                        dataset.num_features[edge_type[-1]],
                    ),
                    out_channels=channels,
                )
                for edge_type in dataset[0].edge_types
            })
            self.convs.append(conv)

            for _ in range(num_layers - 1):
                conv = HeteroConv({
                    edge_type:
                    SAGEConv((channels, channels), channels)
                    for edge_type in dataset[0].edge_types
                })
                self.convs.append(conv)

            self.lin = Linear(channels, 1)

        def forward(self, x_dict, edge_index_dict):
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: x.relu() for key, x in x_dict.items()}
            return self.lin(x_dict['v0'])

    model = HeteroGNN().to(args.device)
    compiled_model = torch_geometric.compile(model)

    benchmark(
        funcs=[model, compiled_model],
        func_names=['Vanilla', 'Compiled'],
        args=gen_args,
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
        backward=args.backward,
    )
