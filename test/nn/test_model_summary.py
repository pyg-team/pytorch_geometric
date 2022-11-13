import pytest
import torch
from torch import Tensor, nn
from torch_sparse import SparseTensor

from torch_geometric.nn import Linear, SAGEConv, summary, to_hetero
from torch_geometric.nn.models import GCN
from torch_geometric.testing import withPackage


class SAGE(torch.nn.Module):
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


class ModuleDictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.activations = nn.ModuleDict({
            "lrelu": nn.LeakyReLU(),
            "prelu": nn.PReLU()
        })

    def forward(self, x: torch.Tensor, activation_type: str) -> torch.Tensor:
        x = self.activations[activation_type](x)
        return x


@pytest.fixture
def gcn():
    model = GCN(32, 16, num_layers=2, out_channels=32)
    x = torch.randn(100, 32)
    edge_index = torch.randint(100, size=(2, 20))
    adj_t = SparseTensor.from_edge_index(edge_index,
                                         sparse_sizes=(100, 100)).t()
    return {'model': model, 'x': x, 'edge_index': edge_index, 'adj_t': adj_t}


@withPackage('tabulate')
def test_basic_summary(gcn):

    expected = '\n'.join([
        '+---------------------+--------------------+----------------+----------+',  # noqa
        '| Layer               | Input Shape        | Output Shape   | #Param   |',  # noqa
        '|---------------------+--------------------+----------------+----------|',  # noqa
        '| GCN                 | [100, 32], [2, 20] | [100, 32]      | 1,072    |',  # noqa
        '| ├─(act)ReLU         | [100, 16]          | [100, 16]      | --       |',  # noqa
        '| ├─(convs)ModuleList | --                 | --             | 1,072    |',  # noqa
        '| │    └─(0)GCNConv   | [100, 32], [2, 20] | [100, 16]      | 528      |',  # noqa
        '| │    └─(1)GCNConv   | [100, 16], [2, 20] | [100, 32]      | 544      |',  # noqa
        '+---------------------+--------------------+----------------+----------+',  # noqa
    ])

    assert summary(gcn['model'], gcn['x'], gcn['edge_index']) == expected


@withPackage('tabulate')
def test_summary_with_sparse_tensor(gcn):

    expected = '\n'.join([
        '+---------------------+-----------------------+----------------+----------+',  # noqa
        '| Layer               | Input Shape           | Output Shape   | #Param   |',  # noqa
        '|---------------------+-----------------------+----------------+----------|',  # noqa
        '| GCN                 | [100, 32], [100, 100] | [100, 32]      | 1,072    |',  # noqa
        '| ├─(act)ReLU         | [100, 16]             | [100, 16]      | --       |',  # noqa
        '| ├─(convs)ModuleList | --                    | --             | 1,072    |',  # noqa
        '| │    └─(0)GCNConv   | [100, 32], [100, 100] | [100, 16]      | 528      |',  # noqa
        '| │    └─(1)GCNConv   | [100, 16], [100, 100] | [100, 32]      | 544      |',  # noqa
        '+---------------------+-----------------------+----------------+----------+',  # noqa
    ])

    assert summary(gcn['model'], gcn['x'], gcn['adj_t']) == expected


@withPackage('tabulate')
def test_summary_with_max_depth(gcn):
    expected = '\n'.join([
        '+---------------------+--------------------+----------------+----------+',  # noqa
        '| Layer               | Input Shape        | Output Shape   | #Param   |',  # noqa
        '|---------------------+--------------------+----------------+----------|',  # noqa
        '| GCN                 | [100, 32], [2, 20] | [100, 32]      | 1,072    |',  # noqa
        '| ├─(act)ReLU         | [100, 16]          | [100, 16]      | --       |',  # noqa
        '| ├─(convs)ModuleList | --                 | --             | 1,072    |',  # noqa
        '+---------------------+--------------------+----------------+----------+',  # noqa
    ])

    assert summary(gcn['model'], gcn['x'], gcn['edge_index'],
                   max_depth=1) == expected


@withPackage('tabulate')
def test_summary_with_leaf_module(gcn):

    expected = '\n'.join([
        '+-----------------------------------------+--------------------+----------------+----------+',  # noqa
        '| Layer                                   | Input Shape        | Output Shape   | #Param   |',  # noqa
        '|-----------------------------------------+--------------------+----------------+----------|',  # noqa
        '| GCN                                     | [100, 32], [2, 20] | [100, 32]      | 1,072    |',  # noqa
        '| ├─(act)ReLU                             | [100, 16]          | [100, 16]      | --       |',  # noqa
        '| ├─(convs)ModuleList                     | --                 | --             | 1,072    |',  # noqa
        '| │    └─(0)GCNConv                       | [100, 32], [2, 20] | [100, 16]      | 528      |',  # noqa
        '| │    │    └─(aggr_module)SumAggregation | [120, 16], [120]   | [100, 16]      | --       |',  # noqa
        '| │    │    └─(lin)Linear                 | [100, 32]          | [100, 16]      | 512      |',  # noqa
        '| │    └─(1)GCNConv                       | [100, 16], [2, 20] | [100, 32]      | 544      |',  # noqa
        '| │    │    └─(aggr_module)SumAggregation | [120, 32], [120]   | [100, 32]      | --       |',  # noqa
        '| │    │    └─(lin)Linear                 | [100, 16]          | [100, 32]      | 512      |',  # noqa
        '+-----------------------------------------+--------------------+----------------+----------+',  # noqa
    ])
    assert summary(gcn['model'], gcn['x'], gcn['edge_index'],
                   leaf_module=None) == expected


@withPackage('tabulate')
def test_reusing_activation_layers():
    act = nn.ReLU(inplace=True)
    model1 = nn.Sequential(act, nn.Identity(), act, nn.Identity(), act)
    model2 = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.Identity(),
        nn.ReLU(inplace=True),
        nn.Identity(),
        nn.ReLU(inplace=True),
    )

    x = torch.randn(10)
    result_1 = summary(model1, x)
    result_2 = summary(model2, x)

    assert len(result_1) == len(result_2)


@withPackage('tabulate')
def test_hetero():

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
    model = to_hetero(SAGE(), metadata, debug=False)

    expected = '\n'.join([
        '+--------------------------------------------+---------------------+----------------+----------+',  # noqa
        '| Layer                                      | Input Shape         | Output Shape   | #Param   |',  # noqa
        '|--------------------------------------------+---------------------+----------------+----------|',  # noqa
        '| GraphModule                                |                     |                | 5,824    |',  # noqa
        '| ├─(lin1)ModuleDict                         | --                  | --             | 544      |',  # noqa
        '| │    └─(paper)Linear                       | [100, 16]           | [100, 16]      | 272      |',  # noqa
        '| │    └─(author)Linear                      | [100, 16]           | [100, 16]      | 272      |',  # noqa
        '| ├─(conv1)ModuleDict                        | --                  | --             | 3,168    |',  # noqa
        '| │    └─(paper__cites__paper)SAGEConv       | [100, 16], [2, 200] | [100, 32]      | 1,056    |',  # noqa
        '| │    └─(paper__written_by__author)SAGEConv | [2, 200]            | [100, 32]      | 1,056    |',  # noqa
        '| │    └─(author__writes__paper)SAGEConv     | [2, 200]            | [100, 32]      | 1,056    |',  # noqa
        '| ├─(lin2)ModuleDict                         | --                  | --             | 2,112    |',  # noqa
        '| │    └─(paper)Linear                       | [100, 32]           | [100, 32]      | 1,056    |',  # noqa
        '| │    └─(author)Linear                      | [100, 32]           | [100, 32]      | 1,056    |',  # noqa
        '+--------------------------------------------+---------------------+----------------+----------+',  # noqa
    ])

    assert summary(model, x_dict, edge_index_dict) == expected


@withPackage('tabulate')
def test_moduledict_model():

    model = ModuleDictModel()
    x = torch.randn(100, 32)

    expected = '\n'.join([
        '+---------------------------+---------------+----------------+----------+',  # noqa
        '| Layer                     | Input Shape   | Output Shape   | #Param   |',  # noqa
        '|---------------------------+---------------+----------------+----------|',  # noqa
        '| ModuleDictModel           | [100, 32]     | [100, 32]      | 1        |',  # noqa
        '| ├─(activations)ModuleDict | --            | --             | 1        |',  # noqa
        '| │    └─(lrelu)LeakyReLU   | --            | --             | --       |',  # noqa
        '| │    └─(prelu)PReLU       | [100, 32]     | [100, 32]      | 1        |',  # noqa
        '+---------------------------+---------------+----------------+----------+',  # noqa
    ])

    summary(model, x, 'prelu') == expected


@withPackage('tabulate')
def test_jit_model():
    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
    model = torch.jit.script(model)
    x = torch.randn(100, 32)

    expected = '\n'.join([
        '+----------------------------+---------------+----------------+----------+',  # noqa
        '| Layer                      | Input Shape   | Output Shape   | #Param   |',  # noqa
        '|----------------------------+---------------+----------------+----------|',  # noqa
        '| RecursiveScriptModule      | --            | --             | 664      |',  # noqa
        '| ├─(0)RecursiveScriptModule | --            | --             | 528      |',  # noqa
        '| ├─(1)RecursiveScriptModule | --            | --             | --       |',  # noqa
        '| ├─(2)RecursiveScriptModule | --            | --             | 136      |',  # noqa
        '+----------------------------+---------------+----------------+----------+',  # noqa
    ])

    assert summary(model, x) == expected
