import torch
from torch import Tensor, nn

from torch_geometric.nn import Linear, SAGEConv, summary, to_hetero
from torch_geometric.nn.models import GCN


class HeteroGNN(torch.nn.Module):
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
    """Model that uses a ModuleDict."""
    def __init__(self):
        super().__init__()
        self.activations = nn.ModuleDict({
            "lrelu": nn.LeakyReLU(),
            "prelu": nn.PReLU()
        })

    def forward(self, x: torch.Tensor, activation_type: str) -> torch.Tensor:
        x = self.activations[activation_type](x)
        return x


def test_basic_summary() -> None:

    model = GCN(32, 16, num_layers=2, out_channels=32)
    x = torch.randn(100, 32)
    edge_index = torch.randint(100, size=(2, 20))
    print(summary(model, x, edge_index))

    print(summary(model, x, edge_index, max_depth=1))

    # test deep models
    model = GCN(32, 16, num_layers=20, out_channels=32)
    print(summary(model, x, edge_index))


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


def test_hetero() -> None:

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
    model = HeteroGNN()
    model = to_hetero(model, metadata, debug=False)
    print(summary(model, x_dict, edge_index_dict))


def test_moduledict_model():

    model = ModuleDictModel()
    x = torch.randn(100, 32)
    print(summary(model, x, 'prelu'))
