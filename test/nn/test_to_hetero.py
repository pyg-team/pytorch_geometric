import torch
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.to_hetero import to_hetero


class Net1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(16, 16)
        self.conv1 = SAGEConv(16, 16)
        self.lin2 = Linear(16, 16)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x = self.lin1(x).relu_()
        x = self.conv1(x, edge_index)
        # x = self.lin2(x).relu_()
        # x = x + x
        return x


def test_to_hetero():
    model = Net1()

    metadata = (['p', 'a'], [('p', '_', 'p'), ('a', '_', 'p'), ('p', '_', 'a'),
                             ('a', '_', 'a')])

    x_dict = {'p': torch.randn(100, 16), 'a': torch.randn(100, 16)}
    edge_index_dict = {
        ('p', '_', 'p'): torch.randint(100, (2, 200), dtype=torch.long),
        ('p', '_', 'a'): torch.randint(100, (2, 200), dtype=torch.long),
        ('a', '_', 'p'): torch.randint(100, (2, 200), dtype=torch.long),
        ('a', '_', 'a'): torch.randint(100, (2, 200), dtype=torch.long),
    }
    # y_dict = {'p': torch.randn(100, 16), 'a': torch.randn(100, 16)}

    out_model = to_hetero(model, metadata)
    out_model(x_dict, edge_index_dict)
