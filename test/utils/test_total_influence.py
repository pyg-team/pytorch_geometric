import torch

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import total_influence


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 6)
        self.conv2 = GCNConv(6, 7)

    def forward(self, x0, edge_index):
        x1 = self.conv1(x0, edge_index)
        x2 = self.conv2(x1, edge_index)
        return x2


def test_total_influence_smoke():
    x = torch.randn(6, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    max_hops = 2
    num_samples = 4
    data = Data(
        x=x,
        edge_index=edge_index,
    )
    model = GNN()
    I, R = total_influence(
        model,
        data,
        max_hops=max_hops,
        num_samples=num_samples,
    )

    assert I.shape == (max_hops + 1, )
    assert 0.0 <= R <= max_hops

    I, R = total_influence(
        model,
        data,
        max_hops=max_hops,
        num_samples=num_samples,
        average=False,
    )
    assert I.shape == torch.Size([num_samples, max_hops + 1])
