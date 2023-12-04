import torch

from torch_geometric.nn import SumAggregation
from torch_geometric.nn.to_fixed_size_transformer import to_fixed_size


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aggr = SumAggregation()

    def forward(self, x, batch):
        return self.aggr(x, batch, dim=0)


def test_to_fixed_size():
    x = torch.randn(10, 16)
    batch = torch.zeros(10, dtype=torch.long)

    model = Model()
    assert model(x, batch).size() == (1, 16)

    model = to_fixed_size(model, batch_size=10)
    assert model(x, batch).size() == (10, 16)
