import torch

from torch_geometric.nn import SumAggregation
from torch_geometric.nn.to_fixed_size_transformer import to_fixed_size


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aggr = SumAggregation()

    def forward(self, x, batch):
        return self.aggr(x, batch, dim=0)


def test_to_fixed_size():
    graph_module = to_fixed_size(Model(), batch_size=10)
    x = torch.randn(10)
    batch = torch.zeros_like(x).long()
    r = graph_module(x, batch)
    assert r.shape[0] == 10
