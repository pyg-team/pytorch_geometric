import torch

from torch_geometric.nn import GraphSAGE, Linear, SAGEConv
from torch_geometric.testing import get_random_edge_index, withPackage


class MyModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)
        self.lin3 = torch.nn.Linear(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        return self.lin3(torch.cat([self.lin1(x), self.lin2(x)], dim=-1))


class MySAGEConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.lin_l = Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.lin_l(x)


@withPackage('fvcore')
def test_fvcore():
    from fvcore.nn import FlopCountAnalysis

    x = torch.randn(10, 16)
    edge_index = get_random_edge_index(10, 10, num_edges=100)

    model = GraphSAGE(16, 32, num_layers=2)

    flops = FlopCountAnalysis(model, (x, edge_index))

    # TODO (matthias) Currently, aggregations are not properly registered.
    assert flops.by_module()['convs.0'] == 2 * 10 * 16 * 32
    assert flops.by_module()['convs.1'] == 2 * 10 * 32 * 32
    assert flops.total() == (flops.by_module()['convs.0'] +
                             flops.by_module()['convs.1'])
    assert flops.by_operator()['linear'] == flops.total()
