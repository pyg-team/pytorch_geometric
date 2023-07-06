import torch

from torch_geometric.nn import GraphSAGE
from torch_geometric.testing import get_random_edge_index, withPackage


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
