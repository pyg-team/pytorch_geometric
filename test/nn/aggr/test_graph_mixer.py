import torch

from torch_geometric.nn import MLPMixerAggregation


def test_graph_mixer_aggregation():
    x = torch.randn(6, 5)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    aggr = MLPMixerAggregation(
        in_channels=5,
        out_channels=7,
        max_num_elements=3,
        num_layers=1,
    )
    out = aggr(x, index)
    assert out.size() == (3, 7)
