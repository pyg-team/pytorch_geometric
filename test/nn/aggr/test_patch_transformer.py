import torch

from torch_geometric.nn import PatchTransformerAggregation
from torch_geometric.testing import withCUDA


@withCUDA
def test_patch_transformer_aggregation(device: torch.device) -> None:
    aggr = PatchTransformerAggregation(
        in_channels=16,
        out_channels=32,
        patch_size=2,
        hidden_channels=8,
        num_transformer_blocks=1,
        heads=2,
        dropout=0.2,
        aggr=['sum', 'mean', 'min', 'max', 'var', 'std'],
        device=device,
    )
    aggr.reset_parameters()
    assert str(aggr) == 'PatchTransformerAggregation(16, 32, patch_size=2)'

    index = torch.tensor([0, 0, 1, 1, 1, 2], device=device)
    x = torch.randn(index.size(0), 16, device=device)

    out = aggr(x, index)
    assert out.device == device
    assert out.size() == (3, aggr.out_channels)
