import pytest
import torch

from torch_geometric.nn.aggr import SetTransformerAggregation


@pytest.mark.parametrize('num_PMA_outputs', [1, 4, 8])
@pytest.mark.parametrize('num_enc_SABs', [0, 1, 2])
@pytest.mark.parametrize('num_dec_SABs', [0, 1, 2])
@pytest.mark.parametrize('dim_hidden', [8, 64])
@pytest.mark.parametrize('num_heads', [1, 8])
def test_set_transformer_aggregation(num_PMA_outputs, num_enc_SABs,
                                     num_dec_SABs, dim_hidden, num_heads):
    x = torch.randn(14, 9)
    index = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    out_channels = 16
    st_aggr = SetTransformerAggregation(
        in_channels=9, out_channels=out_channels,
        num_PMA_outputs=num_PMA_outputs, num_enc_SABs=num_enc_SABs,
        num_dec_SABs=num_dec_SABs, dim_hidden=dim_hidden, num_heads=num_heads)
    st_aggr.reset_parameters()

    assert str(st_aggr) == (
        f'{SetTransformerAggregation.__name__}(9, {out_channels}, '
        f'{num_PMA_outputs}, {num_enc_SABs}, '
        f'{num_dec_SABs}, {dim_hidden}, {num_heads}, {False})')

    assert st_aggr(x, index).size() == (2, out_channels)
