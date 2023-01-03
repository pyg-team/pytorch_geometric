import pytest
import torch

from torch_geometric.nn.aggr import (
    DeepSetsAggregation,
    GRUAggregation,
    MLPAggregation,
    SetTransformerAggregation,
)


def test_mlp_aggregation():
    x = torch.randn(14, 8)
    index = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    aggr = MLPAggregation(
        in_channels=8,
        out_channels=32,
        max_num_elements=8,
        num_layers=1,
    )
    aggr.reset_parameters()

    assert str(aggr) == 'MLPAggregation(8, 32, max_num_elements=8)'
    assert aggr(x, index).size() == (2, 32)


@pytest.mark.parametrize('num_PMA_outputs', [1, 4, 8])
@pytest.mark.parametrize('num_enc_SABs', [0, 1, 2])
@pytest.mark.parametrize('num_dec_SABs', [0, 1, 2])
@pytest.mark.parametrize('dim_hidden', [8, 64])
@pytest.mark.parametrize('num_heads', [1, 8])
@pytest.mark.parametrize('max_num_nodes_in_dataset', [14, 50])
def test_set_transformer_aggregation(num_PMA_outputs, num_enc_SABs,
                                     num_dec_SABs, max_num_nodes_in_dataset,
                                     dim_hidden, num_heads):
    x = torch.randn(14, 9)
    index = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    out_channels = 16
    st_aggr = SetTransformerAggregation(
        in_channels=9, out_channels=out_channels,
        max_num_nodes_in_dataset=max_num_nodes_in_dataset,
        num_PMA_outputs=num_PMA_outputs, num_enc_SABs=num_enc_SABs,
        num_dec_SABs=num_dec_SABs, dim_hidden=dim_hidden, num_heads=num_heads)
    st_aggr.reset_parameters()

    assert str(st_aggr) == (
        f'{SetTransformerAggregation.__name__}({max_num_nodes_in_dataset}, '
        f'9, {out_channels}, {num_PMA_outputs}, {num_enc_SABs}, '
        f'{num_dec_SABs}, {dim_hidden}, {num_heads}, {False})')

    assert st_aggr(x, index).size() == (2, out_channels)


@pytest.mark.parametrize('num_layers', [1, 2, 3])
def test_gru_aggregation(num_layers):
    x = torch.randn(14, 9)
    index = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    out_channels = 16
    gru_aggr = GRUAggregation(in_channels=9, out_channels=out_channels,
                              num_layers=num_layers)
    gru_aggr.reset_parameters()

    assert str(gru_aggr) == (f'{GRUAggregation.__name__}(9, 16, {num_layers})')

    assert gru_aggr(x, index).size() == (2, out_channels)


@pytest.mark.parametrize('num_hidden_layers_phi', [0, 1, 2])
@pytest.mark.parametrize('num_hidden_layers_rho', [0, 1, 2])
@pytest.mark.parametrize('max_num_nodes_in_dataset', [14, 50])
def test_deep_sets_aggregation(num_hidden_layers_phi, num_hidden_layers_rho,
                               max_num_nodes_in_dataset):
    x = torch.randn(14, 9)
    index = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    out_channels = 16
    deep_sets_aggr = DeepSetsAggregation(
        in_channels=9, out_channels=out_channels,
        num_hidden_layers_phi=num_hidden_layers_phi,
        num_hidden_layers_rho=num_hidden_layers_rho, hidden_dim=32,
        max_num_nodes_in_dataset=max_num_nodes_in_dataset)
    deep_sets_aggr.reset_parameters()

    assert str(deep_sets_aggr) == (
        f'{DeepSetsAggregation.__name__}({max_num_nodes_in_dataset}, 9, '
        f'{out_channels}, {num_hidden_layers_phi}, {num_hidden_layers_rho}, '
        '32)')

    assert deep_sets_aggr(x, index).size() == (2, out_channels)
