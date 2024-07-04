import torch

from torch_geometric.nn import PatchTransformerAggregation


def test_mlp_aggregation():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    edge_feats = torch.ones(index.shape[0], 16)
    edge_ts = torch.ones(index.shape[0], 1)

    aggr = PatchTransformerAggregation(
        channels=16,
        edge_channels=16,
        time_dim=16,
        hidden_dim=64,
        out_dim=128,
        num_edge_per_node=64,
        num_encoder_blocks=1,
        heads=1,
        patch_size=8,
        layer_norm=False,
        dropout=0.2,
    )
    assert str(aggr) == ('PatchTransformerAggregation(16,'
                         ' edge_channels=16,'
                         'time_dim=16,'
                         'heads=1,'
                         'layer_norm=False,'
                         'dropout=0.2)')
    out = aggr(x, index, edge_feats=edge_feats, edge_ts=edge_ts, batch_size=6,
               ptr=None, dim_size=None)
    assert out.size() == (6, 128)
