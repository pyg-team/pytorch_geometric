import torch

from torch_geometric.nn import PatchTransformerAggregation


def test_patch_transformer_aggregation():
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    x = torch.ones(index.shape[0], 16)
    torch.ones(index.shape[0], 1)
    max_num_elements = 3
    max_num_edge_per_node = 18
    patch_size = max_num_elements

    aggr = PatchTransformerAggregation(
        channels=16,
        edge_channels=16,
        hidden_dim=64,
        out_dim=128,
        max_num_edge_per_node=max_num_edge_per_node,
        num_encoder_blocks=1,
        heads=1,
        patch_size=patch_size,
        layer_norm=False,
        dropout=0.2,
    )
    assert str(aggr) == ('PatchTransformerAggregation(16,'
                         ' edge_channels=16,'
                         'heads=1,'
                         'layer_norm=False,'
                         'dropout=0.2)')
    out = aggr(x=x, index=index, ptr=None, dim_size=None,
               max_num_elements=max_num_elements)
    assert out.size() == (3, 128)
