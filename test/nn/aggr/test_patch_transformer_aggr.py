import torch

from torch_geometric.nn import PatchTransformerAggregation


def test_patch_transformer_aggregation():
    index = torch.tensor([0, 0, 1, 1, 1, 2])
    x = torch.ones(index.shape[0], 16)
    max_elements = 3
    max_edge = max_elements
    patch_size = max_elements

    aggr = PatchTransformerAggregation(
        channels=16,
        hidden_dim=64,
        out_dim=128,
        max_edge=max_edge,
        num_encoder_blocks=1,
        heads=1,
        patch_size=patch_size,
        layer_norm=False,
        dropout=0.2,
    )
    assert str(aggr) == ('PatchTransformerAggregation(16,'
                         'heads=1,layer_norm=False,dropout=0.2)')
    out = aggr(x=x, index=index, ptr=None, dim_size=None,
               max_num_elements=max_elements)
    assert out.size() == (3, 128)


if __name__ == "__main__":
    test_patch_transformer_aggregation()
