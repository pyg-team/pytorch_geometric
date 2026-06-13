import pytest
import torch

from torch_geometric.nn import RGATConv


@pytest.mark.parametrize('mod', [
    'additive',
    'scaled',
    'f-additive',
    'f-scaled',
])
@pytest.mark.parametrize('attention_mechanism', [
    'within-relation',
    'across-relation',
])
@pytest.mark.parametrize('attention_mode', [
    'additive-self-attention',
    'multiplicative-self-attention',
])
@pytest.mark.parametrize('concat', [True, False])
@pytest.mark.parametrize('edge_dim', [8, None])
def test_rgat_conv(mod, attention_mechanism, attention_mode, concat, edge_dim):
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_type = torch.tensor([0, 2, 1, 2])
    edge_attr = torch.randn((4, edge_dim)) if edge_dim else None

    conv1 = RGATConv(  # `num_bases` is not None:
        in_channels=8,
        out_channels=16,
        num_relations=4,
        num_bases=4,
        mod=mod,
        attention_mechanism=attention_mechanism,
        attention_mode=attention_mode,
        heads=2,
        dim=1,
        concat=concat,
        edge_dim=edge_dim,
    )

    conv2 = RGATConv(  # `num_blocks` is not `None`
        in_channels=8,
        out_channels=16,
        num_relations=4,
        num_blocks=4,
        mod=mod,
        attention_mechanism=attention_mechanism,
        attention_mode=attention_mode,
        heads=2,
        dim=1,
        concat=concat,
        edge_dim=edge_dim,
    )

    conv3 = RGATConv(  # Both `num_bases` and `num_blocks` are `None`:
        in_channels=8,
        out_channels=16,
        num_relations=4,
        mod=mod,
        attention_mechanism=attention_mechanism,
        attention_mode=attention_mode,
        heads=2,
        dim=1,
        concat=concat,
        edge_dim=edge_dim,
    )

    conv4 = RGATConv(  # `dropout > 0` and `mod` is `None`:
        in_channels=8,
        out_channels=16,
        num_relations=4,
        mod=None,
        attention_mechanism=attention_mechanism,
        attention_mode=attention_mode,
        heads=2,
        dim=1,
        concat=concat,
        edge_dim=edge_dim,
        dropout=0.5,
    )

    for conv in [conv1, conv2, conv3, conv4]:
        assert str(conv) == 'RGATConv(8, 16, heads=2)'

        out = conv(x, edge_index, edge_type, edge_attr)
        assert out.size() == (4, 16 * (2 if concat else 1))

        out, (adj, alpha) = conv(x, edge_index, edge_type, edge_attr,
                                 return_attention_weights=True)
        assert out.size() == (4, 16 * (2 if concat else 1))
        assert adj.size() == edge_index.size()
        assert alpha.size() == (4, 2)
