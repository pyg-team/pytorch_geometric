import pytest
import torch

from torch_geometric.nn.models import NodeFormer


@pytest.mark.parametrize('use_bn', [True, False])
@pytest.mark.parametrize('use_gumbel', [True, False])
@pytest.mark.parametrize('use_residual', [True, False])
@pytest.mark.parametrize('use_act', [True, False])
@pytest.mark.parametrize('use_jk', [True, False])
@pytest.mark.parametrize('use_edge_loss', [True, False])
@pytest.mark.parametrize('rb_trans', ['sigmoid', 'identity'])
@pytest.mark.parametrize('rb_order', [0, 1])
def test_nodeformer(
    use_bn,
    use_gumbel,
    use_residual,
    use_act,
    use_jk,
    use_edge_loss,
    rb_trans,
    rb_order,
):
    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    ])

    model = NodeFormer(
        in_channels=16,
        hidden_channels=128,
        out_channels=40,
        num_layers=3,
        use_bn=use_bn,
        use_gumbel=use_gumbel,
        use_residual=use_residual,
        use_act=use_act,
        use_jk=use_jk,
        use_edge_loss=use_edge_loss,
        rb_trans=rb_trans,
        rb_order=rb_order,
    )
    if use_edge_loss:
        out, link_loss = model(x, edge_index)
        assert len(link_loss) == 3
    else:
        out = model(x, edge_index)
    assert out.size() == (10, 40)
