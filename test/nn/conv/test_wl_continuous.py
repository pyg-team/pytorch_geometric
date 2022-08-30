import torch

from torch_geometric.nn import WLConvContinuous


def test_wl_conv():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    conv = WLConvContinuous()
    assert str(conv) == 'WLConvContinuous()'

    out = conv(x, edge_index)
    assert (out == torch.tensor([[-0.5], [0.0], [0.5]],
                                dtype=torch.float)).all()
