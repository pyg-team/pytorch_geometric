import torch

import torch_geometric.typing
from torch_geometric.nn import GeneralConv
from torch_geometric.typing import SparseTensor


def test_general_conv():
    x1 = torch.randn(4, 8)
    e1 = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])

    conv1 = GeneralConv(8, 32, 16)
    assert str(conv1) == 'GeneralConv(8, 32)'
    out1 = conv1(x1, edge_index, edge_attr=e1)
    assert out1.size() == (4, 32)
    assert torch.allclose(conv1(x1, edge_index, edge_attr=e1), out1, atol=1e-7)

    conv2 = GeneralConv(8, 32, 16, skip_linear=True)
    assert str(conv2) == 'GeneralConv(8, 32)'
    out2 = conv2(x1, edge_index, edge_attr=e1)
    assert out2.size() == (4, 32)
    assert torch.allclose(conv2(x1, edge_index, edge_attr=e1), out2, atol=1e-7)

    conv3 = GeneralConv(8, 32, 16, directed_msg=False)
    assert str(conv3) == 'GeneralConv(8, 32)'
    out3 = conv3(x1, edge_index, edge_attr=e1)
    assert out3.size() == (4, 32)
    assert torch.allclose(conv3(x1, edge_index, edge_attr=e1), out3, atol=1e-7)

    conv4 = GeneralConv(8, 32, 16, heads=3)
    assert str(conv4) == 'GeneralConv(8, 32)'
    out4 = conv4(x1, edge_index, edge_attr=e1)
    assert out4.size() == (4, 32)
    assert torch.allclose(conv4(x1, edge_index, edge_attr=e1), out4, atol=1e-7)

    conv5 = GeneralConv(8, 32, 16, attention=True)
    assert str(conv5) == 'GeneralConv(8, 32)'
    out5 = conv5(x1, edge_index, edge_attr=e1)
    assert out5.size() == (4, 32)
    assert torch.allclose(conv5(x1, edge_index, edge_attr=e1), out5, atol=1e-7)

    conv6 = GeneralConv(8, 32, 16, heads=3, attention=True)
    assert str(conv6) == 'GeneralConv(8, 32)'
    out6 = conv6(x1, edge_index, edge_attr=e1)
    assert out6.size() == (4, 32)
    assert torch.allclose(conv6(x1, edge_index, edge_attr=e1), out6, atol=1e-7)

    conv7 = GeneralConv(8, 32, 16, heads=3, attention=True,
                        attention_type='dot_product')
    assert str(conv7) == 'GeneralConv(8, 32)'
    out7 = conv7(x1, edge_index, edge_attr=e1)
    assert out7.size() == (4, 32)
    assert torch.allclose(conv7(x1, edge_index, edge_attr=e1), out7, atol=1e-7)

    conv8 = GeneralConv(8, 32, 16, l2_normalize=True)
    assert str(conv8) == 'GeneralConv(8, 32)'
    out8 = conv8(x1, edge_index, edge_attr=e1)
    assert out8.size() == (4, 32)
    assert torch.allclose(conv8(x1, edge_index, edge_attr=e1), out8, atol=1e-7)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj2 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        conv_list = [conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8]
        out_list = [out1, out2, out3, out4, out5, out6, out7, out8]
        for conv, out in zip(conv_list, out_list):
            assert torch.allclose(conv(x1, adj2.t(), edge_attr=e1), out,
                                  atol=1e-7)
