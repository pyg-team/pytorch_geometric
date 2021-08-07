import torch
from torch_geometric.nn import PDNConv


def test_pdn_conv():
    x = torch.randn(4, 16)
    edge_weight = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = PDNConv(16, 32, 8, 128)
    assert conv.__repr__() == 'PDNConv(16, 32, 8, 128)'
    out = conv(x, edge_index, edge_weight)
    assert out.size() == (4, 32)


def test_pdn_conv_with_sparse_node_input_feature():
    x = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0], [0, 1]]),
                                values=torch.tensor([1., 1.]),
                                size=torch.Size([4, 16]))
    edge_weight = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = PDNConv(16, 32, 8, 128)

    out = conv(x, edge_index, edge_weight)
    assert out.size() == (4, 32)


def test_pdn_conv_with_sparse_edge_input_feature():
    x = torch.randn(4, 16)
    edge_weight = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0],
                                                                [0, 1]]),
                                values=torch.tensor([1., 1.]),
                                size=torch.Size([6, 8]))

    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = PDNConv(16, 32, 8, 128)

    out = conv(x, edge_index, edge_weight)
    assert out.size() == (4, 32)


def test_pdn_conv_with_sparse_node_and_edge_input_feature():
    x = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0], [0, 1]]),
                                values=torch.tensor([1., 1.]),
                                size=torch.Size([4, 16]))

    edge_weight = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0],
                                                                [0, 1]]),
                                values=torch.tensor([1., 1.]),
                                size=torch.Size([6, 8]))

    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])

    conv = PDNConv(16, 32, 8, 128, bias=False)

    out = conv(x, edge_index, edge_weight)
    assert out.size() == (4, 32)
