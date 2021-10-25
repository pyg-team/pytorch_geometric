import torch
from torch_sparse import SparseTensor
from torch_geometric.nn import HEATConv


def test_heat_conv():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    row, col = edge_index
    edge_attr = torch.randn((4, 2))
    adj = SparseTensor(row=row, col=col, value=edge_attr, sparse_sizes=(4, 4))
    node_type = torch.tensor([0, 0, 1, 2])
    edge_type = torch.tensor([0, 2, 1, 2])

    conv = HEATConv(in_channels=8, out_channels=16, num_node_types=3,
                    num_edge_types=3, edge_type_emb_dim=5, edge_dim=2,
                    edge_attr_emb_dim=6, heads=2, concat=True)
    assert str(conv) == 'HEATConv(8, 16, heads=2)'
    out = conv(x, edge_index, node_type, edge_type, edge_attr)
    assert out.size() == (4, 32)
    assert torch.allclose(conv(x, adj.t(), node_type, edge_type), out)

    t = '(Tensor, Tensor, Tensor, Tensor, OptTensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x, edge_index, node_type, edge_type, edge_attr),
                          out)

    conv = HEATConv(in_channels=8, out_channels=16, num_node_types=3,
                    num_edge_types=3, edge_type_emb_dim=5, edge_dim=2,
                    edge_attr_emb_dim=6, heads=2, concat=False)
    assert str(conv) == 'HEATConv(8, 16, heads=2)'
    out = conv(x, edge_index, node_type, edge_type, edge_attr)
    assert out.size() == (4, 16)
    assert torch.allclose(conv(x, adj.t(), node_type, edge_type), out)

    t = '(Tensor, SparseTensor, Tensor, Tensor, OptTensor, Size) -> Tensor'
    jit = torch.jit.script(conv.jittable(t))
    assert torch.allclose(jit(x, adj.t(), node_type, edge_type), out)
