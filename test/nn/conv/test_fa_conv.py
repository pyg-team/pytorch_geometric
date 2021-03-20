import torch
from torch.nn import ReLU
from torch_geometric.nn import FAConv


def test_fa_conv():
    faconv = FAConv(2, 5, raw_in_channels=4, add_self_loops=False, phi=ReLU())
    x_0 = torch.rand(3, 4)
    x = torch.rand(3, 2)
    edge_index = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 1]]).T

    x = faconv(x, edge_index, x_0=x_0, return_coeff=False)
    assert x.shape == (3, 5)
    assert faconv.__repr__(
    ) == 'FAConv(2, 5 , 4, phi=ReLU(), dropout=0.0, add_self_loops=False)'

    x = torch.rand(3, 2)
    x, (edge_index_ret, edge_weight) = faconv(x, edge_index, x_0=x_0,
                                              return_coeff=True)
    assert edge_index_ret.tolist() == edge_index.tolist()
    assert torch.logical_and(edge_weight < 1, edge_weight > -1).all()
