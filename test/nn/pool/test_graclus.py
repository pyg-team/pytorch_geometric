import torch

from torch_geometric.nn import graclus
from torch_geometric.testing import withPackage


@withPackage('pyg_lib')
def test_graclus():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    assert graclus(edge_index).tolist() == [0, 0]
