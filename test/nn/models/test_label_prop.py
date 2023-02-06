from itertools import product

import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.nn.models import LabelPropagation

num_layers = [0, 5]
alphas = [0, 0.5]


@pytest.mark.parametrize('num_layer,alpha', product(num_layers, alphas))
def test_label_prop(num_layer, alpha):
    y = torch.tensor([1, 0, 0, 2, 1, 1])
    edge_index = torch.tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    mask = torch.randint(0, 2, (6, )).to(torch.bool)

    model = LabelPropagation(num_layer, alpha)
    assert str(model) == f'LabelPropagation(num_layers={num_layer}, ' \
                         f'alpha={alpha})'

    out1 = model(y, edge_index)
    assert out1.size() == (6, 3)

    # Test with SparseTensor:
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(6, 6))
    out2 = model(y, adj)
    assert torch.allclose(out1, out2)

    # Test with mask:
    assert model(y, edge_index, mask).size() == (6, 3)
    assert model(y, adj, mask).size() == (6, 3)
