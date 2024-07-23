import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import NeuralFingerprint
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor


@pytest.mark.parametrize('batch', [None, torch.tensor([0, 1, 1])])
def test_neural_fingerprint(batch):
    x = torch.randn(3, 7)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = NeuralFingerprint(7, 16, out_channels=5, num_layers=4)
    assert str(model) == 'NeuralFingerprint(7, 5, num_layers=4)'
    model.reset_parameters()

    out = model(x, edge_index, batch)
    assert out.size() == (1, 5) if batch is None else (2, 5)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(3, 3))
        assert torch.allclose(model(x, adj.t(), batch), out)

    if is_full_test():
        jit = torch.jit.export(model)
        assert torch.allclose(jit(x, edge_index, batch), out)
