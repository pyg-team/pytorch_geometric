import torch

import torch_geometric.typing
from torch_geometric.nn import NeuralFingerprint
from torch_geometric.typing import SparseTensor


def test_neural_fingerprint():
    x = torch.randn(3, 7)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    model = NeuralFingerprint(num_features=7, fingerprint_length=5,
                              num_layers=4)
    jit = torch.jit.export(model)
    fingerprint = model(x, edge_index)
    assert fingerprint.size() == torch.Size([5])

    assert torch.allclose(fingerprint, jit(x, edge_index))

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index)
        assert torch.allclose(model(x, adj.t()), fingerprint)
