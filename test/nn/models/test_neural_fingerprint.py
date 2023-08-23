import pytest
import torch

from torch_geometric.nn import NeuralFingerprint

def test_neural_fingerprint():
    x = torch.randn(3,7)
    edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]])
    model = NeuralFingerprint(num_features=7, fingerprint_length=5, num_layers=4)
    fingerprint = model.forward(x, edge_index)
    assert fingerprint.size() == torch.Size([5])