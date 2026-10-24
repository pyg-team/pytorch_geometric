import pytest
import torch

from torch_geometric.contrib.nn.positional.svd import SVDEncoder
from torch_geometric.data import Data


def test_svd_encoder_shape():
    num_nodes, r, hidden_dim = 5, 3, 16
    data = Data(svd_pos_emb=torch.randn(num_nodes, 2 * r))

    encoder = SVDEncoder(r=r, hidden_dim=hidden_dim)
    out = encoder(data)
    assert out.shape == (num_nodes, hidden_dim)


def test_svd_encoder_invalid_shape():
    data = Data(svd_pos_emb=torch.randn(5, 7))  # Odd number of features
    encoder = SVDEncoder(r=4, hidden_dim=16)  # Expects 8 features
    with pytest.raises(ValueError):
        encoder(data)


def test_svd_encoder_seeded():
    data = Data(svd_pos_emb=torch.randn(4, 6))  # r=3

    # Same seed gives same results
    enc1 = SVDEncoder(r=3, hidden_dim=8, seed=42)
    enc2 = SVDEncoder(r=3, hidden_dim=8, seed=42)
    assert torch.allclose(enc1(data), enc2(data))

    # Different seeds give different results
    enc3 = SVDEncoder(r=3, hidden_dim=8, seed=43)
    assert not torch.allclose(enc1(data), enc3(data))
