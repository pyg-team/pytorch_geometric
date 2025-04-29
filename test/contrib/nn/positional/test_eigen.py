import torch

from torch_geometric.contrib.nn.positional.eigen import EigEncoder
from torch_geometric.data import Data


def test_eig_encoder_shape():
    num_nodes = 5
    num_eigvec = 3
    hidden_dim = 16

    data = Data(
        x=torch.randn(num_nodes, 4),
        eig_pos_emb=torch.randn(num_nodes, num_eigvec)
    )

    encoder = EigEncoder(num_eigvec=num_eigvec, hidden_dim=hidden_dim)
    out = encoder(data)

    assert out.shape == (num_nodes, hidden_dim)


def test_eig_encoder_consistency():
    data = Data(
        x=torch.randn(4, 2),
        eig_pos_emb=torch.randn(4, 2)  # 2 eigenvectors
    )

    encoder = EigEncoder(num_eigvec=2, hidden_dim=8)

    # Same encoder should give same results
    out1 = encoder(data)
    out2 = encoder(data)
    assert torch.allclose(out1, out2)

    # Different encoders may differ but maintain valid embeddings
    encoder2 = EigEncoder(num_eigvec=2, hidden_dim=8)
    out3 = encoder2(data)

    # Check shapes match even if values differ
    assert out1.shape == out3.shape
    # Verify embeddings differ (due to random initialization)
    assert not torch.allclose(out1, out3)
