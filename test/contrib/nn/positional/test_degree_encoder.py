import torch

from torch_geometric.contrib.nn.positional.degree import DegreeEncoder
from torch_geometric.data import Data


def test_degree_encoder_shape():
    hidden_dim = 16
    num_nodes = 5
    max_degree = 3  # Maximum degree in test data

    data = Data(
        x=torch.randn(num_nodes, 4),
        in_degree=torch.LongTensor([0, 1, 2, 1, 0]),
        out_degree=torch.LongTensor([1, 2, 0, 1, 0])
    )

    encoder = DegreeEncoder(
        num_in=max_degree, num_out=max_degree, hidden_dim=hidden_dim
    )
    out = encoder(data)

    assert out.shape == (num_nodes, hidden_dim)


def test_degree_encoder_values():
    hidden_dim = 8
    max_degree = 3

    data = Data(
        x=torch.randn(3, 4),
        in_degree=torch.LongTensor([0, 1, 2]),
        out_degree=torch.LongTensor([1, 1, 0])
    )

    encoder = DegreeEncoder(
        num_in=max_degree, num_out=max_degree, hidden_dim=hidden_dim
    )
    out = encoder(data)

    # Check same degrees map to same embeddings for both in and out degrees
    in_emb = encoder.in_embed(data.in_degree[1])
    out_emb = encoder.out_embed(data.out_degree[1])
    assert torch.allclose(out[1, :], in_emb + out_emb)

    # Check different degrees map to different embeddings
    assert not torch.allclose(
        encoder.in_embed(data.in_degree[0]),
        encoder.in_embed(data.in_degree[1])
    )
