import pytest
import torch

from torch_geometric.contrib.nn.encoding import SubgraphGNNPE
from torch_geometric.testing import withDevice


@withDevice
def test_subgraph_gnn_pe(device):
    # Two subgraphs with 4 nodes each, the last node of each is isolated:
    edge_index = torch.tensor([
        [0, 1, 1, 2, 4, 5, 5, 6],
        [1, 0, 2, 1, 5, 4, 6, 5],
    ], device=device)

    model = SubgraphGNNPE(16).to(device)
    assert str(model) == 'SubgraphGNNPE(16, hidden_channels=4, num_layers=4)'

    out1 = model(edge_index, num_nodes=8)
    assert out1.size() == (8, 16)

    # Random features are resampled in every forward pass:
    out2 = model(edge_index, num_nodes=8)
    assert not torch.allclose(out1, out2)

    model.eval()
    assert not torch.allclose(model(edge_index, 8), model(edge_index, 8))


def test_subgraph_gnn_pe_fixed_features():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2],
    ])
    x = torch.randn(4, 2)

    model = SubgraphGNNPE(16, hidden_channels=8, num_layers=2, in_channels=2)
    model.eval()

    out = model(edge_index, 4, x=x)
    assert out.size() == (4, 16)
    assert torch.allclose(out, model(edge_index, 4, x=x))

    # Permutation equivariance for fixed input features:
    perm = torch.tensor([2, 0, 3, 1])
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel())
    out_perm = model(inv_perm[edge_index], 4, x=x[perm])
    assert torch.allclose(out_perm, out[perm], atol=1e-6)

    with pytest.raises(ValueError, match="Expected 'x' to hold features"):
        model(edge_index, 5, x=x)


def test_subgraph_gnn_pe_batch():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3, 4, 4, 5],
        [1, 0, 2, 1, 4, 3, 5, 4],
    ])
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    x = torch.randn(6, 1)

    model = SubgraphGNNPE(16, hidden_channels=8, num_layers=2,
                          norm='graph_norm')
    out = model(edge_index, 6, x=x, batch=batch, batch_size=2)
    assert out.size() == (6, 16)

    # Graph-level normalization is applied per subgraph:
    assert not torch.allclose(out, model(edge_index, 6, x=x))


def test_subgraph_gnn_pe_dtype():
    edge_index = torch.tensor([[0, 1], [1, 0]])

    model = SubgraphGNNPE(16, hidden_channels=8).half()
    assert model(edge_index, num_nodes=3).dtype == torch.float16


def test_subgraph_gnn_pe_empty_edge_index():
    edge_index = torch.empty(2, 0, dtype=torch.long)

    model = SubgraphGNNPE(16, hidden_channels=8)
    assert model(edge_index, num_nodes=5).size() == (5, 16)
