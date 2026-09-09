import pytest
import torch

from torch_geometric.nn import LPFormer
from torch_geometric.testing import withPackage
from torch_geometric.utils import to_undirected


@withPackage('numba')  # For ppr calculation
def test_lpformer():
    model = LPFormer(16, 32, num_gnn_layers=2, num_transformer_layers=1)
    assert str(
        model
    ) == 'LPFormer(16, 32, num_gnn_layers=2, num_transformer_layers=1)'

    num_nodes = 20
    x = torch.randn(num_nodes, 16)
    edges = torch.randint(0, num_nodes - 1, (2, 110))
    edge_index, test_edges = edges[:, :100], edges[:, 100:]
    edge_index = to_undirected(edge_index)

    ppr_matrix = model.calc_sparse_ppr(edge_index, num_nodes, eps=1e-4)

    assert ppr_matrix.is_sparse
    assert ppr_matrix.size() == (num_nodes, num_nodes)
    assert ppr_matrix.sum().item() > 0

    # Test with dense edge_index
    out = model(test_edges, x, edge_index, ppr_matrix)
    assert out.size() == (10, )

    # Test with sparse edge_index
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)),
                                  [num_nodes, num_nodes])
    out2 = model(test_edges, x, adj, ppr_matrix)
    assert out2.size() == (10, )


@pytest.mark.parametrize('num_heads', [1, 2])
@pytest.mark.parametrize('num_transformer_layers', [1, 2, 3])
@withPackage('numba')  # For ppr calculation
def test_lpformer_attention_dimensions(num_transformer_layers, num_heads):
    torch.manual_seed(12345)
    model = LPFormer(16, 32, num_gnn_layers=2,
                     num_transformer_layers=num_transformer_layers,
                     num_heads=num_heads, gnn_dropout=0.0,
                     transformer_dropout=0.0).eval()

    num_nodes = 20
    x = torch.randn(num_nodes, 16)
    edges = torch.randint(0, num_nodes - 1, (2, 110))
    edge_index, test_edges = edges[:, :100], edges[:, 100:]
    edge_index = to_undirected(edge_index)

    ppr_matrix = model.calc_sparse_ppr(edge_index, num_nodes, eps=1e-4)

    with torch.no_grad():
        out = model(test_edges, x, edge_index, ppr_matrix)
    assert out.size() == (10, )
