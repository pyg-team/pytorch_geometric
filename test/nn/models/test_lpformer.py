import torch

import torch_geometric.typing
from torch_geometric.nn import LPFormer
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_undirected


def test_lpformer():
    model = LPFormer(16, 32, num_gnn_layers=2, 
                     num_transformer_layers=1)
    assert str(model) == 'LPFormer(16, 32, num_gnn_layers=2, num_transformer_layers=1)'

    num_nodes = 20
    x = torch.randn(num_nodes, 16)
    edges = torch.randint(0, num_nodes-1, (2, 110))
    edge_index, test_edges = edges[:, :100], edges[:, 100:]
    edge_index = to_undirected(edge_index)

    ppr_matrix = model.calc_sparse_ppr(edge_index, num_nodes, eps=1e-4)
    
    assert ppr_matrix.is_sparse
    assert ppr_matrix.size() == (num_nodes, num_nodes)
    assert ppr_matrix.sum().item() > 0

    # Test with dense edge_index
    out = model(test_edges, x, edge_index, ppr_matrix)
    assert out.size() == (10,)

    # Test with sparse edge_index
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, 
                                           sparse_sizes=(num_nodes, num_nodes))
        out2 = model(test_edges, x, adj, ppr_matrix)
        assert out2.size() == (10,)
