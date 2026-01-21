import pytest
import torch

import torch_geometric
from torch_geometric.contrib.nn.models.geognn import GeoGNN


@pytest.mark.parametrize('embedding_size', [16, 32, 64])
@pytest.mark.parametrize('layer_num', [2, 3, 4])
@pytest.mark.parametrize('dropout', [0.0, 0.2, 0.4])
def test_geo_gnn(embedding_size, layer_num, dropout):
    model = GeoGNN(embedding_size, layer_num, dropout)

    ab_graph = torch_geometric.data.Data(
        x=torch.randn(10, embedding_size),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                 [1, 2, 3, 4, 5, 6, 7, 8, 9,
                                  0]]), batch=torch.tensor([0] * 10))
    ba_graph = torch_geometric.data.Data(
        x=torch.randn(10, embedding_size),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]),
        edge_attr=torch.randn(10,
                              embedding_size), batch=torch.tensor([0] * 10))

    ab_graph_repr, ba_graph_repr, node_repr, edge_repr = model(
        ab_graph, ba_graph)

    assert ab_graph_repr.size() == (1, embedding_size)
    assert ba_graph_repr.size() == (1, embedding_size)
    assert node_repr.size() == (10, embedding_size)
    assert edge_repr.size() == (10, embedding_size)
