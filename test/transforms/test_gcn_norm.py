import torch

from torch_geometric.data import Data
from torch_geometric.transforms import GCNNorm


def test_gcn_norm():
    assert GCNNorm().__repr__() == 'GCNNorm()'

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.Tensor([1, 1, 1, 1])

    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    data = GCNNorm(add_self_loops=False)(data)
    assert data.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert torch.allclose(data.edge_weight,
                          torch.tensor([0.70711, 0.70711, 0.70711, 0.70711]))

    data = Data(edge_index=edge_index, edge_weight=edge_attr, num_nodes=3)
    data = GCNNorm(add_self_loops=False)(data)
    assert data.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert torch.allclose(data.edge_weight,
                          torch.tensor([0.70711, 0.70711, 0.70711, 0.70711]))

    data = GCNNorm(add_self_loops=True)(data)
    assert data.edge_index.tolist() == [[0, 1, 1, 2, 0, 1, 2],
                                        [1, 0, 2, 1, 0, 1, 2]]
    assert torch.allclose(
        data.edge_weight,
        torch.tensor(
            [0.34831, 0.34831, 0.34831, 0.34831, 0.58579, 0.41421, 0.58579]))
