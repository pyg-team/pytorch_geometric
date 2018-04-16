import torch
from torch_geometric.nn.functional.pool import graclus_pool
from torch_geometric.datasets.dataset import Data


def test_graclus_pool():
    x = torch.Tensor([[1, 6], [2, 5], [3, 4], [4, 3], [5, 2], [6, 1]])
    edge_index = torch.LongTensor([
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        [1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4],
    ])
    batch = torch.LongTensor([0, 0, 0, 1, 1, 1])
    data = Data(x, None, edge_index, None, None, batch)

    data = graclus_pool(data)

    assert data.num_nodes == 4
