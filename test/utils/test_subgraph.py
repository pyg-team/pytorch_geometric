import torch
from torch_geometric.utils import subgraph


def test_subgraph():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5],
    ])
    edge_attr = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    idx = torch.tensor([3, 4, 5], dtype=torch.long)
    mask = torch.tensor([0, 0, 0, 1, 1, 1, 0], dtype=torch.uint8)
    indices = [3, 4, 5]

    for subset in [idx, mask, indices]:
        out = subgraph(subset, edge_index, edge_attr)
        assert out[0].tolist() == [[3, 4, 4, 5], [4, 3, 5, 4]]
        assert out[1].tolist() == [7, 8, 9, 10]

        out = subgraph(subset, edge_index, edge_attr, relabel_nodes=True)
        assert out[0].tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
        assert out[1].tolist() == [7, 8, 9, 10]
