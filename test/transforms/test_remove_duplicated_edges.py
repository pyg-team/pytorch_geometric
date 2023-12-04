import torch

from torch_geometric.data import Data
from torch_geometric.transforms import RemoveDuplicatedEdges


def test_remove_duplicated_edges():
    edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1],
                               [0, 0, 1, 1, 0, 0, 1, 1]])
    edge_weight = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=2)

    transform = RemoveDuplicatedEdges()
    assert str(transform) == 'RemoveDuplicatedEdges()'

    out = transform(data)
    assert len(out) == 3
    assert out.num_nodes == 2
    assert out.edge_index.tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert out.edge_weight.tolist() == [3, 7, 11, 15]
