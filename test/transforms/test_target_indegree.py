import torch
from torch_geometric.transforms import TargetIndegree
from torch_geometric.data import Data


def test_target_indegree():
    assert TargetIndegree().__repr__() == 'TargetIndegree(max_value=None)'

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.Tensor([1, 1, 1, 1])

    data = Data(edge_index=edge_index)
    data = TargetIndegree()(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == edge_index.tolist()
    assert data.edge_attr.tolist() == [[1], [0.5], [0.5], [1]]

    data = Data(edge_index=edge_index, edge_attr=edge_attr)
    data = TargetIndegree(max_value=4)(data)
    assert len(data) == 2
    assert data.edge_index.tolist() == edge_index.tolist()
    assert data.edge_attr.tolist() == [[1, 0.5], [1, 0.25], [1, 0.25],
                                       [1, 0.5]]
