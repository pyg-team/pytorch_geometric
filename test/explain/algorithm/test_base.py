import pytest
import torch
import torch.nn.functional as F

from torch_geometric.explain.algorithm.base import ExplainerAlgorithm
from torch_geometric.nn import GCNConv


class GCN_one_layer(torch.nn.Module):
    def __init__(self):
        super(GCN_one_layer, self).__init__()
        self.conv1 = GCNConv(3, 1)

    def forward(self, x, edge_index):
        return self.conv1(x, edge_index)


class GCN_two_layers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)


@pytest.mark.parametrize('model', [GCN_one_layer(), GCN_two_layers()])
def test_single_subgraph(model):
    algo = ExplainerAlgorithm()

    x = torch.randn(7, 3)
    edge_index = torch.tensor([
        [0, 2, 1, 2, 2, 4, 3, 4, 4, 6, 5, 6],
        [2, 0, 2, 1, 4, 2, 4, 3, 6, 4, 6, 5],
    ])
    """
    Undirected Graph:
    0 -- 2 -- 4 -- 6
         |    |    |
         1    3    5
    """

    x, edge_index, mapping, node_idx, edge_mask, _ = algo.subgraph(
        model, 2, x, edge_index)

    # If GCN_one_layer, then the subgraph should be (0, 1, 2, 4)
    # If GCN_two_layers, then the subgraph should be (0, 1, 2, 3, 4, 6)

    if type(model) == GCN_one_layer:
        assert x.size() == (4, 3)
        assert set(node_idx.tolist()) == set([0, 1, 2, 4])
        assert edge_mask.sum() == 6
    else:
        assert x.size() == (6, 3)
        assert set(node_idx.tolist()) == set([0, 1, 2, 3, 4, 6])
        assert edge_mask.sum() == 10

    assert mapping.size() == (1, )
    assert edge_mask.size() == (12, )


def test_multiple_subgraphs():
    model = GCN_one_layer()
    algo = ExplainerAlgorithm()

    x = torch.randn(7, 3)
    edge_index = torch.tensor([
        [0, 2, 1, 2, 2, 4, 3, 4, 4, 6, 5, 6],
        [2, 0, 2, 1, 4, 2, 4, 3, 6, 4, 6, 5],
    ])

    x, edge_index, mapping, node_idx, edge_mask, _ = algo.subgraph(
        model, torch.tensor([4, 6]), x, edge_index)

    # The subgraph should be (2, 3, 4, 5, 6)
    assert x.size() == (5, 3)
    assert set(node_idx.tolist()) == set([2, 3, 4, 5, 6])
    assert edge_mask.sum() == 8
    assert mapping.size() == (2, )
    assert edge_mask.size() == (12, )
