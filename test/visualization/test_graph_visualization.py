import os.path as osp

import pytest
import torch

from torch_geometric.testing import onlyGraphviz, withPackage
from torch_geometric.visualization import visualize_graph


@onlyGraphviz
@pytest.mark.parametrize('backend', [None, 'graphviz'])
def test_visualize_graph_via_graphviz(tmp_path, backend):
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ])
    edge_weight = (torch.rand(edge_index.size(1)) > 0.5).float()

    path = osp.join(tmp_path, 'graph.pdf')
    visualize_graph(edge_index, edge_weight, path, backend)
    assert osp.exists(path)


@withPackage('networkx', 'matplotlib')
@pytest.mark.parametrize('backend', [None, 'networkx'])
def test_visualize_graph_via_networkx(tmp_path, backend):
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ])
    edge_weight = (torch.rand(edge_index.size(1)) > 0.5).float()

    path = osp.join(tmp_path, 'graph.pdf')
    visualize_graph(edge_index, edge_weight, path, backend)
    assert osp.exists(path)
