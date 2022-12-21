import os.path
import random
import sys

import pytest
import torch

from torch_geometric.testing import onlyGraphviz, withPackage
from torch_geometric.visualization import visualize_graph


@onlyGraphviz
@pytest.mark.parametrize('backend', [None, 'graphviz'])
def test_visualize_graph_via_graphviz(backend):
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ])
    edge_weight = (torch.rand(edge_index.size(1)) > 0.5).float()

    path = os.path.join('/', 'tmp', f'{random.randrange(sys.maxsize)}.pdf')

    assert not os.path.exists(path)
    visualize_graph(edge_index, edge_weight, path, backend)
    assert os.path.exists(path)
    os.remove(path)


@withPackage('networkx', 'matplotlib')
@pytest.mark.parametrize('backend', [None, 'networkx'])
def test_visualize_graph_via_networkx(backend):
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3],
    ])
    edge_weight = (torch.rand(edge_index.size(1)) > 0.5).float()

    path = os.path.join('/', 'tmp', f'{random.randrange(sys.maxsize)}.pdf')

    assert not os.path.exists(path)
    visualize_graph(edge_index, edge_weight, path, backend)
    assert os.path.exists(path)
    os.remove(path)
