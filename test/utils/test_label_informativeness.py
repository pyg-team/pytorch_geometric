import pytest
import torch

import torch_geometric.typing
from torch_geometric.typing import SparseTensor
from torch_geometric.testing import withPackage
from torch_geometric.utils import label_informativeness


@withPackage('networkx')
def test_label_informativeness():
    edge_index = torch.tensor([
        [0, 1, 2, 2, 3, 4],
        [1, 2, 0, 3, 4, 5]
    ])
    y = torch.tensor([0, 0, 0, 0, 1, 1])
    row, col = edge_index
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor(row=row, col=col, sparse_sizes=(6, 6))

    method = 'edge'
    assert pytest.approx(
        label_informativeness(edge_index, y, method=method)
    ) == 0.251776
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert pytest.approx(
            label_informativeness(adj, y, method=method)
        ) == 0.251776

    method = 'node'
    assert pytest.approx(
        label_informativeness(edge_index, y, method=method)
    ) == 0.3381874
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        assert pytest.approx(
            label_informativeness(adj, y, method=method)
        ) == 0.3381874

    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
        [1, 2, 0, 4, 5, 3, 7, 8, 9, 6, 7]
    ])
    y = torch.tensor([0, 0, 1, 0, 0, 1, 0, 0, 0, 1])
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

    method = 'edge'
    assert pytest.approx(
        label_informativeness(edge_index, y, batch, method=method).tolist()
    ) == [0.2740172, 0.2740172, 0.2174435]

    method = 'node'
    assert pytest.approx(
        label_informativeness(edge_index, y, batch, method=method).tolist()
    ) == [0.2740173, 0.2740173, 0.2203068]
