from typing import Optional

import pytest
import torch
from torch import Tensor

from torch_geometric.nn import (
    fps,
    knn,
    knn_graph,
    nearest,
    radius,
    radius_graph,
)
from torch_geometric.testing import onlyFullTest, withPackage


@onlyFullTest
@withPackage('pyg_lib')
def test_radius_graph_jit():
    class Net(torch.nn.Module):
        def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
            return radius_graph(x, r=2.5, batch=batch, loop=False)

    x = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float)
    batch = torch.tensor([0, 0, 0, 0])

    model = Net()
    jit = torch.jit.script(model)
    assert model(x, batch).size() == jit(x, batch).size()


@withPackage('pyg_lib')
def test_unsorted_batch_raises():
    x = torch.tensor([[0., 1.], [1., 1.], [10., 10.], [10., 11.]])
    batch = torch.tensor([0, 1, 0, 0])  # Not sorted.

    with pytest.raises(ValueError, match="needs to be sorted"):
        fps(x, batch)
    with pytest.raises(ValueError, match="needs to be sorted"):
        knn(x, x, 1, batch, batch)
    with pytest.raises(ValueError, match="needs to be sorted"):
        knn_graph(x, k=1, batch=batch)
    with pytest.raises(ValueError, match="needs to be sorted"):
        radius(x, x, 2.0, batch, batch)
    with pytest.raises(ValueError, match="needs to be sorted"):
        radius_graph(x, r=2.0, batch=batch)
    with pytest.raises(ValueError, match="needs to be sorted"):
        nearest(x, x, batch, batch)


@withPackage('pyg_lib')
def test_sorted_batch():
    x = torch.tensor([[0., 1.], [10., 10.], [10., 11.], [1., 1.]])
    batch = torch.tensor([0, 0, 0, 1])  # Sorted, with a single-node example.

    edge_index = radius_graph(x, r=2.0, batch=batch)
    assert sorted(edge_index.t().tolist()) == [[1, 2], [2, 1]]

    edge_index = knn_graph(x, k=1, batch=batch)
    assert sorted(edge_index.t().tolist()) == [[1, 0], [1, 2], [2, 1]]
