from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn import radius_graph
from torch_geometric.testing import onlyFullTest, withPackage


@onlyFullTest
@withPackage('torch_cluster')
def test_radius_graph_jit():
    class Net(torch.nn.Module):
        def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
            return radius_graph(x, r=2.5, batch=batch, loop=False)

    x = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float)
    batch = torch.tensor([0, 0, 0, 0])

    model = Net()
    jit = torch.jit.script(model)
    assert model(x, batch).size() == jit(x, batch).size()
