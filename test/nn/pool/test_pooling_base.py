import torch

from torch_geometric.nn import MaxAggregation
from torch_geometric.nn.pool.base import Pooling, PoolingOutput
from torch_geometric.nn.pool.connect import Connect
from torch_geometric.nn.pool.select import Select
from torch_geometric.utils import scatter


class DummySelect(Select):
    def forward(self, x, edge_index, edge_attr, batch):
        # Pool into a single node for each graph.
        if batch is None:
            return edge_index.new_zeros(x.size(0), dtype=torch.long), 1
        return batch, int(batch.max()) + 1


class DummyConnect(Connect):
    def forward(self, x, edge_index, edge_attr, batch):
        # Return empty graph connection:
        if edge_attr is not None:
            edge_attr = edge_attr.new_empty((0, ) + edge_attr.size()[1:])
        return edge_index.new_empty(2, 0), edge_attr


def test_pooling():
    pool = Pooling(DummySelect(), MaxAggregation(), DummyConnect())
    pool.reset_parameters()
    assert str(pool) == ('Pooling(\n'
                         '  select=DummySelect(),\n'
                         '  reduce=MaxAggregation(),\n'
                         '  connect=DummyConnect(),\n'
                         ')')

    x = torch.randn(10, 8)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty(0, 4)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    out = pool(x, edge_index)
    assert isinstance(out, PoolingOutput)
    assert torch.allclose(out.x, x.max(dim=0, keepdim=True)[0])
    assert out.edge_index.size() == (2, 0)
    assert out.edge_attr is None
    assert out.batch is None

    out = pool(x, edge_index, edge_attr, batch)
    assert isinstance(out, PoolingOutput)
    assert torch.allclose(out.x, scatter(x, batch, reduce='max'))
    assert out.edge_index.size() == (2, 0)
    assert out.edge_attr.size() == (0, 4)
    assert out.batch.tolist() == [0, 1]
