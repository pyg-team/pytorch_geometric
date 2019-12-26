import torch
import numpy as np
import pytest
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr="add")
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, **kwargs):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(
            edge_index, size=(x.size(0), x.size(0)), x=x, **kwargs
        )

    def message(self, x_j, edge_index, size, zeros=True):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        result = norm.view(-1, 1) * x_j
        if not zeros:
            return result
        else:
            return result * 0

    def update(self, aggr_out):
        return aggr_out


class BadGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(BadGCNConv, self).__init__(aggr="add")
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, **kwargs):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        return self.propagate(
            edge_index, size=(x.size(0), x.size(0)), x=x, **kwargs
        )

    def message(self, x_j, edge_index, size, zeros=True):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        result = norm.view(-1, 1) * x_j
        if not zeros:
            return result
        else:
            return result * 0

    def update(self, aggr_out_haha):
        return aggr_out_haha


def test_create_gnn():
    conv = GCNConv(16, 32)
    x = torch.randn(5, 16)
    edge_index = torch.randint(5, (2, 64), dtype=torch.long)
    out = conv(x, edge_index)
    assert out.size() == (5, 32)
    assert np.allclose(out.cpu().detach().numpy(), np.zeros((5, 32)))
    out = conv(x, edge_index, zeros=False)
    assert not np.allclose(out.cpu().detach().numpy(), np.zeros((5, 32)))


def test_fail_init():
    with pytest.raises(TypeError) as e:
        BadGCNConv(16, 32)
    assert e.match(
        "Incomplete signature of update: "
        "{'aggr_out'} are missing required arguments"
    )
