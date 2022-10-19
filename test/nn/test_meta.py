from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_scatter import scatter_mean

from torch_geometric.nn import MetaLayer
from torch_geometric.testing import is_full_test

count = 0


def test_meta_layer():
    assert str(MetaLayer()) == ('MetaLayer(\n'
                                '  edge_model=None,\n'
                                '  node_model=None,\n'
                                '  global_model=None\n'
                                ')')

    def dummy_model(*args):
        global count
        count += 1
        return None

    x = torch.randn(20, 10)
    edge_index = torch.randint(0, high=10, size=(2, 20), dtype=torch.long)

    for edge_model in (dummy_model, None):
        for node_model in (dummy_model, None):
            for global_model in (dummy_model, None):
                model = MetaLayer(edge_model, node_model, global_model)
                out = model(x, edge_index)
                assert isinstance(out, tuple) and len(out) == 3

    assert count == 12


def test_meta_layer_example():
    class EdgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_mlp = Seq(Lin(2 * 10 + 5 + 20, 5), ReLU(), Lin(5, 5))

        def forward(
            self,
            src: Tensor,
            dest: Tensor,
            edge_attr: Optional[Tensor],
            u: Optional[Tensor],
            batch: Optional[Tensor],
        ) -> Tensor:
            assert edge_attr is not None
            assert u is not None
            assert batch is not None
            out = torch.cat([src, dest, edge_attr, u[batch]], 1)
            return self.edge_mlp(out)

    class NodeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.node_mlp_1 = Seq(Lin(15, 10), ReLU(), Lin(10, 10))
            self.node_mlp_2 = Seq(Lin(2 * 10 + 20, 10), ReLU(), Lin(10, 10))

        def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Optional[Tensor],
            u: Optional[Tensor],
            batch: Optional[Tensor],
        ) -> Tensor:
            assert edge_attr is not None
            assert u is not None
            assert batch is not None
            row = edge_index[0]
            col = edge_index[1]
            out = torch.cat([x[row], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
            out = torch.cat([x, out, u[batch]], dim=1)
            return self.node_mlp_2(out)

    class GlobalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.global_mlp = Seq(Lin(20 + 10, 20), ReLU(), Lin(20, 20))

        def forward(
            self,
            x: Tensor,
            edge_index: Tensor,
            edge_attr: Optional[Tensor],
            u: Optional[Tensor],
            batch: Optional[Tensor],
        ) -> Tensor:
            assert u is not None
            assert batch is not None
            out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
            return self.global_mlp(out)

    op = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())

    x = torch.randn(20, 10)
    edge_attr = torch.randn(40, 5)
    u = torch.randn(2, 20)
    batch = torch.tensor([0] * 10 + [1] * 10)
    edge_index = torch.randint(0, high=10, size=(2, 20), dtype=torch.long)
    edge_index = torch.cat([edge_index, 10 + edge_index], dim=1)

    x_out, edge_attr_out, u_out = op(x, edge_index, edge_attr, u, batch)
    assert x_out.size() == (20, 10)
    assert edge_attr_out.size() == (40, 5)
    assert u_out.size() == (2, 20)

    if is_full_test():
        jit = torch.jit.script(op)

        x_out, edge_attr_out, u_out = jit(x, edge_index, edge_attr, u, batch)
        assert x_out.size() == (20, 10)
        assert edge_attr_out.size() == (40, 5)
        assert u_out.size() == (2, 20)
