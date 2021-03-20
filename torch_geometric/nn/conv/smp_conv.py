from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList
from torch_geometric.nn.conv import MessagePassing


class Linear(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, towers: int = 1,
                 bias: bool = True):
        super(Linear, self).__init__()

        assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.towers = towers
        self.bias = bias

        self.lins = [
            torch.nn.Linear(in_channels // towers, out_channels // towers,
                            bias=bias) for _ in range(towers)
        ]

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        size = self.in_channels // self.towers
        x = [lin(y) for y, lin in zip(x.split(size, dim=-1), self.lins)]
        return torch.cat(x, dim=-1) if len(x) > 1 else x[0]

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'bias={self.bias})')


class SMPConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 edge_dim: Optional[int], towers: int = 1, pre_layers: int = 1,
                 post_layers: int = 1, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(SMPConv, self).__init__(**kwargs, node_dim=-3)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.towers = towers
        self.edge_dim = edge_dim

        assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.pre_lin1 = Linear(in_channels, out_channels, towers, bias=True)
        self.pre_lin2 = Linear(in_channels, out_channels, towers, bias=False)

        if edge_dim:
            self.edge_lin = torch.nn.Linear(edge_dim, out_channels, bias=False)

        self.pre_lins = ModuleList([
            Linear(out_channels, out_channels, towers, bias=True)
            for _ in range(pre_layers - 1)
        ])

        self.post_lin1 = Linear(in_channels, out_channels, towers, bias=True)
        self.post_lin2 = Linear(out_channels, out_channels, towers, bias=False)

        self.post_lins = ModuleList([
            Linear(out_channels, out_channels, towers, bias=True)
            for _ in range(post_layers - 1)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        self.pre_lin1.reset_parameters()
        self.pre_lin2.reset_parameters()
        if self.edge_dim is not None:
            self.edge_lin.reset_parameters()
        for lin in self.pre_lins:
            lin.reset_parameters()
        self.post_lin1.reset_parameters()
        self.post_lin2.reset_parameters()
        for lin in self.post_lins:
            lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None,
                batch: OptTensor = None):
        assert x.dim() == 3 and x.size(-1) == self.in_channels

        x1, x2 = self.pre_lin1(x), self.pre_lin2(x)

        # propagate_type: (x1: Tensor, x2: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x1=x1, x2=x2, edge_attr=edge_attr)

        out = self.post_lin1(x) + self.post_lin2(out)

        for i, lin in self.post_lins:
            out = lin(out.relu_())

        return out

    def message(self, x1_i: Tensor, x2_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        out = x1_i + x2_j
        if edge_attr is not None:
            out += self.edge_lin(edge_attr).unsqueeze(1)

        for i, lin in self.pre_lins:
            out = lin(out.relu_())

        return out

    def local_context(self, x: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        ptr = torch.ops.torch_sparse.ind2ptr(batch, int(batch[-1]) + 1)
        count = ptr[1:] - ptr[:-1]

        coloring = torch.arange(batch.numel())
        coloring -= ptr[:-1].repeat_interleave(count)

        out1 = x.new_zeros((x.size(0), count.max()))
        out1.scatter_(dim=1, index=coloring.view(-1, 1), value=1.)

        out2 = x.new_zeros((*out1.size(), x.size(-1)))
        out2.scatter_(
            dim=1,
            index=coloring.view(-1, 1, 1).expand(-1, -1, x.size(-1)),
            src=x.unsqueeze(1),
        )

        return torch.cat([out1.unsqueeze(-1), out2], dim=-1), None

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')
