from typing import Optional

import torch
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing


class SMPConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 edge_dim: Optional[int], towers: int = 1, **kwargs):
        # pass

        # improved: bool = False, cached: bool = False,
        # add_self_loops: bool = True, normalize: bool = True,
        # bias: bool = True, **kwargs):

        # self.lin_l = EntryWiseX(out_channels, out_channels, towers)

        # self.lin2 = Linear(out_channels, out_channels)
        # self.lin3 = Linear(out_channels, out_channels)
        # self.lin4 = Linear(out_channels, out_channels)

        kwargs.setdefault('aggr', 'add')
        super(SMPConv, self).__init__(**kwargs, node_dim=-3)

        print(in_channels, towers)
        assert in_channels % towers == 0
        assert out_channels % towers == 0

        # self.towers = towers
        self.F_in = F_in = in_channels // towers
        self.F_out = F_out = out_channels // towers

        self.lins = [Linear(F_in, F_out, bias=False) for _ in range(towers)]
        self.lins_l = [Linear(F_out, F_out, bias=False) for _ in range(towers)]
        self.lins_r = [Linear(F_out, F_out, bias=False) for _ in range(towers)]

        self.in_channels = in_channels

    def forward(self, x, edge_index, edge_attr, batch, mask=None):
        if x.dim() < 3:  # Crete feature matrx for each node.
            x, mask = self.gen_local_context(x, batch)
        assert x.size(-1) == self.in_channels

        x = [lin(u) for u, lin in zip(x.split(self.F_in, -1), self.lins)]
        x = torch.cat(x, dim=-1) if len(x) > 1 else x[0]

        x_l = [lin(u) for u, lin in zip(x.split(self.F_out, -1), self.lins_l)]
        x_l = torch.cat(x_l, dim=-1) if len(x_l) > 1 else x_l[0]

        x_r = [lin(u) for u, lin in zip(x.split(self.F_out, -1), self.lins_l)]
        x_r = torch.cat(x_r, dim=-1) if len(x_r) > 1 else x_r[0]

        out = self.propagate(edge_index, x=x, x_l=x_l, x_r=x_r,
                             edge_attr=edge_attr)

        print(x.size())

    def message(self, x_j, x_l_i, x_r_j, edge_attr):

        pass

    def gen_local_context(self, x, batch):
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
