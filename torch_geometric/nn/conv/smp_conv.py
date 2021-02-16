from typing import Optional

import torch
from torch_geometric.nn.conv import MessagePassing


class SMPConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_towers: int,
                 edge_dim: Optional[int], **kwargs):
        pass

        # improved: bool = False, cached: bool = False,
        # add_self_loops: bool = True, normalize: bool = True,
        # bias: bool = True, **kwargs):

        self.lin_l = EntryWiseX(out_channels, out_channels, num_towers)

        self.lin2 = Linear(out_channels, out_channels)
        self.lin3 = Linear(out_channels, out_channels)
        self.lin4 = Linear(out_channels, out_channels)

        kwargs.setdefault('aggr', 'add')
        super(SMPConv, self).__init__(**kwargs)

        self.in_channels = in_channels

    def forward(self, x, edge_index, batch, mask=None):
        if x.dim() < 3:  # Crete feature matrx for each node.
            x, mask = self.gen_local_context(x, batch)

    def gen_local_context(x, batch):
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
