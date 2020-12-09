from torch_geometric.typing import Adj

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_scatter import scatter_add


class WLConv(torch.nn.Module):
    def __init__(self):
        super(WLConv, self).__init__()
        self.hashmap = {}

    def reset_parameters(self):
        self.hashmap = {}

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        if x.dim() > 1:
            assert (x.sum(dim=-1) == 1).sum() == x.size(0)
            x = x.argmax(dim=-1)  # one-hot -> integer.
        assert x.dtype == torch.long

        adj_t = edge_index
        if not isinstance(adj_t, SparseTensor):
            adj_t = SparseTensor(row=edge_index[1], col=edge_index[0],
                                 sparse_sizes=(x.size(0), x.size(0)))

        out = []
        _, col, _ = adj_t.coo()
        deg = adj_t.storage.rowcount().tolist()
        for node, neighbors in zip(x.tolist(), x[col].split(deg)):
            idx = hash(tuple([node] + neighbors.sort()[0].tolist()))
            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)
            out.append(self.hashmap[idx])

        return torch.tensor(out, device=x.device)

    def histogram(self, x: Tensor, batch: Tensor = None,
                  norm: bool = False) -> Tensor:

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_colors = len(self.hashmap)
        batch_size = int(batch.max()) + 1

        index = batch * num_colors + x
        out = scatter_add(torch.ones_like(index), index, dim=0,
                          dim_size=num_colors * batch_size)
        out = out.view(batch_size, num_colors)

        if norm:
            out = out.to(torch.float)
            out /= out.norm(dim=-1, keepdim=True)

        return out

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
