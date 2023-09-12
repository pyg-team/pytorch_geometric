from math import ceil, log2
from typing import Optional

import torch
from torch import Tensor
from torch.nn import GRUCell, Linear

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation


class LCMAggregation(Aggregation):
    r"""The Learnable Commutative Monoid aggregation from the
    `"Learnable Commutative Monoids for Graph Neural Networks"
    <https://arxiv.org/abs/2212.08541>`_ paper, in which the elements are
    aggregated using a binary tree reduction with
    :math:`\mathcal{O}(\log |\mathcal{V}|)` depth.

    .. note::

        :class:`LCMAggregation` requires sorted indices :obj:`index` as input.
        Specifically, if you use this aggregation as part of
        :class:`~torch_geometric.nn.conv.MessagePassing`, ensure that
        :obj:`edge_index` is sorted by destination nodes, either by manually
        sorting edge indices via :meth:`~torch_geometric.utils.sort_edge_index`
        or by calling :meth:`torch_geometric.data.Data.sort`.

    .. warning::

        :class:`LCMAggregation` is not a permutation-invariant operator.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        project (bool, optional): If set to :obj:`True`, the layer will apply a
            linear transformation followed by an activation function before
            aggregation. (default: :obj:`True`)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        project: bool = True,
    ):
        super().__init__()

        if in_channels != out_channels and not project:
            raise ValueError(f"Inputs of '{self.__class__.__name__}' must be "
                             f"projected if `in_channels != out_channels`")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.project = project

        if self.project:
            self.lin = Linear(in_channels, out_channels)
        else:
            self.lin = None

        self.gru_cell = GRUCell(out_channels, out_channels)

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        self.gru_cell.reset_parameters()

    def binary_op(self, left: Tensor, right: Tensor) -> Tensor:
        return (self.gru_cell(left, right) + self.gru_cell(right, left)) / 2.0

    @disable_dynamic_shapes(required_args=['dim_size', 'max_num_elements'])
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:

        if self.project:
            x = self.lin(x).relu()

        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                   max_num_elements=max_num_elements)
        
        x = x.permute(1, 0, 2)  # [num_neighbors, num_nodes, num_features]
        num_nodes, num_features = x.size(1), x.size(2)
        
        depth = ceil(log2(x.size(0)))
        for _ in range(depth):
            half_size = ceil(x.size(0) / 2)
            
            lr, rl = [], []
            remainder = None
            for i in range(half_size):
                if (2 * i + 1) >= x.size(0):
                    # this level of the tree has an odd number of nodes, so the 
                    # remaining, unmatched node gets moved to the next level
                    remainder = x[2 * i].unsqueeze(0)
                    break
                left, right = x[2 * i], x[2 * i + 1]
                lr += [left, right]
                rl += [right, left]
            lr, rl = torch.cat(lr), torch.cat(rl)
            
            # compute `self.binary_op(left, right)` for all `(left, right)`
            # pairs in the current level of the tree, in parallel. Results in 
            # `O(log V)` total `GRUCell` forward passes, instead of `O(V)`.
            out = self.gru_cell(lr, rl)
            out = out.view(-1, 2, num_nodes, num_features)
            out = out.mean(1)
            if remainder is not None:
                out = torch.cat([out, remainder], 0)

            x = out.view(half_size, num_nodes, num_features)
        
        assert x.size(0) == 1
        return x.squeeze(0)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, project={self.project})')
