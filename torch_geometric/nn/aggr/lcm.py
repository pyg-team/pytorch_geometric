from math import ceil, log2
from typing import Optional

from torch import Tensor
from torch.nn import GRUCell, Linear

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation


class LCMAggregation(Aggregation):
    r"""Performs LCM aggregation in which the elements are aggregated using a
    binary tree reduction with $O(\log V)$ depth, as described in the
    `"Learnable Commutative Monoids for Graph Neural Networks"
    <https://arxiv.org/abs/2212.08541>`_ paper.

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
    def __init__(self, in_channels: int, out_channels: int,
                 project: Optional[bool] = True):
        super().__init__()
        if in_channels != out_channels and not project:
            raise ValueError("input must be projected if"
                             "`in_channels != out_channels`")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.project = project
        if project:
            self.lin = Linear(in_channels, out_channels, bias=True)

        self.gru_cell = GRUCell(out_channels, out_channels)

        self.bin_op = lambda x, y: \
            (self.gru_cell(x, y) + self.gru_cell(y, x)) / 2

    def reset_parameters(self):
        self.gru_cell.reset_parameters()
        if self.project:
            self.lin.reset_parameters()

    @disable_dynamic_shapes(required_args=['dim_size', 'max_num_elements'])
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ):
        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                   max_num_elements=max_num_elements)

        if self.project:
            x = self.lin(x).relu()
        x = x.permute(1, 0, 2)

        depth = ceil(log2(x.shape[0]))
        for _ in range(depth):
            x = [
                self.bin_op(x[2 * i], x[2 * i + 1]) if 2 * i +
                1 < len(x) else x[2 * i] for i in range(ceil(len(x) / 2))
            ]

        assert len(x) == 1
        return x[0]

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, '
                f'project={self.project})')
