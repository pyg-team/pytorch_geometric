from typing import Optional

from torch import Tensor
from torch.nn import GRU

from torch_geometric.nn.aggr import Aggregation


class GRUAggregation(Aggregation):
    r"""Performs GRU aggregation in which the elements to aggregate are
    interpreted as a sequence, as described in the `"Graph Neural Networks
    with Adaptive Readouts" <https://arxiv.org/abs/2211.04952>`_ paper.

    .. warning::
        :class:`GRUAggregation` is not a permutation-invariant operator.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        **kwargs (optional): Additional arguments of :class:`torch.nn.GRU`.
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gru = GRU(in_channels, out_channels, batch_first=True, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.gru.reset_parameters()

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim)
        return self.gru(x)[0][:, -1]

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
