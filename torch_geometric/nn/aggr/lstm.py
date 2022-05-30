from typing import Optional

from torch import Tensor
from torch.nn import LSTM

from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import to_dense_batch


class LSTMAggregation(Aggregation):
    r"""Performs LSTM-style aggregation in which the elements to aggregate are
    interpreted as a sequence.

    .. warn::
        :class:`LSTMAggregation` is not permutation-invariant.

    .. note::
        :class:`LSTMAggregation` requires sorted indices.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        **kwargs (optional): Additional arguments of :class:`torch.nn.LSTM`.
    """
    requires_sorted_index = True

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm = LSTM(in_channels, out_channels, batch_first=True, **kwargs)

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, x: Tensor, index: Optional[Tensor] = None, *,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        if index is None:  # TODO
            raise NotImplementedError(f"'{self.__class__.__name__}' with "
                                      f"'ptr' not yet supported")

        if x.dim() != 2:
            raise ValueError(f"'{self.__class__.__name__}' requires "
                             f"two-dimensional inputs (got '{x.dim()}')")

        if dim not in [-2, 0]:
            raise ValueError(f"'{self.__class__.__name__}' needs to perform "
                             f"aggregation in first dimension (got '{dim}')")

        x, _ = to_dense_batch(x, index, batch_size=dim_size)
        return self.lstm(x)[0][:, -1]

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
