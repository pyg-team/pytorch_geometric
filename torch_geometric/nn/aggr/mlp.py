from typing import Optional

from torch import Tensor

from torch_geometric.nn.aggr import Aggregation


class MLPAggregation(Aggregation):
    r"""Performs MLP aggregation in which the elements to aggregate are
    flattened into a single vectorial representation, and are then processed by
    a Multi-Layer Perceptron (MLP), as described in the `"Graph Neural Networks
    with Adaptive Readouts" <https://arxiv.org/abs/2211.04952>`_ paper.

    .. warning::
        :class:`MLPAggregation` is not a permutation-invariant operator.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        max_num_elements (int): The maximum number of elements to aggregate per
            group.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.models.MLP`.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 max_num_elements: int, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_num_elements = max_num_elements

        from torch_geometric.nn import MLP
        self.mlp = MLP(in_channels=in_channels * max_num_elements,
                       out_channels=out_channels, **kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                   max_num_elements=self.max_num_elements)
        return self.mlp(x.view(-1, x.size(1) * x.size(2)))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, '
                f'max_num_elements={self.max_num_elements})')
