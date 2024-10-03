from typing import Optional

from torch import Tensor

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.aggr import Aggregation


class MLPMixerAggregation(Aggregation):
    r"""Performs MLP Mixer.

    .. note::

        :class:`MLPAggregation` requires sorted indices :obj:`index` as input.
        Specifically, if you use this aggregation as part of
        :class:`~torch_geometric.nn.conv.MessagePassing`, ensure that
        :obj:`edge_index` is sorted by destination nodes, either by manually
        sorting edge indices via :meth:`~torch_geometric.utils.sort_edge_index`
        or by calling :meth:`torch_geometric.data.Data.sort`.

    .. warning::

        :class:`MLPAggregation` is not a permutation-invariant operator.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        max_num_elements (int): The maximum number of elements to aggregate per
            group. This is :math:`k` in the paper.
        dropout (float, optional): Dropout probability of attention weights.
            (default: :obj:`0.0`)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_num_elements: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_num_elements = max_num_elements
        self.dropout = dropout

        from torch_geometric.nn.models.graph_mixer import _MLPMixer

        # from torch_geometric.nn.models.graph_mixer import LinkEncoder
        # self.link_encoder = LinkEncoder(
        #     k=max_num_elements,
        #     in_channels=in_channels,
        #     hidden_channels=kwargs.get('hidden_channels', 128),
        #     time_channels=kwargs.get('time_channels', 128),
        #     out_channels=out_channels,
        # )
        self.mlp_mixer = _MLPMixer(
            num_tokens=max_num_elements,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp_mixer.reset_parameters()

    @disable_dynamic_shapes(required_args=['dim_size', 'max_num_elements'])
    def forward(
        self,
        x: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:
        x, mask = self.to_dense_batch(
            x,
            index,
            ptr,
            dim_size,
            dim,
            max_num_elements=self.max_num_elements,
        )

        return self.mlp_mixer(x.view(-1,
                                     x.size(1) * x.size(2)), index, dim_size)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, '
                f'max_num_elements={self.max_num_elements})')
