from typing import Union, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear


class LEConv(MessagePassing):
    r"""The local extremum graph neural network operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph
    Representations" <https://arxiv.org/abs/1911.07979>`_ paper, which finds
    the importance of nodes with respect to their neighbors using the
    difference operator:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \cdot \mathbf{\Theta}_1 +
        \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        (\mathbf{\Theta}_2 \mathbf{x}_i - \mathbf{\Theta}_3 \mathbf{x}_j)

    where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
    target node :obj:`i` (default: :obj:`1`)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will
            not learn an additive bias. (default: :obj:`True`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin1 = Linear(in_channels[0], out_channels, bias=bias)
        self.lin2 = Linear(in_channels[1], out_channels, bias=False)
        self.lin3 = Linear(in_channels[1], out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x = (x, x)

        a = self.lin1(x[0])
        b = self.lin2(x[1])

        # propagate_type: (a: Tensor, b: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, a=a, b=b, edge_weight=edge_weight,
                             size=None)

        return out + self.lin3(x[1])

    def message(self, a_j: Tensor, b_i: Tensor,
                edge_weight: OptTensor) -> Tensor:
        out = a_j - b_i
        return out if edge_weight is None else out * edge_weight.view(-1, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
