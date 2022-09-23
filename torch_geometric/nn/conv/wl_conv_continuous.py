from typing import Union

from torch import Tensor
from torch_scatter import scatter_add

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, OptTensor, Size


class WLConvContinuous(MessagePassing):
    r"""The Weisfeiler Lehman operator from the `"Wasserstein
    Weisfeiler-Lehman Graph Kernels" <https://arxiv.org/abs/1906.01277>`_
    paper. Refinement is done though a degree-scaled mean aggregation and
    works on nodes with continuous attributes:

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{2}\big(\mathbf{x}_i +
        \frac{1}{\textrm{deg}(i)}
        \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot \mathbf{x}_j \big)

    where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
    target node :obj:`i` (default: :obj:`1`)

    Args:
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)` or
          :math:`((|\mathcal{V_s}|, F), (|\mathcal{V_t}|, F))` if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)` or
          :math:`(|\mathcal{V}_t|, F)` if bipartite
    """
    def __init__(self, **kwargs):
        super().__init__(aggr='add', **kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Tensor,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if edge_weight is None:
            edge_weight = x[0].new_ones(edge_index.size(1))

        # propagate_type: (x: OptPairTensor, edge_weight: Tensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)

        deg = scatter_add(edge_weight, edge_index[1], dim_size=out.size(0))
        deg_inv = 1. / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        out = deg_inv.view(-1, 1) * out

        x_dst = x[1]
        if x_dst is not None:
            out = 0.5 * (x_dst + out)

        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j
