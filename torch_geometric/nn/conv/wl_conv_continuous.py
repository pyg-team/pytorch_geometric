from typing import Tuple, Union

from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import degree

# kill these
import torch
from torch_geometric.data import Data


class WLConvContinuous(MessagePassing):
    r""" Weisfeiler Lehman operator for graphs with continuous attributes.
    Wasserstein Weisfeiler-Lehman Graph Kernels"
    <https://proceedings.neurips.cc/paper/2019/file/73fed7fd472e502d8908794430511f4d-Paper.pdf>`_ paper
    Refinement is done though a degree-scaled mean aggregation:

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{2}\big(\mathbf{x}_i + \frac{1}{\textrm{deg}(i)}
        \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot \mathbf{x}_j \big)

    where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
    target node :obj:`i` (default: :obj:`1`)

    Args:
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(aggr='mean', **kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=size)
        x_r = x[1]
        deg = degree(edge_index[1]).view(-1, 1)
        out = 1/2 * (x_r + (1/deg * out))

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return matmul(adj_t, x[0], reduce=self.aggr)

if __name__ == "__main__":
    model = WLConvContinuous()
    edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    z = model(data.x, data.edge_index)
    assert (z == torch.tensor([[-0.5], [0.0], [0.5]], dtype=torch.float)).all()
