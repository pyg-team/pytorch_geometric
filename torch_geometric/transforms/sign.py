import torch

from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter


@functional_transform('sign')
class SIGN(BaseTransform):
    r"""The Scalable Inception Graph Neural Network module (SIGN) from the
    `"SIGN: Scalable Inception Graph Neural Networks"
    <https://arxiv.org/abs/2004.11198>`_ paper (functional name: :obj:`sign`),
    which precomputes the fixed representations.

    .. math::
        \mathbf{X}^{(i)} = {\left( \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \right)}^i \mathbf{X}

    for :math:`i \in \{ 1, \ldots, K \}` and saves them in
    :obj:`data.x1`, :obj:`data.x2`, ...

    .. note::

        Since intermediate node representations are pre-computed, this operator
        is able to scale well to large graphs via classic mini-batching.
        For an example of using SIGN, see `examples/sign.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        sign.py>`_.

    Args:
        K (int): The number of hops/layer.
    """
    def __init__(self, K: int) -> None:
        self.K = K

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        edge_index = data.edge_index
        row, col = data.edge_index
        num_nodes = data.num_nodes

        edge_weight = data.edge_weight
        if edge_weight is None:
            edge_weight = torch.ones(data.num_edges, device=edge_index.device)

        deg = scatter(edge_weight, col, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        edge_index = EdgeIndex(edge_index, sparse_size=(num_nodes, num_nodes))
        edge_index, perm = edge_index.sort_by('col')
        edge_weight = edge_weight[perm]

        assert data.x is not None
        xs = [data.x]
        for i in range(1, self.K + 1):
            xs.append(edge_index.matmul(xs[-1], edge_weight, transpose=True))
            data[f'x{i}'] = xs[-1]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K})'
