import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter, to_torch_csc_tensor


@functional_transform('sign')
class SIGN(BaseTransform):
    r"""The Scalable Inception Graph Neural Network module (SIGN) from the
    `"SIGN: Scalable Inception Graph Neural Networks"
    <https://arxiv.org/abs/2004.11198>`_ paper (functional name: :obj:`sign`),
    which precomputes the fixed representations

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
    def __init__(self, K: int):
        self.K = K

    def __call__(self, data: Data) -> Data:
        assert data.edge_index is not None
        row, col = data.edge_index
        N = data.num_nodes

        edge_weight = data.edge_weight
        if edge_weight is None:
            edge_weight = torch.ones(data.num_edges, device=row.device)

        deg = scatter(edge_weight, col, dim_size=N, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        adj = to_torch_csc_tensor(data.edge_index, edge_weight, size=(N, N))
        adj_t = adj.t()

        assert data.x is not None
        xs = [data.x]
        for i in range(1, self.K + 1):
            xs += [adj_t @ xs[-1]]
            data[f'x{i}'] = xs[-1]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K})'
