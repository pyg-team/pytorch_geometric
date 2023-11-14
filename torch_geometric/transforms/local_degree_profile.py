import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


@functional_transform('local_degree_profile')
class LocalDegreeProfile(BaseTransform):
    r"""Appends the Local Degree Profile (LDP) from the `"A Simple yet
    Effective Baseline for Non-attribute Graph Classification"
    <https://arxiv.org/abs/1811.03508>`_ paper
    (functional name: :obj:`local_degree_profile`).

    .. math::
        \mathbf{x}_i = \mathbf{x}_i \, \Vert \, (\deg(i), \min(DN(i)),
        \max(DN(i)), \textrm{mean}(DN(i)), \textrm{std}(DN(i)))

    to the node features, where :math:`DN(i) = \{ \deg(j) \mid j \in
    \mathcal{N}(i) \}`.
    """
    def __init__(self):
        from torch_geometric.nn.aggr.fused import FusedAggregation
        self.aggr = FusedAggregation(['min', 'max', 'mean', 'std'])

    def forward(self, data: Data) -> Data:
        row, col = data.edge_index
        N = data.num_nodes

        deg = degree(row, N, dtype=torch.float).view(-1, 1)
        xs = [deg] + self.aggr(deg[col], row, dim_size=N)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x] + xs, dim=-1)
        else:
            data.x = torch.cat(xs, dim=-1)

        return data
