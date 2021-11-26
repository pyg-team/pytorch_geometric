import torch
from torch_scatter import scatter_min, scatter_max, scatter_mean, scatter_std

from torch_geometric.utils import degree
from torch_geometric.transforms import BaseTransform


class LocalDegreeProfile(BaseTransform):
    r"""Appends the Local Degree Profile (LDP) from the `"A Simple yet
    Effective Baseline for Non-attribute Graph Classification"
    <https://arxiv.org/abs/1811.03508>`_ paper

    .. math::
        \mathbf{x}_i = \mathbf{x}_i \, \Vert \, (\deg(i), \min(DN(i)),
        \max(DN(i)), \textrm{mean}(DN(i)), \textrm{std}(DN(i)))

    to the node features, where :math:`DN(i) = \{ \deg(j) \mid j \in
    \mathcal{N}(i) \}`.
    """
    def __call__(self, data):
        row, col = data.edge_index
        N = data.num_nodes

        deg = degree(row, N, dtype=torch.float)
        deg_col = deg[col]

        min_deg, _ = scatter_min(deg_col, row, dim_size=N)
        min_deg[min_deg > 10000] = 0
        max_deg, _ = scatter_max(deg_col, row, dim_size=N)
        max_deg[max_deg < -10000] = 0
        mean_deg = scatter_mean(deg_col, row, dim_size=N)
        std_deg = scatter_std(deg_col, row, dim_size=N)

        x = torch.stack([deg, min_deg, max_deg, mean_deg, std_deg], dim=1)

        if data.x is not None:
            data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([data.x, x], dim=-1)
        else:
            data.x = x

        return data
