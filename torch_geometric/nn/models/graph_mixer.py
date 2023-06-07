import torch
from torch import Tensor

from torch_geometric.utils import scatter


class NodeEncoder(torch.nn.Module):
    r"""The node encoder module from the `"Do We Really Need Complicated
    Model Architectures for Temporal Networks?"
    <https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.
    :class:`NodeEncoder` captures the 1-hop temporal neighborhood information
    via mean pooling.

    .. math::
        \mathbf{x}_v^{\prime}(t_0) = \mathbf{x}_v + \textrm{mean} \left\{
        \mathbf{x}_w : w \in \mathcal{N}(v, t_0 - T, t_0) \right\}

    Args:
        time_window (int): The temporal window size :math:`T` to define the
            1-hop temporal neighborhood.
    """
    def __init__(self, time_window: int):
        super().__init__()
        self.time_window = time_window

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        seed_time: Tensor,
    ) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_time (torch.Tensor): The timestamp attached to every edge.
            seed_time (torch.Tensor): The seed time :math:`t_0` for every
                destination node.
        """
        mask = ((edge_time <= seed_time[edge_index[1]]) &
                (edge_time > seed_time[edge_index[1]] - self.time_window))

        src, dst = edge_index[:, mask]
        mean = scatter(x[src], dst, dim=0, dim_size=x.size(0), reduce='mean')
        return x + mean

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(time_window={self.time_window})'
