from torch_geometric.typing import Adj

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing


class LGConv(MessagePassing):
    r"""The neighbor aggregation operator from the `"LightGCN: Simplifying and
    Powering Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper

    .. math::
        \textbf{e}_u^{(k+1)} =
        \sum_{i \in N_u} \frac{1}{\sqrt{|N_u|} \sqrt{|N_i|}} \textbf{e}_i^{(k)}

    .. math::
        \textbf{e}_i^{(k+1)} =
        \sum_{i \in N_i} \frac{1}{\sqrt{|N_i|} \sqrt{|N_u|}} \textbf{e}_u^{(k)}

    Args:
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, **kwargs):
        super().__init__(aggr='add')

        self.reset_parameters()

    def reset_parameters(self):
        pass  # There are no layer parameters to learn.

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """"""
        num_users = max(edge_index[0]) + 1
        num_items = max(edge_index[1]) + 1
        user_item = \
            torch.zeros(num_users, num_items, device=x.device)
        user_item[edge_index[0], edge_index[1]] = 1
        user_neighbor_counts = torch.sum(user_item, axis=1)
        item_neightbor_counts = torch.sum(user_item, axis=0)
        # Compute weight for aggregation: 1 / sqrt(N_u * N_i)
        weights = user_item / torch.sqrt(
            user_neighbor_counts.repeat(num_items, 1).T
            * item_neightbor_counts.repeat(num_users, 1))
        weights = torch.nan_to_num(weights, nan=0)
        out = self.propagate(edge_index, x=(x, x),
                             norm=weights[edge_index[0], edge_index[1]])
        return out
