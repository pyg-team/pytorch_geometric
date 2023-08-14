import torch

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('half_hop')
class HalfHop(BaseTransform):
    r"""Graph upsampling augmentation from the
    `"Half-Hop: A graph upsampling approach for slowing down
    message passing" <https://openreview.net/forum?id=lXczFIwQkv>`_ paper.
    The graph is augmented by adding artificial slow nodes between neighbors
    to slow down message propagation. (functional name: :obj:`half_hop`).

    .. note::
        `HalfHop` augmentation is not supported if `data` has
        :attr:`edge_weight` or :attr:`edge_attr`.

    Args:
        alpha (float, optional): The interpolation factor
            used to compute slow node features
            :math:`x = \alpha*x_src + (1-\alpha)*x_dst` (default: :obj:`0.5`)
        p (float, optional): The probability of half-hopping
            an edge. (default: :obj:`1.0`)

    .. code-block:: python

        import torch_geometric.transforms as T

        data = ...
        data = T.HalfHop(alpha=0.5)(data) # Apply transformation
        out = model(data.x, data.edge_index) # Feed-forward
        out = out[~data.slow_node_mask] # Get rid of slow nodes
    """
    def __init__(self, alpha: float = 0.5, p: float = 1.0):
        if alpha < 0. or alpha > 1.:
            raise ValueError(f'Interpolation factor has to be between 0 and 1 '
                             f'(got {alpha}')
        if p < 0. or p > 1.:
            raise ValueError(
                f'Ratio of half-hopped edges has to be between 0 and 1 '
                f'(got {p}')
        self.p = p
        self.alpha = alpha

    def forward(self, data: Data) -> Data:

        if data.edge_weight is not None or data.edge_attr is not None:
            raise ValueError("'HalfHop' augmentation is not supported if "
                             "'data' has 'edge_weight' or 'edge_attr'")

        x, edge_index = data.x, data.edge_index

        # isolate self loops which are not half-hopped
        self_loop_mask = edge_index[0] == edge_index[1]
        edge_index_self_loop = edge_index[:, self_loop_mask]
        edge_index = edge_index[:, ~self_loop_mask]

        # randomly sample nodes and half-hop their edges
        node_mask = torch.rand(data.num_nodes, device=x.device) < self.p
        edge_mask = node_mask[edge_index[1]]
        edge_index_to_halfhop = edge_index[:, edge_mask]
        edge_index_to_keep = edge_index[:, ~edge_mask]

        # add new slow nodes of which features are initialized
        # by linear interpolation
        num_halfhop_edges = edge_index_to_halfhop.size(1)
        slow_node_ids = torch.arange(num_halfhop_edges,
                                     device=x.device) + data.num_nodes
        x_src = x[edge_index_to_halfhop[0]]
        x_dst = x[edge_index_to_halfhop[1]]
        x_slow_node = self.alpha * x_src + (1 - self.alpha) * x_dst
        new_x = torch.cat([x, x_slow_node], dim=0)

        # add new edges between slow nodes and the original nodes
        edge_index_slow = [
            torch.stack([edge_index_to_halfhop[0], slow_node_ids]),
            torch.stack([slow_node_ids, edge_index_to_halfhop[1]]),
            torch.stack([edge_index_to_halfhop[1], slow_node_ids])
        ]
        new_edge_index = torch.cat(
            [edge_index_to_keep, edge_index_self_loop, *edge_index_slow],
            dim=1)

        # prepare a mask that distinguishes between original nodes & slow nodes
        slow_node_mask = torch.cat(
            [x.new_zeros(x.size(0)),
             x.new_ones(slow_node_ids.size(0))], dim=0).bool()

        data.x, data.edge_index = new_x, new_edge_index
        data.slow_node_mask = slow_node_mask

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha}, p={self.p})'
