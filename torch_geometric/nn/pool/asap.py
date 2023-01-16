import copy
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_sparse import (
    SparseTensor,
    fill_diag,
    index_select,
    matmul,
    remove_diag,
)
from torch_sparse import t as transpose

from torch_geometric.nn import LEConv
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import add_remaining_self_loops, scatter, softmax


class ASAPooling(torch.nn.Module):
    r"""The Adaptive Structure Aware Pooling operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical
    Graph Representations" <https://arxiv.org/abs/1911.07979>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`. (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            using intra-cluster properties.
            Especially helpful for graphs with higher degree of neighborhood
            (one of :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv` or
            any GNN which supports the :obj:`edge_weight` parameter).
            (default: :obj:`None`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        add_self_loops (bool, optional): If set to :obj:`True`, will add self
            loops to the new graph connectivity. (default: :obj:`False`)
        **kwargs (optional): Additional parameters for initializing the
            graph neural network layer.

    Returns:
        A tuple of tensors containing

            - **x** (*Tensor*): The pooled node embeddings.
            - **edge_index** (*Tensor*): The coarsened graph connectivity.
            - **edge_weight** (*Tensor*): The edge weights corresponding to the
              coarsened graph connectivity.
            - **batch** (*Tensor*): The pooled :obj:`batch` vector.
            - **perm** (*Tensor*): The top-:math:`k` node indices of nodes
              which are kept after pooling.
    """
    def __init__(self, in_channels: int, ratio: Union[float, int] = 0.5,
                 GNN: Optional[Callable] = None, dropout: float = 0.0,
                 negative_slope: float = 0.2, add_self_loops: bool = False,
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.GNN = GNN
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, in_channels)
        self.att = Linear(2 * in_channels, 1)
        self.gnn_score = LEConv(self.in_channels, 1)
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels,
                                         **kwargs)
        else:
            self.gnn_intra_cluster = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att.reset_parameters()
        self.gnn_score.reset_parameters()
        if self.gnn_intra_cluster is not None:
            self.gnn_intra_cluster.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:
        """"""
        N = x.size(0)

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1., num_nodes=N)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        x_pool = x
        if self.gnn_intra_cluster is not None:
            x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index,
                                            edge_weight=edge_weight)

        x_pool_j = x_pool[edge_index[0]]
        x_q = scatter(x_pool_j, edge_index[1], dim=0, reduce='max')
        x_q = self.lin(x_q)[edge_index[1]]

        score = self.att(torch.cat([x_q, x_pool_j], dim=-1)).view(-1)
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[1], num_nodes=N)

        # Sample attention coefficients stochastically.
        score = F.dropout(score, p=self.dropout, training=self.training)

        v_j = x[edge_index[0]] * score.view(-1, 1)
        x = scatter(v_j, edge_index[1], dim=0, reduce='sum')

        # Cluster selection.
        fitness = self.gnn_score(x, edge_index).sigmoid().view(-1)
        perm = topk(fitness, self.ratio, batch)
        x = x[perm] * fitness[perm].view(-1, 1)
        batch = batch[perm]

        # Graph coarsening.
        row, col = edge_index[0], edge_index[1]
        A = SparseTensor(row=row, col=col, value=edge_weight,
                         sparse_sizes=(N, N))
        S = SparseTensor(row=row, col=col, value=score, sparse_sizes=(N, N))

        S = index_select(S, 1, perm)
        A = matmul(matmul(transpose(S), A), S)

        if self.add_self_loops:
            A = fill_diag(A, 1.)
        else:
            A = remove_diag(A)

        row, col, edge_weight = A.coo()
        edge_index = torch.stack([row, col], dim=0)

        return x, edge_index, edge_weight, batch, perm

    @torch.jit.unused
    def jittable(self) -> 'ASAPooling':
        out = copy.deepcopy(self)
        out.gnn_score = out.gnn_score.jittable()
        if out.gnn_intra_cluster is not None:
            out.gnn_intra_cluster = out.gnn_intra_cluster.jittable()
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'ratio={self.ratio})')
