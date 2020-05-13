import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_scatter import scatter_add, scatter_max
from torch_geometric.nn import LEConv
from torch_geometric.utils import (add_remaining_self_loops,
                                   remove_self_loops, softmax)
from torch_geometric.nn.pool.topk_pool import topk
from torch_sparse import coalesce
from torch_sparse import transpose
from torch_sparse import spspmm


def StAS(index_A, value_A, index_S, value_S, device, N, kN):
    r"""StAS: a function which returns new edge weights for
        the pooled graph using the formula S^{T}AS"""

    index_A, value_A = coalesce(index_A, value_A, m=N, n=N)
    index_S, value_S = coalesce(index_S, value_S, m=N, n=kN)
    index_B, value_B = spspmm(index_A, value_A, index_S, value_S, N, N, kN)

    index_St, value_St = transpose(index_S, value_S, N, kN)
    index_B, value_B = coalesce(index_B, value_B, m=N, n=kN)
    index_E, value_E = spspmm(index_St, value_St, index_B, value_B, kN, N, kN)

    return index_E, value_E


def graph_connectivity(device, perm, edge_index, edge_weight,
                       score, ratio, batch, N):
    r"""graph_connectivity: is a function which internally
        calls StAS func to maintain graph connectivity"""

    kN = perm.size(0)
    perm2 = perm.view(-1, 1)

    # mask contains bool mask of edges which originate from
    #       perm (selected) nodes
    mask = (edge_index[0] == perm2).sum(0, dtype=torch.bool)

    # ---create the S---
    S0 = edge_index[1][mask].view(1, -1)
    S1 = edge_index[0][mask].view(1, -1)
    index_S = torch.cat([S0, S1], dim=0)
    value_S = score[mask].detach().squeeze()

    # relabel for pooling ie: make S [N x kN]
    n_idx = torch.zeros(N, dtype=torch.long)
    n_idx[perm] = torch.arange(perm.size(0))
    index_S[1] = n_idx[index_S[1]]

    # ---create A---
    index_A = edge_index.clone()
    if edge_weight is None:
        value_A = value_S.new_ones(edge_index[0].size(0))
    else:
        value_A = edge_weight.clone()

    fill_value = 1
    index_E, value_E = StAS(index_A, value_A, index_S, value_S,
                            device, N, kN)
    index_E, value_E = remove_self_loops(edge_index=index_E,
                                         edge_attr=value_E)
    index_E, value_E = add_remaining_self_loops(edge_index=index_E,
                                                edge_weight=value_E,
                                                fill_value=fill_value,
                                                num_nodes=kN)

    return index_E, value_E


class ASAP_Pooling(torch.nn.Module):
    r"""The Adaptive Structure Aware Pooling operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical
    Graph Representations" <https://arxiv.org/pdf/1911.07979.pdf>`_ paper

    For Pseudocode please refer to Appendix Sec. I of the paper above.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float, optional): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            using intra-cluster properties more. More helpful for graphs with
            higher degree of neighborhood. (one of
            :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv` or
            any GNN which supports edge_weights parameter).
            (default: :obj:`None`)
        dropout_att (float, optional): Dropout to be applied on the
            attention scores found via Master2Token attention framework
            for obtaining cluster representation
            (default: :obj:`0`)
        negative_slope (float, optional): Negative Slope for LeakyRelu
            activation
            (default: :obj:`0.2`)
        self_loops (bool, optional): Adds self_loops while returning
            edge index and weights
            (default: :obj:`True`)
        **kwargs (optional): Additional parameters for initializing the
            graph neural network layer.
    """

    def __init__(self, in_channels, ratio=0.5, GNN=None, dropout_att=0,
                 negative_slope=0.2, self_loops=True, **kwargs):
        super(ASAP_Pooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout_att = dropout_att
        self.lin_q = Linear(in_channels, in_channels)
        self.gat_att = Linear(2*in_channels, 1)
        self.GNN = GNN
        self.self_loops = self_loops
        # gnn_score: uses LEConv to find cluster fitness scores
        self.gnn_score = LEConv(self.in_channels, 1)
        # gnn_intra_cluster: uses GCN to account for
        #       intra cluster properties, e.g., edge-weights
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels,
                                         **kwargs)
        self.reset_parameters()

    def reset_parameters(self):

        self.lin_q.reset_parameters()
        self.gat_att.reset_parameters()
        self.gnn_score.reset_parameters()
        if self.GNN is not None:
            self.gnn_intra_cluster.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # NxF
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # Add Self Loops
        fill_value = 1
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        edge_index, edge_weight = add_remaining_self_loops(
                                    edge_index=edge_index,
                                    edge_weight=edge_weight,
                                    fill_value=fill_value,
                                    num_nodes=num_nodes.sum())

        N = x.size(0)  # total num of nodes in batch

        # ExF
        if self.GNN is not None:
            x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index,
                                            edge_weight=edge_weight)
        else:
            x_pool = x

        x_pool_j = x_pool[edge_index[1]]
        x_j = x[edge_index[1]]

        # ---Master query formation---
        # NxF
        X_q, _ = scatter_max(x_pool_j, edge_index[0], dim=0)
        # NxF
        M_q = self.lin_q(X_q)
        # ExF
        M_q = M_q[edge_index[0].tolist()]

        score = self.gat_att(torch.cat((M_q, x_pool_j), dim=-1))
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[0], num_nodes=num_nodes.sum())

        # Sample attention coefficients stochastically.
        score = F.dropout(score, p=self.dropout_att, training=self.training)
        # ExF
        v_j = x_j * score.view(-1, 1)
        # ---Aggregation---
        # NxF
        out = scatter_add(v_j, edge_index[0], dim=0)

        # ---Cluster Selection---
        fitness = torch.sigmoid(self.gnn_score(x=out,
                                edge_index=edge_index)).view(-1)  # Nx1
        perm = topk(x=fitness, ratio=self.ratio, batch=batch)
        x = out[perm] * fitness[perm].view(-1, 1)

        # ---Maintaining Graph Connectivity---
        batch = batch[perm]
        edge_index, edge_weight = graph_connectivity(
            device=x.device,
            perm=perm,
            edge_index=edge_index,
            edge_weight=edge_weight,
            score=score,
            ratio=self.ratio,
            batch=batch,
            N=N)
        if self.self_loops is False:
            edge_index, edge_weight = remove_self_loops(
                                        edge_index=edge_index,
                                        edge_attr=edge_weight)

        return x, edge_index, edge_weight, batch, perm

    def __repr__(self):
        return '{}({}, {}, ratio={}, {}, {}, {})'.format(
            self.__class__.__name__,
            self.gnn_intra_cluster.__class__.__name__,
            self.in_channels, self.ratio, self.dropout_att,
            self.negative_slope, self.self_loops)
