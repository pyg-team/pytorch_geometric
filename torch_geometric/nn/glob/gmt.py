from typing import List, Callable

import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.typing import OptTensor

import math


class MAB(torch.nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, conv=None, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads

        self.fc_q = torch.nn.Linear(dim_Q, dim_V)

        if conv is None:
            self.layer_k = torch.nn.Linear(dim_K, dim_V)
            self.layer_v = torch.nn.Linear(dim_K, dim_V)
        else:
            self.layer_k = conv(dim_K, dim_V)
            self.layer_v = conv(dim_K, dim_V)

        if ln:
            self.ln0 = torch.nn.LayerNorm(dim_V)
            self.ln1 = torch.nn.LayerNorm(dim_V)

        self.fc_o = torch.nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, graph=None, attention_mask=None):
        Q = self.fc_q(Q)

        if graph is not None:
            (x, edge_index, batch) = graph
            K, V = self.layer_k(x, edge_index), self.layer_v(x, edge_index)
            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)
        else:
            K, V = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask for _ in range(self.num_heads)], 0
            )
            attention_score = Q_.bmm(K_.transpose(1, 2))
            attention_score = attention_score / math.sqrt(self.dim_V)
            A = torch.softmax(attention_mask + attention_score, 1)
        else:
            A = torch.softmax(
                Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 1
            )

        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        out = out if getattr(self, 'ln0', None) is None else self.ln0(out)
        out = out + torch.nn.functional.relu(self.fc_o(out))
        out = out if getattr(self, 'ln1', None) is None else self.ln1(out)

        return out


class SAB(torch.nn.Module):
    def __init__(self, dim_in, dim_out, num_heads,
                 conv=None, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, conv=conv, ln=ln)

    def forward(self, X, graph=None, attention_mask=None):
        return self.mab(X, X, graph, attention_mask)


class PMA(torch.nn.Module):
    def __init__(self, dim, num_heads,
                 num_seeds, conv=None, ln=False):
        super(PMA, self).__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, dim))
        torch.nn.init.xavier_uniform_(self.S)

        self.mab = MAB(dim, dim, dim, num_heads, conv=conv, ln=ln)

    def forward(self, X, graph=None, attention_mask=None):
        return self.mab(
            self.S.repeat(X.size(0), 1, 1), X, graph, attention_mask
        )


class GraphMultisetTransformer(torch.nn.Module):
    r"""The global pooling operator, Graph Multiset Transformer,
    from the `"Accurate Learning of Graph Representations
    with Graph Multiset Pooling" <https://arxiv.org/abs/2102.11533>`_ paper.

    In short, nodes of the entire graph are clustered via attention-based
    pooling operations (GMPool_G or GMPool_I), and self-attention (SelfAtt)
    could be used to calculate the inter-relationships among nodes.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        conv (torch.nn.Module, optional): A graph neural network layer
            for calculating hidden representations of nodes for GMPool_G
            (one of :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv` or
            :class:`torch_geometric.nn.conv.GATConv`). (default:
            :class:`torch_geometric.nn.conv.GCNConv`)
        num_nodes (int, optional): The number of average
            or maximum nodes. (default: :obj:`300`)
        pooling_ratio (float, optional): Graph pooling ratio
            for each pooling. (default: :obj:`0.25`)
        pool_sequences ([str], optional): A sequence of pooling layers
            consisting of Graph Multiset Transformer (one of
            :obj:`['GMPool_I']`,
            :obj:`['GMPool_G']`,
            :obj:`['GMPool_G', 'GMPool_I']`,
            :obj:`['GMPool_G', 'SelfAtt', 'GMPool_I']` or
            :obj:`['GMPool_G', 'SelfAtt', 'SelfAtt', 'GMPool_I']`).
            (default: :obj:`['GMPool_G', 'SelfAtt', 'GMPool_I']`)
        num_heads (int, optional): Number of heads in attention.
            (default: :obj:`4`)
        ln (bool, optional): If set to :obj:`True`, will make use of
            layer normalization. (default: :obj:`False`)
    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, conv: Callable = None,
                 num_nodes: int = 300, pooling_ratio: float = 0.25,
                 pool_sequences: List[str] = [
                     'GMPool_G', 'SelfAtt', 'GMPool_I'],
                 num_heads: int = 4, ln: bool = False):
        super(GraphMultisetTransformer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.pooling_ratio = pooling_ratio
        self.pool_sequences = pool_sequences
        self.num_heads = num_heads
        self.ln = ln

        self.conv = GCNConv if conv is None else conv

        self.pools = torch.nn.ModuleList()
        n_out_nodes = math.ceil(self.num_nodes * self.pooling_ratio)
        self.pools.append(torch.nn.Linear(in_channels, hidden_channels))
        for index, pool_type in enumerate(pool_sequences):
            if pool_type not in ['GMPool_G', 'GMPool_I', 'SelfAtt']:
                raise ValueError(
                    "The element in 'pools'",
                    "should be one of 'GMPool_G', 'GMPool_I', 'SelfAtt'"
                )
            if (index == len(pool_sequences) - 1):
                n_out_nodes = 1

            if pool_type == 'GMPool_G':
                self.pools.append(
                    PMA(hidden_channels, num_heads,
                        n_out_nodes, conv=self.conv, ln=ln)
                )
                n_out_nodes = math.ceil(n_out_nodes * self.pooling_ratio)
            elif pool_type == 'GMPool_I':
                self.pools.append(
                    PMA(hidden_channels, num_heads,
                        n_out_nodes, conv=None, ln=ln)
                )
                n_out_nodes = math.ceil(n_out_nodes * self.pooling_ratio)
            elif pool_type == 'SelfAtt':
                self.pools.append(
                    SAB(hidden_channels, hidden_channels,
                        num_heads, conv=None, ln=ln)
                )
        self.pools.append(torch.nn.Linear(hidden_channels, out_channels))

    def forward(self, x: Tensor, batch: Tensor,
                edge_index: OptTensor = None) -> Tensor:
        """"""
        x = self.pools[0](x)
        batch_x, mask = to_dense_batch(x, batch)

        extended_attention_mask = mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        for index, pool_type in enumerate(self.pool_sequences):
            if pool_type == 'GMPool_G':
                batch_x = self.pools[index + 1](
                    batch_x,
                    graph=(x, edge_index, batch),
                    attention_mask=extended_attention_mask
                )
            else:
                batch_x = self.pools[index + 1](
                    batch_x,
                    graph=None,
                    attention_mask=extended_attention_mask
                )
            extended_attention_mask = None
        batch_x = self.pools[-1](batch_x)

        out = batch_x.squeeze(1)
        return out

    def __repr__(self):
        return '{}(pool={})'.format(self.__class__.__name__,
                                    self.pools)
