from typing import List, Type, Optional, Tuple

import math

import torch
from torch import Tensor
from torch.nn import Linear, LayerNorm

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


class MAB(torch.nn.Module):
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.fc_q = Linear(dim_Q, dim_V)

        if Conv is None:
            self.layer_k = Linear(dim_K, dim_V)
            self.layer_v = Linear(dim_K, dim_V)
        else:
            self.layer_k = Conv(dim_K, dim_V)
            self.layer_v = Conv(dim_K, dim_V)

        if layer_norm:
            self.ln0 = LayerNorm(dim_V)
            self.ln1 = LayerNorm(dim_V)

        self.fc_o = Linear(dim_V, dim_V)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.layer_k.reset_parameters()
        self.layer_v.reset_parameters()
        if self.layer_norm:
            self.ln0.reset_parameters()
            self.ln1.reset_parameters()
        self.fc_o.reset_parameters()
        pass

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        Q = self.fc_q(Q)

        if graph is not None:
            x, edge_index, batch = graph
            K, V = self.layer_k(x, edge_index), self.layer_v(x, edge_index)
            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)
        else:
            K, V = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2))
            attention_score = attention_score / math.sqrt(self.dim_V)
            A = torch.softmax(mask + attention_score, 1)
        else:
            A = torch.softmax(
                Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 1)

        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        if self.layer_norm:
            out = self.ln0(out)

        out = out + self.fc_o(out).relu()

        if self.layer_norm:
            out = self.ln1(out)

        return out


class SAB(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.mab = MAB(in_channels, in_channels, out_channels, num_heads,
                       Conv=Conv, layer_norm=layer_norm)

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(x, x, graph, mask)


class PMA(torch.nn.Module):
    def __init__(self, channels: int, num_heads: int, num_seeds: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))
        self.mab = MAB(channels, channels, channels, num_heads, Conv=Conv,
                       layer_norm=layer_norm)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(self.S.repeat(x.size(0), 1, 1), x, graph, mask)


class GraphMultisetTransformer(torch.nn.Module):
    r"""The global Graph Multiset Transformer pooling operator from the
    `"Accurate Learning of Graph Representations
    with Graph Multiset Pooling" <https://arxiv.org/abs/2102.11533>`_ paper.

    The Graph Multiset Transformer clusters nodes of the entire graph via
    attention-based pooling operations (:obj:`"GMPool_G"` or
    :obj:`"GMPool_I"`).
    In addition, self-attention (:obj:`"SelfAtt"`) can be used to calculate
    the inter-relationships among nodes.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        conv (Type, optional): A graph neural network layer
            for calculating hidden representations of nodes for
            :obj:`"GMPool_G"` (one of
            :class:`~torch_geometric.nn.conv.GCNConv`,
            :class:`~torch_geometric.nn.conv.GraphConv` or
            :class:`~torch_geometric.nn.conv.GATConv`).
            (default: :class:`~torch_geometric.nn.conv.GCNConv`)
        num_nodes (int, optional): The number of average
            or maximum nodes. (default: :obj:`300`)
        pooling_ratio (float, optional): Graph pooling ratio
            for each pooling. (default: :obj:`0.25`)
        pool_sequences ([str], optional): A sequence of pooling layers
            consisting of Graph Multiset Transformer submodules (one of
            :obj:`["GMPool_I"]`,
            :obj:`["GMPool_G"]`,
            :obj:`["GMPool_G", "GMPool_I"]`,
            :obj:`["GMPool_G", "SelfAtt", "GMPool_I"]` or
            :obj:`["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"]`).
            (default: :obj:`["GMPool_G", "SelfAtt", "GMPool_I"]`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`4`)
        layer_norm (bool, optional): If set to :obj:`True`, will make use of
            layer normalization. (default: :obj:`False`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        Conv: Optional[Type] = None,
        num_nodes: int = 300,
        pooling_ratio: float = 0.25,
        pool_sequences: List[str] = ['GMPool_G', 'SelfAtt', 'GMPool_I'],
        num_heads: int = 4,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.Conv = Conv or GCNConv
        self.num_nodes = num_nodes
        self.pooling_ratio = pooling_ratio
        self.pool_sequences = pool_sequences
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

        self.pools = torch.nn.ModuleList()
        num_out_nodes = math.ceil(num_nodes * pooling_ratio)
        for i, pool_type in enumerate(pool_sequences):
            if pool_type not in ['GMPool_G', 'GMPool_I', 'SelfAtt']:
                raise ValueError("Elements in 'pool_sequences' should be one "
                                 "of 'GMPool_G', 'GMPool_I', or 'SelfAtt'")

            if i == len(pool_sequences) - 1:
                num_out_nodes = 1

            if pool_type == 'GMPool_G':
                self.pools.append(
                    PMA(hidden_channels, num_heads, num_out_nodes,
                        Conv=self.Conv, layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

            elif pool_type == 'GMPool_I':
                self.pools.append(
                    PMA(hidden_channels, num_heads, num_out_nodes, Conv=None,
                        layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

            elif pool_type == 'SelfAtt':
                self.pools.append(
                    SAB(hidden_channels, hidden_channels, num_heads, Conv=None,
                        layer_norm=layer_norm))

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()

    def forward(self, x: Tensor, batch: Tensor,
                edge_index: Optional[Tensor] = None) -> Tensor:
        """"""
        x = self.lin1(x)
        batch_x, mask = to_dense_batch(x, batch)
        mask = (~mask).unsqueeze(1).to(dtype=x.dtype) * -1e9

        for i, (name, pool) in enumerate(zip(self.pool_sequences, self.pools)):
            graph = (x, edge_index, batch) if name == 'GMPool_G' else None
            batch_x = pool(batch_x, graph, mask)
            mask = None

        return self.lin2(batch_x.squeeze(1))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, pool_sequences={self.pool_sequences})')
