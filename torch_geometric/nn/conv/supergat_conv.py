import math

from torch.jit import RecursiveScriptModule

from typing import Optional
from torch_geometric.typing import OptTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Module
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, \
    softmax, is_undirected, negative_sampling, batched_negative_sampling, \
    to_undirected, dropout_adj

from ..inits import glorot, zeros


class SuperGATConv(MessagePassing):
    r"""The self-supervised graph attentional operator from the `'How to Find
    Your Friendly Neighborhood: Graph Attention Design with Self-Supervision'
    <https://openreview.net/forum?id=Wi5KUNlqWty>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
    where the two types of attention :math:`\alpha_{i,j}^{\mathrm{MX\ or\ SD}}`
    are computed as

    .. math::
        \alpha_{i,j}^{\mathrm{MX\ or\ SD}} &=
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(
            e_{i,j}^{\mathrm{MX\ or\ SD}}
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(
            e_{i,k}^{\mathrm{MX\ or\ SD}}
        \right)\right)},

        e_{i,j}^{\mathrm{MX}} &= \mathbf{a}^{\top}
            [\mathbf{\Theta}\mathbf{x}_i \, \Vert \,
             \mathbf{\Theta}\mathbf{x}_j]
            \cdot \sigma \left(
                \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
                \mathbf{\Theta}\mathbf{x}_j
            \right)

        e_{i,j}^{\mathrm{SD}} &= \frac{
            \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
            \mathbf{\Theta}\mathbf{x}_j
        }{ \sqrt{d} }.

    The self-supervised task is a link prediction using the attention value
    as input to predict the likelihood :math:`\phi_{i,j}^{\mathrm{MX\ or\ SD}}`
    that an edge exists between nodes.

    .. math::
        \phi_{i,j}^{\mathrm{MX}} &= \sigma \left(
            \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
            \mathbf{\Theta}\mathbf{x}_j
        \right)

        \phi_{i,j}^{\mathrm{SD}} &= \sigma \left(
            \frac{
                \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
                \mathbf{\Theta}\mathbf{x}_j
            }{ \sqrt{d} }.
        \right).

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        attention_type (string, optional): Type of attention to use.
            (:obj:`'MX'`, :obj:`'SD'`). (default: :obj:`'MX'`)
        neg_sample_ratio (float, optional): The ratio of the number of sampled
            negative edges to the number of positive edges.
            (default: :obj:`0.5`)
        edge_sample_ratio (float, optional): The ratio of samples to use for
            training among the number of training edges. (default: :obj:`1.0`)
        is_undirected (bool, optional): Whether the input graph is undirected.
            If not given, will be automatically computed with the input graph
            when negative sampling is performed. (default: :obj:`False`)
        att_label_cached (bool, optional): If set to :obj:`True`, the layer
            will cache the labels of attention in the self-supervised task on
            first execution, and will use the cached version for further
            execution. This parameter should only be set to :obj:`True` in
            transductive learning scenarios. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    att_with_neg_edges: OptTensor
    att_label: OptTensor

    def __init__(self, in_channels: int,
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True,
                 attention_type: str = 'MX', neg_sample_ratio: float = 0.5,
                 edge_sample_ratio: float = 1.0, is_undirected: bool = False,
                 att_label_cached: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(SuperGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        assert attention_type in ['MX', 'SD']
        assert 0.0 < neg_sample_ratio and 0.0 < edge_sample_ratio <= 1.0
        self.attention_type = attention_type
        self.neg_sample_ratio = neg_sample_ratio
        self.edge_sample_ratio = edge_sample_ratio
        self.is_undirected = is_undirected
        self.att_label_cached = att_label_cached

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))

        if self.attention_type == 'MX':
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        elif self.attention_type == 'SD':
            self.register_parameter('att', None)
        else:
            raise ValueError

        self.att_with_neg_edges = None  # X for self-supervision
        self.att_label = None  # Y for self-supervision

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor,
                neg_edge_index: OptTensor = None,
                batch: OptTensor = None) -> Tensor:
        r"""

        Args:
            neg_edge_index (Tensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """
        N, H, C = x.size(0), self.heads, self.out_channels

        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        x = torch.matmul(x, self.weight).view(-1, H, C)

        # propagate_type: (x: Tensor)
        out = self.propagate(edge_index, x=x, size=None)

        if self.training:

            if neg_edge_index is None:
                neg_edge_index = self.negative_sampling(edge_index, N, batch)

            if self.edge_sample_ratio < 1.0:
                pos_edge_index = self.sample_pos_edges(edge_index)
            else:
                pos_edge_index = edge_index

            att_with_neg_edges = self._get_att_logits_with_neg_edges(
                x=x,
                pos_edge_index=pos_edge_index,
                neg_edge_index=neg_edge_index,
            )  # [E + neg_E, heads]

            # Attention labels for the self-supervised task
            if not self.att_label_cached or self.att_label is None:
                num_neg_edges = att_with_neg_edges.size(0)
                att_label = torch.zeros(num_neg_edges).float().to(x.device)
                att_label[:pos_edge_index.size(1)] = 1.
            else:
                att_label = self.att_label

            self.att_with_neg_edges = att_with_neg_edges
            self.att_label = att_label

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,
                size_i: Optional[int]) -> Tensor:
        alpha = self._get_attention(edge_index_i, x_i, x_j, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    @torch.jit.ignore
    def negative_sampling(self, edge_index: Tensor,
                          num_nodes: int,
                          batch: OptTensor = None) -> Tensor:
        N, E = num_nodes, edge_index.size(1)
        num_neg_samples = int(self.neg_sample_ratio
                              * self.edge_sample_ratio * E)
        if batch is None:
            if self.is_undirected or is_undirected(edge_index,
                                                   num_nodes=N):
                edge_index_for_ns = edge_index
            else:
                edge_index_for_ns = to_undirected(edge_index, num_nodes=N)

            neg_edge_index = negative_sampling(
                edge_index=edge_index_for_ns,
                num_nodes=N,
                num_neg_samples=num_neg_samples,
            )
        else:
            neg_edge_index = batched_negative_sampling(
                edge_index=edge_index,
                batch=batch,
                num_neg_samples=num_neg_samples,
            )
        return neg_edge_index

    @torch.jit.ignore
    def sample_pos_edges(self, edge_index: Tensor) -> Tensor:
        pos_edge_index, _ = dropout_adj(edge_index,
                                        p=1 - self.edge_sample_ratio,
                                        training=self.training)
        return pos_edge_index

    def _get_attention(self, edge_index_i: Tensor,
                       x_i: Tensor, x_j: Tensor, size_i: Optional[int],
                       return_logits: bool = False) -> Tensor:

        if self.attention_type == 'MX' and self.att is not None:
            logits = torch.einsum('ehf,ehf->eh', x_i, x_j)
            if return_logits:
                return logits

            alpha = torch.einsum('ehf,xhf->eh',
                                 torch.cat([x_i, x_j], dim=-1),
                                 self.att)
            alpha = torch.einsum('eh,eh->eh', alpha, torch.sigmoid(logits))

        elif self.attention_type == 'SD':
            sqrt_d = math.sqrt(self.out_channels)
            logits = torch.einsum('ehf,ehf->eh', x_i, x_j) / sqrt_d
            if return_logits:
                return logits

            alpha = logits

        else:
            raise ValueError

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        return alpha

    def _get_att_logits_with_neg_edges(self, x: Tensor,
                                       pos_edge_index: Tensor,
                                       neg_edge_index: Tensor) -> Tensor:
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        edge_index_j, edge_index_i = edge_index[0], edge_index[1]
        x_i = torch.index_select(x, 0, edge_index_i)
        x_j = torch.index_select(x, 0, edge_index_j)
        size_i = x.size(0)
        alpha = self._get_attention(edge_index_i, x_i, x_j, size_i,
                                    return_logits=True)
        return alpha

    @classmethod
    def get_attention_loss(cls, model: Module) -> Tensor:
        r"""Compute the self-supervised graph attention loss from the neural
            network, which consists of :obj:`SuperGATConv` layers.

        Args:
            model (Module): The neural network module which contains
                :obj:`SuperGATConv` layers.
        """
        loss_list = []
        for i, m in enumerate(model.modules()):
            is_supergat = isinstance(m, cls)
            is_supergat_jit = (isinstance(m, RecursiveScriptModule) and
                               m.original_name.startswith(cls.__name__))
            if is_supergat or is_supergat_jit:
                att = m.att_with_neg_edges.mean(dim=-1)
                att_loss = F.binary_cross_entropy_with_logits(att, m.att_label)
                loss_list.append(att_loss)
        return sum(loss_list)

    def __repr__(self):
        return '{}({}, {}, heads={}, type={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.heads, self.attention_type,
        )
