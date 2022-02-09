from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ReLU
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, ones, zeros
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class RGATConv(MessagePassing):
    r"""The relational graph attentional operator from the `"Relational Graph
    Attention Networks" <https://arxiv.org/abs/1904.05811>`_ paper where the
    attention logits :math:`\mathbf{E}^{(r)}_{i,j}` for each relation type
    (:math:`\mathbf{r}`) are computed with the help of both query and key
    kernels,
    *i.e.*

    .. math::
        \mathbf{q}^{(r)}_i = \mathbf{\Theta}^{(r)}\mathbf{x}_{i} \ \cdot
        \ \mathbf{Q}^{(r)}

    and

    .. math::
        \mathbf{k}^{(r)}_i = \mathbf{\Theta}^{(r)}\mathbf{x}_{i} \ \cdot
        \ \mathbf{K}^{(r)},

    where the query (:math:`\mathbf{Q}^{(r)}`) and key kernels
    (:math:`\mathbf{K}^{(r)}`) are combined to form the attention kernels
    :math:`\mathbf{A}^{(r)}`. Now, two schemes have been proposed to compute
    attention logits :math:`\mathbf{E}^{(r)}_{i,j}` for each relation type
    (:math:`\mathbf{r}`)

    .. math:: \mathbf{E}^{(r)}_{i,j} = \mathrm{LeakyReLU}(\mathbf{q}^{(r)}_i +
        \mathbf{k}^{(r)}_j) \ \ \ \ (Additive \ attention \ logits)

    and

    .. math:: \mathbf{E}^{(r)}_{i,j} = \mathbf{q}^{(r)}_i \ \cdot
        \ \mathbf{k}^{(r)}_j \ \ \ \ (Multiplicative \ attention \ logits)

    If the graph has multi-dimensional edge features
    :math:`\mathbf{e}^{(r)}_{i,j}`, the attention logits
    :math:`\mathbf{E}^{(r)}_{i,j}` for each relation type (:math:`\mathbf{r}`)
    are computed as

    .. math:: \mathbf{E}^{(r)}_{i,j} = \mathrm{LeakyReLU}(\mathbf{q}^{(r)}_i +
        \mathbf{k}^{(r)}_j + (\mathbf{\Theta}^{(r)}_e\mathbf{e}^{(r)}_{i,j}
        \ \cdot \ \mathbf{E}^{(r)})) \ \ \ \ (Additive \ attention \ logits)

    and

    .. math:: \mathbf{E}^{(r)}_{i,j} = (\mathbf{q}^{(r)}_i \ \cdot
        \ \mathbf{k}^{(r)}_j) \ \cdot \ (\mathbf{\Theta}^{(r)}_e
        \ \mathbf{e}^{(r)}_{i,j} \ \cdot \ \mathbf{E}^{(r)})
        \ \ \ \ (Multiplicative \ attention \ logits),

    where the edge kernel (:math:`\mathbf{E}^{(r)}`) when combine with the
    query and key kernels form the final attention kernel
    (:math:`\mathbf{A}^{(r)}`). The attention coefficients
    :math:`\alpha^{(r)}_{i,j}` calculation for each relation type
    (:math:`\mathbf{r}`) is done via two different attention mechanisms

    .. math::
        \alpha^{(r)}_{i,j} =
        \frac{
        \exp\left(\mathbf{E}^{(r)}_{i,j}\right)}
        {\sum_{k \in \mathcal{N}(i)^{(r)}}
        \exp\left(\mathbf{E}^{(r)}_{i,k}\right)}
        \ \ \ (within-relation \ attention \ mechanism)

    and

    .. math::
        \alpha^{(r)}_{i,j} =
        \frac{
        \exp\left(\mathbf{E}^{(r)}_{i,j}\right)}
        {\sum_{r^{\prime} \in \mathcal{R}}\sum_{k \in
        \mathcal{N}(i)^{(r^{\prime})}}
        \exp\left(\mathbf{E}^{(r^{\prime})}_{i,k}\right)}
        \ \ \ (across-relation \ attention \ mechanism)

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}`
    for each edge. To enhance the discriminative power of attention-based GNNs,
    this layer implements four different cardinality preservation options (for
    each relation type (:math:`\mathbf{r}`)) proposed in the `"Improving
    Attention Mechanism in Graph Neural Networks via Cardinality Preservation"
    <https://arxiv.org/abs/1907.02204>`_ paper

    .. math::
        \mathbf{x}^{{\prime}(r)}_i = \sum_{j \in \mathcal{N}(i)^{(r)}}
        \alpha^{(r)}_{i,j} \mathbf{x}^{(r)}_j + \mathcal{W} \ \odot
        \ \sum_{j \in \mathcal{N}(i)^{(r)}} \mathbf{x}^{(r)}_j \ \ \ (additive)

    .. math::
        \mathbf{x}^{{\prime}(r)}_i = \psi(|\mathcal{N}^{(r)}_i|) \ \odot
        \ \sum_{j \in \mathcal{N}(i)^{(r)}} \alpha^{(r)}_{i,j}
        \mathbf{x}^{(r)}_j \ \ \ (scaled)

    .. math::
        \mathbf{x}^{{\prime}(r)}_i = \sum_{j \in \mathcal{N}(i)^{(r)}}
        (\alpha^{(r)}_{i,j} + 1) \mathbf{x}^{(r)}_j \ \ \ (f-additive)

    .. math::
        \mathbf{x}^{{\prime}(r)}_i = |\mathcal{N}^{(r)}_i| \ \odot \ \sum_{j
        \in \mathcal{N}(i)^{(r)}} \alpha^{(r)}_{i,j} \mathbf{x}^{(r)}_j
        \ \ \ (f-scaled)

    .. note::
        When :obj:`attention_mode=additive-self-attention` and
        :obj:`concat=True`, the layer outputs :obj:`heads * out_channels`
        features for each node, and when
        :obj:`attention_mode=multiplicative-self-attention` and
        :obj:`concat=True`, the layer outputs :obj:`heads * d * out_channels`
        features for each node. Besides, if
        :obj:`attention_mode=additive-self-attention` and
        :obj:`concat=False/None`, the layer outputs :obj:`out_channels`
        features for each node and if
        :obj:`attention_mode=multiplicative-self-attention` and
        :obj:`concat=False/None`, the layer outputs :obj:`d * out_channels`
        features for each node. Make sure to set the :obj:`in_channels`
        argument of this layer accordingly, if more than one instance of
        this layer is used. Furthermore, to stochastically sample the attention
        coefficients, please use :obj:`dropout` value between 0 and 1, and
        set :obj:`mod=None`.


    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int, optional): If set to not :obj:`None`, this layer will
            use the basis-decomposition regularization scheme where
            :obj:`num_bases` denotes the number of bases to use.
            (default: :obj:`None`)
        num_blocks (int, optional): If set to not :obj:`None`, this layer will
            use the block-diagonal-decomposition regularization scheme where
            :obj:`num_blocks` denotes the number of blocks to use.
            (default: :obj:`None`)
        aggr (string, optional): The aggregation scheme to use.
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`)
        mod (string, optional): The cardinality preservation option to use.
            (:obj:`"additive"`, :obj:`"scaled"`, :obj:`"f-additive"`,
            :obj:`"f-scaled"`, default: :obj:`None`)
        attention_mechanism (string): The attention mechanism to use.
            (:obj:`"within-relation"`, :obj:`"across-relation"`)
        attention_mode (string): The mode to calculate attention logits.
            (:obj:`"additive-self-attention"`,
            :obj:`"multiplicative-self-attention"`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        d (int): Number of dimensions for query and key kernels.
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
            self-loops to the input graph. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_dim != None`).
            If given as :obj:`float`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            (default: :obj:`1.`)
        bias (bool, optional): If set to :obj:`False`, the layer will not
            learn an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        num_bases: Optional[int] = None,
        num_blocks: Optional[int] = None,
        aggr: str = 'add',
        mod: Optional[str] = None,
        attention_mechanism: str = "within-relation",
        attention_mode: str = "additive-self-attention",
        heads: int = 1,
        d: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        edge_dim: Optional[int] = None,
        fill_value: float = 1.,
        bias: bool = True,
        **kwargs,
    ):

        super().__init__(aggr=aggr, node_dim=0, **kwargs)

        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.mod = mod
        self.activation = ReLU()
        self.concat = concat
        self.attention_mode = attention_mode
        self.attention_mechanism = attention_mechanism
        self.d = d
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        mod_types = ['additive', 'scaled', 'f-additive', 'f-scaled']

        if (self.attention_mechanism != "within-relation"
                and self.attention_mechanism != "across-relation"):
            raise ValueError('attention mechanism must either be '
                             '"within-relation" or "across-relation"')

        if (self.attention_mode != "additive-self-attention"
                and self.attention_mode != "multiplicative-self-attention"):
            raise ValueError('attention mode must either be '
                             '"additive-self-attention" or '
                             '"multiplicative-self-attention"')

        if self.attention_mode == "additive-self-attention" and self.d > 1:
            raise ValueError('"additive-self-attention" mode cannot be '
                             'applied when value of d is greater than 1. '
                             'Use "multiplicative-self-attention" instead.')

        if (self.edge_dim is not None and self.edge_dim != 1
                and self.add_self_loops):
            raise ValueError('"edge_dim" with value greater than 1 is '
                             'currently not yet supported for '
                             '"edge_attr" while adding self loops to the '
                             'input graph')

        if self.dropout > 0.0 and self.mod in mod_types:
            raise ValueError('mod must be None with dropout value greater '
                             'than 0 in order to sample attention '
                             'coefficients stochastically')

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        # The learnable parameters to compute both attention logits and
        # attention coefficients:
        self.q = Parameter(
            torch.Tensor(self.heads * self.out_channels, self.heads * self.d))
        self.k = Parameter(
            torch.Tensor(self.heads * self.out_channels, self.heads * self.d))

        if bias and concat:
            self.bias = Parameter(
                torch.Tensor(self.heads * self.d * self.out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(self.d * self.out_channels))
        else:
            self.register_parameter('bias', None)

        if edge_dim is not None:
            self.lin_edge = Linear(self.edge_dim,
                                   self.heads * self.out_channels, bias=False,
                                   weight_initializer='glorot')
            self.e = Parameter(
                torch.Tensor(self.heads * self.out_channels,
                             self.heads * self.d))
        else:
            self.lin_edge = None
            self.register_parameter('e', None)

        if num_bases is not None:
            self.att = Parameter(
                torch.Tensor(self.num_relations, self.num_bases))
            self.basis = Parameter(
                torch.Tensor(self.num_bases, self.in_channels,
                             self.heads * self.out_channels))
        elif num_blocks is not None:
            assert (self.in_channels % self.num_blocks == 0
                    and (self.heads * self.out_channels) %
                    self.num_blocks == 0), \
                "both 'in_channels' and 'heads * out_channels' must be " \
                "multiple of 'num_blocks' used"
            self.weight = Parameter(
                torch.Tensor(self.num_relations, self.num_blocks,
                             self.in_channels // self.num_blocks,
                             (self.heads * self.out_channels) //
                             self.num_blocks))
        else:
            self.weight = Parameter(
                torch.Tensor(self.num_relations, self.in_channels,
                             self.heads * self.out_channels))

        self.w = Parameter(torch.ones(self.out_channels))
        self.l1 = Parameter(torch.Tensor(1, self.out_channels))
        self.b1 = Parameter(torch.Tensor(1, self.out_channels))
        self.l2 = Parameter(torch.Tensor(self.out_channels, self.out_channels))
        self.b2 = Parameter(torch.Tensor(1, self.out_channels))

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.num_bases is not None:
            glorot(self.basis)
            glorot(self.att)
        else:
            glorot(self.weight)
        glorot(self.q)
        glorot(self.k)
        zeros(self.bias)
        ones(self.l1)
        zeros(self.b1)
        torch.full(self.l2.size(), 1 / self.out_channels)
        zeros(self.b2)
        if self.lin_edge is not None:
            glorot(self.lin_edge)
            glorot(self.e)

    def forward(self, x: Tensor, edge_index: Adj, edge_type: OptTensor = None,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):
        r"""
        Args:
            x: The input node features. Can be either a :obj:`[num_nodes,
                in_channels]` node feature matrix, or an optional
                one-dimensional node index tensor (in which case input features
                are treated as trainable node embeddings).
            edge_type: The one-dimensional relation type/index for each edge in
                :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.tensor.SparseTensor`.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x.size(0)
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        out = self.propagate(edge_index=edge_index, edge_type=edge_type, x=x,
                             size=size, edge_attr=edge_attr)

        alpha = self._alpha
        assert alpha is not None, "attention coefficients don't have any value"
        self._alpha = None

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_type: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.num_bases is not None:  # Basis-decomposition =================
            w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
            w = w.view(self.num_relations, self.in_channels,
                       self.heads * self.out_channels)
        if self.num_blocks is not None:  # Block-diagonal-decomposition =======
            if (x_i.dtype == torch.long and x_j.dtype == torch.long
                    and self.num_blocks is not None):
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')
            w = self.weight
            x_i, x_j = x_i.view(-1, 1, w.size(1), w.size(2)), x_j.\
                view(-1, 1, w.size(1), w.size(2))
            w = torch.index_select(w, 0, edge_type)
            outi = torch.einsum('abcd,acde->ace', x_i, w).contiguous().\
                view(-1, self.heads * self.out_channels)
            outj = torch.einsum('abcd,acde->ace', x_j, w).contiguous().\
                view(-1, self.heads * self.out_channels)
        else:  # No regularization/Basis-decomposition ========================
            if self.num_bases is None:
                w = self.weight
            w = torch.index_select(w, 0, edge_type)
            outi = torch.bmm(x_i.unsqueeze(1), w).squeeze(-2)
            outj = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        qi = torch.matmul(outi, self.q)
        kj = torch.matmul(outj, self.k)

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None, "Please set 'edge_dim = " \
                                              "edge_attr.size(-1)' while " \
                                              "calling the RGATConv layer"
            edge_attributes = self.lin_edge(edge_attr).view(
                -1, self.heads * self.out_channels)
            if edge_attributes.size(0) != edge_attr.size(0):
                edge_attributes = torch.index_select(edge_attributes, 0,
                                                     edge_type)
            alpha_edge = torch.matmul(edge_attributes, self.e)

        if self.attention_mode == "additive-self-attention":
            if edge_attr is not None:
                alpha = torch.add(qi, kj) + alpha_edge
            else:
                alpha = torch.add(qi, kj)
            alpha = F.leaky_relu(alpha, self.negative_slope)
        elif self.attention_mode == "multiplicative-self-attention":
            if edge_attr is not None:
                alpha = (qi * kj) * alpha_edge
            else:
                alpha = qi * kj

        if self.attention_mechanism == "within-relation":
            across_out = torch.zeros_like(alpha)

            for r in range(index.size(0)):
                g = softmax(alpha[r], index[r], ptr)
                across_out[r] += g.view(self.heads * self.d)
            alpha = across_out
        elif self.attention_mechanism == "across-relation":
            alpha = softmax(alpha, index, ptr, size_i)

        self._alpha = alpha

        if self.mod == "additive":
            if self.attention_mode == "additive-self-attention":
                ones = torch.ones_like(alpha)
                h = outj.view(-1, self.heads, self.out_channels) * \
                    ones.view(-1, self.heads, 1)
                h = torch.mul(self.w, h)

                return outj.view(-1, self.heads, self.out_channels) * alpha.\
                    view(-1, self.heads, 1) + h
            elif self.attention_mode == "multiplicative-self-attention":
                ones = torch.ones_like(alpha)
                h = outj.view(-1, self.heads, 1, self.out_channels) * \
                    ones.view(-1, self.heads, self.d, 1)
                h = torch.mul(self.w, h)

                return outj.view(-1, self.heads, 1,
                                 self.out_channels) * alpha.\
                    view(-1, self.heads, self.d, 1) + h

        elif self.mod == "scaled":
            if self.attention_mode == "additive-self-attention":
                ones = alpha.new_ones(index.size())
                degree = scatter_add(ones, index,
                                     dim_size=size_i)[index].unsqueeze(-1)
                degree = torch.matmul(degree, self.l1) + self.b1
                degree = self.activation(degree)
                degree = torch.matmul(degree, self.l2) + self.b2

                return torch.mul(
                    outj.view(-1, self.heads, self.out_channels) *
                    alpha.view(-1, self.heads, 1),
                    degree.view(-1, 1, self.out_channels))
            elif self.attention_mode == "multiplicative-self-attention":
                ones = alpha.new_ones(index.size())
                degree = scatter_add(ones, index,
                                     dim_size=size_i)[index].unsqueeze(-1)
                degree = torch.matmul(degree, self.l1) + self.b1
                degree = self.activation(degree)
                degree = torch.matmul(degree, self.l2) + self.b2

                return torch.mul(
                    outj.view(-1, self.heads, 1, self.out_channels) *
                    alpha.view(-1, self.heads, self.d, 1),
                    degree.view(-1, 1, 1, self.out_channels))

        elif self.mod == "f-additive":
            alpha = torch.where(alpha > 0, alpha + 1, alpha)

        elif self.mod == "f-scaled":
            ones = alpha.new_ones(index.size())
            degree = scatter_add(ones, index,
                                 dim_size=size_i)[index].unsqueeze(-1)
            alpha = alpha * degree

        elif self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        else:
            alpha = alpha  # original

        if self.attention_mode == "additive-self-attention":
            return alpha.view(-1, self.heads, 1) * outj.view(
                -1, self.heads, self.out_channels)
        elif self.attention_mode == "multiplicative-self-attention":
            return (alpha.view(-1, self.heads, self.d, 1) *
                    outj.view(-1, self.heads, 1, self.out_channels))

    def update(self, aggr_out: Tensor) -> Tensor:
        if self.attention_mode == "additive-self-attention":
            if self.concat is True:
                aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
            else:
                aggr_out = aggr_out.mean(dim=1)

            if self.bias is not None:
                aggr_out = aggr_out + self.bias

            return aggr_out
        elif self.attention_mode == "multiplicative-self-attention":
            if self.concat is True:
                aggr_out = aggr_out.view(
                    -1, self.heads * self.d * self.out_channels)
            else:
                aggr_out = aggr_out.mean(dim=1)
                aggr_out = aggr_out.view(-1, self.d * self.out_channels)

            if self.bias is not None:
                aggr_out = aggr_out + self.bias

            return aggr_out

    def __repr__(self) -> str:
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
