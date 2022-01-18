from typing import Optional, Tuple
from torch_geometric.typing import OptTensor, Adj

from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class FAConv(MessagePassing):
    r"""The Frequency Adaptive Graph Convolution operator from the
    `"Beyond Low-Frequency Information in Graph Convolutional Networks"
    <https://arxiv.org/abs/2101.00797>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i= \epsilon \cdot \mathbf{x}^{(0)}_i +
        \sum_{j \in \mathcal{N}(i)} \frac{\alpha_{i,j}}{\sqrt{d_i d_j}}
        \mathbf{x}_{j}

    where :math:`\mathbf{x}^{(0)}_i` and :math:`d_i` denote the initial feature
    representation and node degree of node :math:`i`, respectively.
    The attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \mathbf{\alpha}_{i,j} = \textrm{tanh}(\mathbf{a}^{\top}[\mathbf{x}_i,
        \mathbf{x}_j])

    based on the trainable parameter vector :math:`\mathbf{a}`.

    Args:
        channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        eps (float, optional): :math:`\epsilon`-value. (default: :obj:`0.1`)
        dropout (float, optional): Dropout probability of the normalized
            coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`).
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\sqrt{d_i d_j}` on first execution, and
            will use the cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops (if
            :obj:`add_self_loops` is :obj:`True`) and compute
            symmetric normalization coefficients on the fly.
            If set to :obj:`False`, :obj:`edge_weight` needs to be provided in
            the layer's :meth:`forward` method. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          initial node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)` or
          :math:`((|\mathcal{V}|, F), ((2, |\mathcal{E}|),
          (|\mathcal{E}|)))` if :obj:`return_attention_weights=True`
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]
    _alpha: OptTensor

    def __init__(self, channels: int, eps: float = 0.1, dropout: float = 0.0,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(FAConv, self).__init__(**kwargs)

        self.channels = channels
        self.eps = eps
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        self._alpha = None

        self.att_l = Linear(channels, 1, bias=False)
        self.att_r = Linear(channels, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.att_l.reset_parameters()
        self.att_r.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, return_attention_weights=None):
        # type: (Tensor, Tensor, Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Tensor, Tensor, SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Tensor, Tensor, Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Tensor, Tensor, SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        if self.normalize:
            if isinstance(edge_index, Tensor):
                assert edge_weight is None
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, None, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                assert not edge_index.has_value()
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, None, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        else:
            if isinstance(edge_index, Tensor):
                assert edge_weight is not None
            elif isinstance(edge_index, SparseTensor):
                assert edge_index.has_value()

        alpha_l = self.att_l(x)
        alpha_r = self.att_r(x)

        # propagate_type: (x: Tensor, alpha: PairTensor, edge_weight: OptTensor)  # noqa
        out = self.propagate(edge_index, x=x, alpha=(alpha_l, alpha_r),
                             edge_weight=edge_weight, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.eps != 0.0:
            out += self.eps * x_0

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: Tensor,
                edge_weight: OptTensor) -> Tensor:
        assert edge_weight is not None
        alpha = (alpha_j + alpha_i).tanh().squeeze(-1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * (alpha * edge_weight).view(-1, 1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, eps={self.eps})'
