from typing import Optional

import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.utils import spmm


class GPRGNN(MessagePassing):
    r"""The adaptive generalized PageRank layer from the
    `"Adaptive Universal Generalized PageRank Graph Neural
    Network" <https://arxiv.org/abs/2006.07988>`_ paper

    .. math::
        \mathbf{X}^{(0)} &= \mathbf{X}

        \mathbf{X}^{(k)} &= \mathbf{\hat{D}}^{-1/2}
        \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(k-1)}

        \mathbf{X}^{\prime} &= \sum_{k=0}^K \gamma_k \mathbf{X}^{(k)},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include values other than :obj:`1`, representing
    edge weights via the optional :obj:`edge_weight` tensor. Here
    :math:`\gamma_k` denotes the learnable generalized PageRank (GPR)
    weight for layer :math:`k`. There are four ways to initialize the
    weights: :obj:`Delta`, Personalized PageRank (:obj:`PPR`), Negative
    Personalized PageRank (:obj:`NPPR`), and :obj:`Random`.

    Args:
        K (int): Number of propagation layers :math:`K`.
        alpha (float): Initialization parameter :math:`\alpha`.
        init (str, optional): Initialization method for GPR weights. Choices
            include ['Delta', 'PPR', 'NPPR', 'Random']. (default: :obj:`PPR`)
        dropout (float, optional): Dropout probability of edges during
            training. (default: :obj:`0`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, alpha: float, init: str = 'PPR',
                 dropout: float = 0., cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.init = init
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None
        self.temp = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

        # Initialize the GPR weights
        assert self.init in ['Delta', 'PPR', 'NPPR', 'Random'], \
            'Not supported initialization method.'
        if self.init == 'Delta':
            # alpha will be converted to integer here. It means
            # where the peak at when initializing GPR weights.
            self.alpha = int(self.alpha)
            TEMP = 0.0 * np.ones(self.K + 1)
            TEMP[self.alpha] = 1.0
        elif self.init == 'PPR':
            TEMP = self.alpha * (1 - self.alpha)**np.arange(self.K + 1)
            TEMP[-1] = (1 - self.alpha)**self.K
        elif self.init == 'NPPR':
            TEMP = self.alpha**np.arange(self.K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        else:
            # Random
            bound = np.sqrt(3 / (self.K + 1))
            TEMP = np.random.uniform(-bound, bound, self.K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))

        self.temp = Parameter(Tensor(TEMP))

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        h = x * self.temp[0]
        for k in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            h = h + self.temp[k + 1] * x

        return h

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, temp={self.temp})'
