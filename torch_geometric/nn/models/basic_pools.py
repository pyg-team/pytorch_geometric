from typing import Optional, Callable, List
from torch_geometric.typing import Adj

import copy

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU

from torch_geometric.data import Batch
from torch_geometric.nn.conv import GCNConv, SAGEConv, GINConv, GINEConv, GATConv, GATv2Conv, GraphConv
from torch_geometric.nn.pool import EdgePooling, SAGPooling, ASAPooling, graclus, max_pool
from torch_geometric.nn.glob import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge


class BasicPoolGNN(torch.nn.Module):
    r"""An abstract class for implementing basic graph pooling architectures.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden and output sample.
        num_layers (int): Number of message passing layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        glob_pool (str, optional): The global pooling operator to use
            (:obj:`"add"`, :obj:`"max"`, :obj:`"mean"`).
            (default: :obj:`"mean"`)
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 glob_pool: str = 'mean'):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        if jk == 'cat':
            self.out_channels = num_layers * hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act

        self.convs = ModuleList()
        self.pools = ModuleList()

        self.jk = None
        if jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        self.norms = None
        if norm is not None:
            self.norms = ModuleList(
                [copy.deepcopy(norm) for _ in range(num_layers)])

        if glob_pool == 'mean':
            self.global_pool = global_mean_pool
        elif glob_pool == 'max':
            self.global_pool = global_max_pool
        elif glob_pool == 'add':
            self.global_pool = global_add_pool

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if self.jk is not None:
            self.jk.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, batch: Tensor, *args,
                **kwargs) -> Tensor:
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)

            xs += [self.global_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, _, batch, = pool(x, edge_index, batch=batch)

            x = F.dropout(x, p=self.dropout, training=self.training)

        return x if self.jk is None else self.jk(xs)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class ASAPool(BasicPoolGNN):
    r"""The Graph Neural Network from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical
    Graph Representations" <https://arxiv.org/abs/1911.07979>`_ paper, using the
    :class:`~torch_geometric.nn.GraphConv` operator for message passing
    and the :class:`~torch_geometric.nn.ASAPooling` operator for pooling.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        glob_pool (str, optional): The global pooling operator to use
            (:obj:`"add"`, :obj:`"max"`, :obj:`"mean"`).
            (default: :obj:`"mean"`)
        ratio (float, optional): Ratio of nodes to aggregate. (default: :obj:`0.8`)
        dropout_ratio (float, optional): Dropout ratio. (default: :obj:`0.`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GraphConv`.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 glob_pool: str = 'mean', ratio: float = 0.8, **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers, dropout,
                         act, norm, jk, glob_pool)

        self.convs.append(GraphConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                GraphConv(hidden_channels, hidden_channels, **kwargs))

        # We need to return only the first 4 arguments of ASAPooling: x, edge_index, edge_weight, batch
        lambda_func = lambda x: (x[0], x[1], x[2], x[3])
        lambda_module = Lambda(lambda_func)

        for _ in range(num_layers // 2):
            self.pools.append(
                lambda_module(
                    ASAPooling(hidden_channels, ratio, dropout=dropout)))


class SAGPool(BasicPoolGNN):
    r"""The Graph Neural Network from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers, using the
    :class:`~torch_geometric.nn.GraphConv` operator for message passing
    and the :class:`~torch_geometric.nn.SAGPooling` operator for pooling.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        glob_pool (str, optional): The global pooling operator to use
            (:obj:`"add"`, :obj:`"max"`, :obj:`"mean"`).
            (default: :obj:`"mean"`)
        ratio (float, optional): Ratio of nodes to aggregate. (default: :obj:`0.8`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GraphConv`.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 glob_pool: str = 'mean', ratio: float = 0.8, **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers, dropout,
                         act, norm, jk, glob_pool)

        self.convs.append(GraphConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                GraphConv(hidden_channels, hidden_channels, **kwargs))

        # We need to return only the first 4 arguments of SAGooling: x, edge_index, edge_attr, batch
        lambda_func = lambda x: (x[0], x[1], None, x[3])
        lambda_module = Lambda(lambda_func)

        for _ in range(num_layers // 2):
            self.pools.append(lambda_module(SAGPooling(hidden_channels,
                                                       ratio)))


class EdgePool(BasicPoolGNN):
    r"""The Graph Neural Network from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers, using the
    :class:`~torch_geometric.nn.GraphConv` operator for message passing
    and the :class:`~torch_geometric.nn.EdgePooling` operator for pooling.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        glob_pool (str, optional): The global pooling operator to use
            (:obj:`"add"`, :obj:`"max"`, :obj:`"mean"`).
            (default: :obj:`"mean"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GraphConv`.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 glob_pool: str = 'mean', **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers, dropout,
                         act, norm, jk, glob_pool)

        self.convs.append(GraphConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                GraphConv(hidden_channels, hidden_channels, **kwargs))

        # We need to return only the first 4 arguments of EdgePooling: x, edge_index, batch, unpool_info
        lambda_func = lambda x: (x[0], x[1], None, x[2])
        lambda_module = Lambda(lambda_func)

        for _ in range(num_layers // 2):
            self.pools.append(lambda_module(EdgePooling(hidden_channels)))


class Graclus(BasicPoolGNN):
    r"""The Graph Neural Network from the `"Weighted Graph Cuts without
    Eigenvectors: A Multilevel Approach" <http://www.cs.utexas.edu/users/
    inderjit/public_papers/multilevel_pami.pdf>`_ paper, using the
    :class:`~torch_geometric.nn.GraphConv` operator for message passing
    and the :class:`~torch_geometric.nn.EdgePooling` operator for pooling.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        num_layers (int): Number of GNN layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        glob_pool (str, optional): The global pooling operator to use
            (:obj:`"add"`, :obj:`"max"`, :obj:`"mean"`).
            (default: :obj:`"mean"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GraphConv`.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 glob_pool: str = 'mean', **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers, dropout,
                         act, norm, jk, glob_pool)

        self.convs.append(GraphConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                GraphConv(hidden_channels, hidden_channels, **kwargs))

        for _ in range(num_layers // 2):
            self.pools.append(graclus_pool())


class graclus_pool(torch.nn.Module):
    """Auxiliary block for Graclus"""
    def __init__(self):
        super(graclus_pool, self).__init__()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        cluster = graclus(self.edge_index, num_nodes=self.x.size(0))
        data = Batch(x=self.x, edge_index=self.edge_index, batch=self.batch)
        data = max_pool(cluster, data)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        return x, edge_index, None, batch


class Lambda(torch.nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."

    def __init__(self, func):
        "create a layer that simply calls `func` with `x`"
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
