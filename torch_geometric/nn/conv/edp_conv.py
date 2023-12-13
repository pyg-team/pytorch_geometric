from typing import Callable, List, Union

import torch
from torch import Tensor
from torch.nn import Module

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.utils import dense_to_sparse, to_dense_adj


class MultiChannelConv(MessagePassing):
    r"""The multi channel convolutional operator from the `"Permutation
    Invariant Graph Generation via Score-Based Generative Modeling"
    <https://arxiv.org/abs/2003.00638>`_ paper.
    "The intuition is to run message-passing simultaneously on many different
    graphs, and collect the node features from all the channels via
    concatenation."
    The key thing is edhes have weights, so we have weighted messages
    to aggregate.

    .. math::
        \widetilde{X}^{\prime}_{[c,\cdot]}=A_{[c,\cdot]}X_{[\cdot]},
        for c = 0, 1, ..., C-1

        \mathbf{x}^{\prime}_i = \texttt{MLP}_{\texttt{Node}} \left(
        \texttt{CONCAT}(\widetilde{X}_{[c,i]} + (1 + \epsilon)X_i\ |
        \ c = 0, 1, \cdots, C-1 )  \right)

    here MLP denotes a neural network, in original version it is an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`\text{MLP}_{\text{Node}}`
            that maps node features :obj:`x` of shape :obj:`[-1, in_channels]`
            to shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)` for each element of List of
          :math:`C` channels
          edge weights :math:`(\mathcal{E}|)` for each element of List of
          :math:`C` channels
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))

    def message(self, x_j, edge_weight):
        return x_j * edge_weight

    def forward(self, x: Union[Tensor, OptPairTensor], edge_indices: List[Adj],
                edge_weights: List[Tensor], size: Size = None) -> Tensor:

        if size is not None and isinstance(size, int):
            size *= len(edge_weights)
        elif size is not None:
            size = (size[0] * len(edge_weights), size[1] * len(edge_weights))
        N, C = x.shape[0], len(edge_indices)
        # concatenate edge indices and weights for different channels
        edge_index_cat = torch.cat(
            [edge_index + i * N for i, edge_index in enumerate(edge_indices)],
            dim=-1)
        edge_weights_cat = torch.cat(edge_weights, dim=-1)[:, None]
        # duplicate features to run over C channels
        if isinstance(x, Tensor):
            x_cat = x.repeat(C, 1)  # N*C x F_in
            x_cat: OptPairTensor = (x_cat, x_cat)
        else:
            x_cat = (x[i].repeat(C, 1) for i in range(2))
        # propagate_type: (x: OptPairTensor)
        out_cat = self.propagate(edge_index_cat, edge_weight=edge_weights_cat,
                                 x=x_cat, size=size)

        x_r = x_cat[1]
        if x_r is not None:
            out_cat = out_cat + (1 + self.eps) * x_r  # N * C x F_in

        # reshape to get concatenated features for each of C channels
        out_cat = out_cat.reshape((C, N, -1)).permute((1, 0, 2))
        out = out_cat.reshape((N, -1))

        return self.nn(out)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class EDPConv(Module):
    r"""The Edgewise Dense Prediction Graph Neural Network (EDP-GNN)
    operator from the `"Permutation Invariant Graph Generation via
    Score-Based Generative Modeling" <https://arxiv.org/abs/2003.00638>` paper.

    1. Node feature inference
        .. math::
            \mathbf{x}^{\prime} = MultiChannelConv(X,edge_indices,edge_weights)

    2. Edge feature inference
        .. math::
            \mathbf{\widetilde{A}}^{\prime}_{[\cdot, i, j]} =
            \texttt{MLP}_{\texttt{Edge}}
            left( \texttt{CONCAT}(A_{[\cdot, i, j]}, X_i, X_j)  \right)

    To ensure that matrix is symmetric,
        .. math::
            \mathbf{A}^{\prime}=\widetilde{A}^{\prime}+\widetilde{A}^{\prime}^T

    here :math:`A^T` denotes a transposed matrix :math:`A`.

    Args:
        nn_node (torch.nn.Module): A neural network
            :math:`\texttt{MLP}_{\texttt{Node}}`
            that maps node features :obj:`x` of shape :obj:`[-1, in_channels]`
            to shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        nn_edge (torch.nn.Module): A neural network
            :math:`\texttt{MLP}_{\texttt{Edge}}`
            that maps node features :obj:`x` of shape :obj:`[-1, in_channels]`
            to shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`
          edge indices :math:`(2, |\mathcal{E}|)`
          edge weights :math:`(|\mathcal{E}|)`
        - **output:**
          node features :math:`(|\mathcal{V}|, F_{out})`
          edge indices :math:`(2, |\mathcal{E'}|)` for each of C adj. matrices
          edge weights :math:`(|\mathcal{E'}|)` for each of C adj. matrices

    """
    def __init__(self, nn_node: Callable, nn_edge: Callable, eps: float = 0.,
                 train_eps: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.nn_edge = nn_edge
        self.mc_gnn = MultiChannelConv(nn_node, eps, train_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_indeces: List[Adj],
                edge_weights: List[Tensor],
                size: Size = None) -> (Tensor, List[Adj], List[Tensor]):
        # 1. get node embeddings
        x_next = self.mc_gnn(x, edge_indeces, edge_weights)  # N x emb_len
        N, _ = x_next.shape
        # 2. get adjacency matrix
        a = [
            to_dense_adj(edge_index=i, edge_attr=w, max_num_nodes=N)
            for i, w in zip(edge_indeces, edge_weights)
        ]
        a = torch.cat(a, dim=0).permute((1, 2, 0))  # N x N x C_in
        x_i = x_next[:, None, :].repeat(1, N, 1)  # N x N x emb_len
        x_j = x_next[None, :, :].repeat(N, 1, 1)  # N x N x emb_len
        out = torch.cat((a, x_i, x_j), dim=2)
        adj_out = self.nn_edge(out.reshape(
            (N * N, -1))).reshape(N, N, -1)  # N x N x C_out
        # sum up with transposed output to be sure that matrix is symmetric
        adj_out = adj_out + adj_out.permute((1, 0, 2))
        # create output edges' predictions
        edge_indices, edge_weights = [], []
        for i in range(adj_out.shape[-1]):
            edge_index, edge_weight = dense_to_sparse(adj_out[:, :, i])
            edge_indices.append(edge_index)
            edge_weights.append(edge_weight)
        return x_next, edge_indices, edge_weights
