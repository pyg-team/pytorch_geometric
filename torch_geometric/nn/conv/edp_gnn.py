from typing import Callable, List, Union

import torch
from torch import Tensor, nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.utils import to_dense_adj  # dense_to_sparse,


class MCGNN(MessagePassing):
    r"""The multichannel gnn operator from the `"Permutation Invariant Graph
    Generation via Score-Based Generative Modeling"
    <https://arxiv.org/abs/2003.00638>`_ paper.

    TODO: Modify everything below
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
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
        # x_j has shape [E, in_channels]
        # edge_weight has shape [E, ]
        return x_j * edge_weight[:, None]

    def forward(self, x: Union[Tensor, OptPairTensor], edge_indices: List[Adj],
                edge_weights: List[Tensor], size: Size = None) -> Tensor:
        N, C = x.shape[0], len(edge_indices)
        # concatenate edge indices for different channels
        edge_index_cat = torch.cat(
            [edge_index + i * N for i, edge_index in enumerate(edge_indices)],
            dim=-1)
        edge_weights_cat = torch.cat(edge_weights, dim=0)
        # duplicate features to run over C channels
        if isinstance(x, Tensor):
            x_cat = torch.cat([x] * C, dim=1)
            x_cat: OptPairTensor = (x_cat, x_cat)
        else:
            x_cat = (torch.cat([x[i]] * C, dim=1) for i in range(2))
        # propagate_type: (x: OptPairTensor)
        out_cat = self.propagate(edge_index_cat, edge_weight=edge_weights_cat,
                                 x=x_cat, size=size)

        x_r = x_cat[1]
        if x_r is not None:
            out_cat = out_cat + (1 + self.eps) * x_r

        # reshape and re-concatenate:
        # TODO: check if the dim is correct
        out_cat = torch.reshape(out_cat, (C, N, -1)).permute((1, 0, 2))
        out = torch.reshape(out_cat, (N, -1))

        return self.nn(out)


class EDPGNN(nn.Module):
    r"""The Edgewise Dense Prediction Graph Neural Network (EDP-GNN) operator
    from the `"Permutation Invariant Graph Generation via
    Score-Based Generative Modeling" <https://arxiv.org/abs/2003.00638>` paper.

    TODO: Modify everything below
    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn_node: Callable, nn_edge: Callable, eps: float = 0.,
                 train_eps: bool = False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn_edge = nn_edge
        self.mc_gnn = MCGNN(nn_node, eps, train_eps)
        self.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_indeces: List[Adj],
                edge_weights: List[Tensor],
                size: Size = None) -> (Tensor, List[Adj]):
        # 1. get node embeddings
        x_next = self.mc_gnn(x)  # N x emb_len
        N, _ = x_next.shape
        # 2. get adjacency matrix
        a = [
            to_dense_adj(edge_index=i, edge_attr=w, max_num_nodes=N)
            for i, w in zip(edge_indeces, edge_weights)
        ]
        a = torch.cat(a, dim=2)  # N x N x C
        x_i = x_next[:, None, :].repeat(1, N, 1)  # N x N x emb_len
        x_j = x_next[None, :, :].repeat(N, 1, 1)  # N x N x emb_len
        out = torch.cat((a, x_i, x_j), dim=2)
        adj_out = self.nn_edge(out.reshape((N * N, -1))).reshape(N, N, -1)
        # TODO: sumup transpose
        return adj_out, [0]
