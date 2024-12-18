import typing
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    scatter,
    softmax,
    to_edge_index,
    to_torch_csr_tensor,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload


@torch.jit._overload
def gat_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, flow, dtype):
    # type: (Tensor, Tensor, Optional[int], str, Optional[int]) -> Tuple[Tensor, Tensor]  # noqa
    pass


@torch.jit._overload
def gat_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, flow, dtype):
    # type: (SparseTensor, Tensor, Optional[int], str, Optional[int]) -> Tuple[SparseTensor, Tensor]  # noqa
    pass


@torch.jit._overload
def gat_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, flow, dtype):
    # type: (Adj, Tensor, Optional[int], str, Optional[int]) -> Tuple[Adj, Tensor]  # noqa
    pass


def gat_norm(  # noqa: F811
    edge_index: Adj,
    edge_weight: Tensor,
    num_nodes: Optional[int] = None,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 1.0

    assert flow in ['source_to_target', 'target_to_source']

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)

        # idx = col if flow == 'source_to_target' else row
        dim = 1 if flow == 'source_to_target' else 0
        deg = torch_sparse.sum(adj_t, dim=dim)

        att_mat = adj_t.set_value(edge_weight)
        # repeat_interleave for any dtype tensor
        num_heads = edge_weight.shape[1] if edge_weight.dim() > 1 else 1
        repeat_deg = deg.view(-1).unsqueeze(1).repeat(1, num_heads).flatten()
        # perform mat multiplication with sparse tensor
        # for multiple attention heads
        att_mat = torch_sparse.mul(att_mat, repeat_deg.view(-1, 1, num_heads))

        # Add self-loop, also called renormalization trick
        att_mat = torch_sparse.fill_diag(att_mat, fill_value)
        deg_tilde = deg + 1
        deg_tilde_inv_sqrt = deg_tilde.pow_(-0.5)
        deg_tilde_inv_sqrt.masked_fill_(deg_tilde_inv_sqrt == float('inf'), 0.)
        # repeat_interleave for any dtype tensor
        repeat_deg_tilde_inv = deg_tilde_inv_sqrt.view(-1).unsqueeze(1).repeat(
            1, num_heads).flatten()
        repeat_deg_tilde_inv = repeat_deg_tilde_inv

        att_mat = torch_sparse.mul(att_mat,
                                   repeat_deg_tilde_inv.view(-1, 1, num_heads))
        att_mat = torch_sparse.mul(att_mat,
                                   repeat_deg_tilde_inv.view(1, -1, num_heads))
        # should never be None, only for typing purpose
        alpha = att_mat.storage.value()
        return att_mat.set_value(
            None), alpha if alpha is not None else edge_weight

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gat_norm'")
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        edge_index, value = to_edge_index(edge_index)

        col, row = edge_index[0], edge_index[1]
        idx = col if flow == 'source_to_target' else row
        deg = scatter(value, idx, 0, dim_size=num_nodes, reduce='sum')
        att_mat = deg[col].view(-1, 1) * edge_weight

        # Add self-loop, also called renormalization trick
        edge_index, att_mat = add_remaining_self_loops(edge_index, att_mat,
                                                       fill_value=fill_value,
                                                       num_nodes=num_nodes)
        col, row = edge_index[0], edge_index[1]
        deg_tilde = deg + 1
        deg_tilde_inv_sqrt = deg_tilde.pow_(-0.5)
        deg_tilde_inv_sqrt.masked_fill_(deg_tilde_inv_sqrt == float('inf'), 0)
        att_mat = deg_tilde_inv_sqrt[row].view(
            -1, 1) * att_mat * deg_tilde_inv_sqrt[col].view(-1, 1)

        # Sort edge_index lexicographically
        sorted_indices = torch.argsort(edge_index[0] * edge_index.size(1) +
                                       edge_index[1])
        edge_index = edge_index[:, sorted_indices]
        att_mat = att_mat[sorted_indices]

        return to_torch_csr_tensor(edge_index), att_mat

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    adj_t = torch.ones((edge_index.size(1), ), dtype=dtype,
                       device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(adj_t, idx, dim=0, dim_size=num_nodes, reduce='sum')
    att_mat = deg[col].view(-1,
                            1) * edge_weight  # unnormalized attention matrix

    # Add self-loop, also called renormalization trick
    adj_t, att_mat = add_self_loops(edge_index, att_mat, fill_value=fill_value,
                                    num_nodes=num_nodes)
    row, col = adj_t[0], adj_t[1]
    deg_tilde = deg + 1
    deg_tilde_inv_sqrt = deg_tilde.pow_(-0.5)
    deg_tilde_inv_sqrt.masked_fill_(deg_tilde_inv_sqrt == float('inf'), 0)
    att_mat = deg_tilde_inv_sqrt[row].unsqueeze(
        1) * att_mat * deg_tilde_inv_sqrt[col].unsqueeze(1)

    return adj_t, att_mat


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i) \cup \{ i \}}
        \alpha_{i,j}\mathbf{\Theta}_t\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(
        \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
        + \mathbf{a}^{\top}_{t} \mathbf{\Theta}_{t}\mathbf{x}_j
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(
        \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
        + \mathbf{a}^{\top}_{t}\mathbf{\Theta}_{t}\mathbf{x}_k
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(
        \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
        + \mathbf{a}^{\top}_{t} \mathbf{\Theta}_{t}\mathbf{x}_j
        + \mathbf{a}^{\top}_{e} \mathbf{\Theta}_{e} \mathbf{e}_{i,j}
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(
        \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
        + \mathbf{a}^{\top}_{t} \mathbf{\Theta}_{t}\mathbf{x}_k
        + \mathbf{a}^{\top}_{e} \mathbf{\Theta}_{e} \mathbf{e}_{i,k}
        \right)\right)}.

    If the graph is not bipartite, :math:`\mathbf{\Theta}_{s} =
    \mathbf{\Theta}_{t}`.

    Normalization will be computed as presented in
    `"Bag of Tricks for Node Classification with Graph Neural Networks"
    <https://arxiv.org/abs/2103.13355>`__ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}_{att}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}_{att}} = A_{att} + \mathbf{I}` denotes the
    unnormalized attention matrix :math:`\mathbf{\hat{A}_{att}}
    = \mathbf{D}\mathbf{\alpha}`
    with inserted self-loops, :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}`
    its diagonal degree matrix based on the adjacency matrix with inserted
    self-loops :math:`\hat{A} = A + I`.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities in case of a bipartite graph.
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
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops (in case
            :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        residual (bool, optional): If set to :obj:`True`, the layer will add
            a learnable skip-connection. (default: :obj:`False`)
        normalize (bool, optional): If set to :obj:`True`, will add
            self-loops to the input graph and compute symmetric normalization
            coefficients on-the-fly. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        residual: bool = False,
        normalize: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.normalize = normalize

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        self.lin = self.lin_src = self.lin_dst = None
        if isinstance(in_channels, int):
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        # The number of output channels:
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels
                if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.lin is not None:
            self.lin.reset_parameters()
        if self.lin_src is not None:
            self.lin_src.reset_parameters()
        if self.lin_dst is not None:
            self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    @overload
    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights: Optional[bool] = None,
    ) -> Union[
            Tensor,
            Tuple[Tensor, Tuple[Tensor, Tensor]],
            Tuple[Tensor, SparseTensor],
    ]:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            size ((int, int), optional): The shape of the adjacency matrix.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        # NOTE: attention weights will be returned whenever
        # `return_attention_weights` is set to a value, regardless of its
        # actual value (might be `True` or `False`). This is a current somewhat
        # hacky workaround to allow for TorchScript support via the
        # `torch.jit._overload` decorator, as we can only change the output
        # arguments conditioned on type (`None` or `bool`), not based on its
        # actual value.

        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

            if self.res is not None:
                res = self.res(x)

            if self.lin is not None:
                x_src = x_dst = self.lin(x).view(-1, H, C)
            else:
                # If the module is initialized as bipartite, transform source
                # and destination node features separately:
                assert self.lin_src is not None and self.lin_dst is not None
                x_src = self.lin_src(x).view(-1, H, C)
                x_dst = self.lin_dst(x).view(-1, H, C)

        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"

            if x_dst is not None and self.res is not None:
                res = self.res(x_dst)

            if self.lin is not None:
                # If the module is initialized as non-bipartite, we expect that
                # source and destination node features have the same shape and
                # that they their transformations are shared:
                x_src = self.lin(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin(x_dst).view(-1, H, C)
            else:
                assert self.lin_src is not None and self.lin_dst is not None

                x_src = self.lin_src(x_src).view(-1, H, C)
                if x_dst is not None:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        # Skipping add_self_loops when normalize
        # as we are performing it on gat_norm function
        if self.add_self_loops and not self.normalize:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        if self.normalize:
            if not isinstance(self.in_channels, int):
                raise NotImplementedError(
                    "The usage of 'normalize' is not supported "
                    "for bipartite message passing.")

            if isinstance(edge_index, Tensor):
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.remove_diag(edge_index)

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr,
                                  size=size)

        if self.normalize:
            num_nodes: Optional[int] = None
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes

            edge_index, alpha = gat_norm(edge_index, alpha,
                                         num_nodes=num_nodes, flow=self.flow,
                                         dtype=alpha.dtype)  # noqa: F811

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
