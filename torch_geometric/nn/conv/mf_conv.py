from typing import Tuple, Union

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import HeteroLinear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import degree, spmm


class MFConv(MessagePassing):
    r"""The graph neural network operator from the
    `"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
    <https://arxiv.org/abs/1509.09292>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}^{(\deg(i))}_1 \mathbf{x}_i +
        \mathbf{W}^{(\deg(i))}_2 \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j

    which trains a distinct weight matrix for each possible vertex degree.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        max_degree (int, optional): The maximum node degree to consider when
            updating weights (default: :obj:`10`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, max_degree: int = 10, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_degree = max_degree

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # self.lins_l = ModuleList([
        #     Linear(in_channels[0], out_channels, bias=bias)
        #     for _ in range(max_degree + 1)
        # ])

        # self.lins_r = ModuleList([
        #     Linear(in_channels[1], out_channels, bias=False)
        #     for _ in range(max_degree + 1)
        # ])

        self.lin_l = HeteroLinear(in_channels[0], out_channels, num_types=max_degree+1, is_sorted=True, bias=bias)
        self.lin_r = HeteroLinear(in_channels[1], out_channels, num_types=max_degree+1, is_sorted=True, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        x_r = x[1]

        deg = x[0]  # Dummy.
        if isinstance(edge_index, SparseTensor):
            deg = edge_index.storage.rowcount()
        elif isinstance(edge_index, Tensor):
            i = 1 if self.flow == 'source_to_target' else 0
            N = x[0].size(self.node_dim)
            N = size[1] if size is not None else N
            N = x_r.size(self.node_dim) if x_r is not None else N
            deg = degree(edge_index[i], N, dtype=torch.long)
        deg.clamp_(max=self.max_degree)

        # propagate_type: (x: OptPairTensor)
        h = self.propagate(edge_index, x=x, size=size)

        out = h.new_empty(list(h.size())[:-1] + [self.out_channels])

        # idx select loop for l
        h_sel_list, type_list_l = [], []
        for i in range(self.max_degree+1):
            idx = (deg == i).nonzero().view(-1)
            h_idx_sel = h.index_select(self.node_dim, idx)
            N = h_idx_sel.size(0)
            h_sel_list.append(h_idx_sel)
            type_list_l.append(torch.full((N, ), i, dtype=torch.long))
        x_l = torch.cat(h_sel_list, dim=0)
        type_vec_l = torch.cat(type_list_l, dim=0)

        # apply lin_l
        print("x_l.shape=",x_l.shape)
        print("type_vec_l.shape=",type_vec_l.shape)
        print("lin_l=", self.lin_l)
        r = self.lin_l(x_l, type_vec_l)

        # idx select loop for r
        if x_r is not None:
            r_sel_list, type_list_r, idx_list = [], [], []
            count = 0
            for i in range(self.max_degree+1):
                idx_i = (deg == i).nonzero().view(-1)
                
                N = idx_i.numel()
                if N == 0:
                    continue
                r_idx_sel = x_r.index_select(self.node_dim, idx_i)
                r_sel_list.append(r_idx_sel)
                idx_list.append(idx_i + count)
                count += N
                type_list_r.append(torch.full((N, ), i, dtype=torch.long))
            x_r = torch.cat(r_sel_list, dim=0)
            type_vec_r = torch.cat(type_list_r, dim=0)
            idx = torch.cat(idx_list, dim=0)
            # apply lin_r
            r += self.lin_r(x_r, type_vec_r)

        out.index_copy_(self.node_dim, idx, r)
        # for i, (lin_l, lin_r) in enumerate(zip(self.lins_l, self.lins_r)):
        #     idx = (deg == i).nonzero().view(-1)
        #     r = lin_l(h.index_select(self.node_dim, idx))

        #     if x_r is not None:
        #         r = r + lin_r(x_r.index_select(self.node_dim, idx))

        #     out.index_copy_(self.node_dim, idx, r)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)
