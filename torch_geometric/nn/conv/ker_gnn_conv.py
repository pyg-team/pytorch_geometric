from typing import Union

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing, SimpleConv
from torch_geometric.nn.dense import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import to_undirected


class KerGNNConv(MessagePassing):
    r"""The kernel graph convolutional operator from the
    `"KerGNNs: Interpretable Graph Neural Networks with Graph Kernels" 
    <https://arxiv.org/abs/2201.00491>`_ paper

    .. math::
        \mathbf{X_k}^{\prime}=\sum_{i=1}^{n_1}\sum_{j=1}^{n_2}[
        (\mathbf{X_1}\mathbf{X_2}^T)\odot(\mathbf{A_1}^p\mathbf{X_1}
        (\mathbf{A_2}^p\mathbf{X_2})^T)]_{ij},

    where :math:`\mathbf{X_k}^{\prime}` means the k-th dimention of node's 
    feature :math:`\mathbf{X}^{\prime}`;
    :math:`\mathbf{X_1}` and :math:`\mathbf{A_1}` denotes the node's 
    feature and adjacency matrix of original graph;
    :math:`\mathbf{X_2}` and :math:`\mathbf{A_2}` denotes the node's 
    feature and adjacency matrix of graph filter.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        kernel (str): Kernel type. (default: :obj:`rw`)
        power (int): The power of the adjacency matrix to use. 
            (default: :obj:`1`)
        size_graph_filter (int): The number of graph filter. 
            (default: :obj:`10`)
        dropout (float, optional): Dropout probability of the nodes 
            during training. (default: 0)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:**
          node features :math:`(|\mathcal{V}|, |P| \cdot F_{out})`
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel: str = 'rw', power: int = 1, 
                 size_graph_filter: int = 6, dropout: float = 0., **kwargs):
        assert kernel in ['rw', 'drw'] and power > 0
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.power = power
        self.size_graph_filter = size_graph_filter

        self.filter_x = Parameter(torch.empty(
            size_graph_filter, in_channels, out_channels))
        self.filter_edge_index = torch.triu_indices(size_graph_filter,
                                                    size_graph_filter, 1)
        self.filter_edge_attr = Parameter(torch.empty(
            size_graph_filter * (size_graph_filter - 1) // 2, out_channels))
        self.filter_g = None

        self.simple_conv = SimpleConv()
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.ker_linear = Linear(
            out_channels,
            out_channels) if kernel == 'drw' else torch.nn.Identity()

        self.reset_parameters()
        assert self.filter_g is not None and self.filter_g.validate()

    def reset_parameters(self):
        super().reset_parameters()
        self.filter_x.data.uniform_(0, 1)
        self.filter_edge_attr.data.uniform_(-1, 1)
        bi_filter_edge_index, bi_filter_edge_attr = to_undirected(
            self.filter_edge_index, self.filter_edge_attr)

        self.filter_g = Data(x=self.filter_x,
                             edge_index=bi_filter_edge_index,
                             edge_attr=bi_filter_edge_attr)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        outs = None
        z = self.filter_g.x
        xz = x @ z
        for i in range(self.power):
            if i == 0:
                o = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   z=z)
            else:
                x = self.simple_conv(x, edge_index=edge_index)
                # propagate_type: (x: Tensor, edge_weight: OptTensor, 
                # z: Tensor, is_kernel: bool)
                z = self.propagate(self.filter_g.edge_index,
                                   x=z.view(self.size_graph_filter, -1), 
                                   edge_weight=edge_weight,
                                   z=self.filter_g.edge_attr, 
                                   is_kernel=True)
                z = z.view(self.size_graph_filter, -1, self.out_channels)
                o = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   z=z)
            t = self.ker_linear(self.dropout(o) * xz)
            outs = t if outs is None else outs + t
        return torch.mean(outs / self.power, dim=[0])

    def message(self, x_j: Tensor, edge_weight: OptTensor, z: Tensor,
                is_kernel: bool = False) -> Tensor:
        x_j = x_j if edge_weight is None else x_j * edge_weight.view(-1, 1)
        if is_kernel:
            num_edges = z.size(0)
            x_j = x_j.view(num_edges, -1, self.out_channels)
            z = z.unsqueeze(1)
            return (x_j * self.relu(z)).view(num_edges, -1)
        else:
            return x_j @ z

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({self.in_channels}, '
            f'{self.out_channels}, kernel={self.kernel}, power={self.power})')
