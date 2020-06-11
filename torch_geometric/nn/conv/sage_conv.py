from typing import Optional
from torch_geometric.typing import OptPairTensor, Size

import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse.matmul import spmm
from torch_geometric.nn.conv import MessagePassing


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple of
            feature sizes corresponds to the dimensionality of source and
            target node dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, normalize=False, bias=True,
                 **kwargs):
        super(SAGEConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_rel = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_root = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    # TODO: Allow different feature sizes (CHECK)
    # TODO: Allow tuples and tensors as inputs (CHECK)
    # TODO: Allow adj and edge_index as input (CHECK)
    # TODO: Make that jittable.
    # TODO: TEST DEFAULT PARAMS

    # propagate_type: (x: OptPairTensor)
    def forward(self, x, edge_index, size=None):
        # type: (Tensor, Tensor, Size) -> Tensor
        # type: (OptPairTensor, Tensor, Size) -> Tensor
        # type: (Tensor, SparseTensor, Size) -> Tensor
        # type: (OptPairTensor, SparseTensor, Size) -> Tensor
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, size=size, x=x)
        out = self.lin_rel(out)
        x_root = x[1]
        if x_root is not None:
            out += self.lin_root(x_root)
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        # TODO: adj_t = adj_t.set_value(None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
