from typing import Union, Tuple, Callable, Optional

from torch import Tensor
from torch.nn import Linear

from torch_geometric.typing import PairTensor, Adj
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import softmax

from ..inits import reset


class PointTransformerConv(MessagePassing):
    r"""The PointTransformer layer from the `"Point Transformer"
    <https://arxiv.org/abs/2012.09164>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i =  \sum_{j \in
        \mathcal{N}(i) \cup \{ i \}} \alpha_{i,j} \left(W_3\mathbf{x}_j
        + \delta_{ij} \right)

    where the attention coefficients :math:`\alpha_{i,j}` and
    positional embedding :math:`\delta_{ij}` are computed as.

    .. math::
        \alpha_{i,j}= \textrm{softmax} \left( \gamma_\mathbf{\Theta}
        W_1\mathbf{x}_i - W_2\mathbf{x}_j + \delta_{i,j}  \right)

    .. math::
        \delta_{i,j}= h_{\mathbf{\Theta}}(pos_i - pos_j)

    where :math:`\gamma_\mathbf{\Theta}` and :math:`h_\mathbf{\Theta}`
    denote neural networks, *.i.e.* MLPs, that are
    used to compute attention coefficients and map relative spatial
    coordinates respectively. :math:`\mathbf{P} \in \mathbb{R}^{N \times D}`
    defines the position of each point.

    Args:
        in_channels (int or tuple): Size of each input sample features
        out_channels (int) : size of each output sample features
        local_nn : (torch.nn.Module, optional): A neural network
            :math:`h_\mathbf{\Theta}` which maps relative spatial coordinates
            :obj:`pos_j - pos_i` of shape :obj:`[-1, 3]` to shape
            :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. Will default to a linear
            mapping if not specified. (default: :obj:`None`)
        attn_nn : (torch.nn.Module, optional): A neural network
            :math:`\gamma_\mathbf{\Theta}` which maps transformed
            node features of shape :obj:`[-1, out_channels]`
            to shape :obj:`[-1, out_channels]`, *e.g.*, defined
            by :class:`torch.nn.Sequential`.(default: :obj:`None`)
        add_self_loops (bool, optional) : If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, local_nn: Optional[Callable] = None,
                 attn_nn: Optional[Callable] = None,
                 add_self_loops: bool = True):

        super(PointTransformerConv,
              self).__init__(aggr='add')  # "Add" aggregation

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if local_nn is None:
            local_nn = Linear(3, out_channels)

        self.local_nn = local_nn
        self.attn_nn = attn_nn
        self.lin = Linear(in_channels[0], out_channels)
        self.lin_src = Linear(in_channels[0], out_channels)
        self.lin_dst = Linear(in_channels[1], out_channels)

        self.add_self_loops = add_self_loops

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.attn_nn)
        reset(self.lin)
        reset(self.lin_src)
        reset(self.lin_dst)

    def forward(self, x: Union[Tensor, PairTensor],
                pos: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""

        if not isinstance(x, tuple):
            alpha = (self.lin_src(x), self.lin_dst(x))
            x = self.lin(x)
            x: PairTensor = (x, x)
        else:
            alpha = (self.lin_src(x[0]), self.lin_dst(x[1]))
            x = (self.lin(x[0]), x[1])

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=pos[1].size(0))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairTensor, pos: PairTensor, alpha: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, alpha=alpha, size=None)
        return out

    def message(self, x_j, pos_i, pos_j, alpha_i, alpha_j, index):

        out_delta = self.local_nn((pos_j - pos_i))
        alpha = alpha_i - alpha_j + out_delta
        if self.attn_nn is not None:
            alpha = self.attn_nn(alpha)
        alpha = softmax(alpha, index)

        msg = alpha * (x_j + out_delta)

        return msg

    def __repr__(self):
        return '{}({}, {}, local_nn={}, attn_nn={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.local_nn, self.attn_nn)
