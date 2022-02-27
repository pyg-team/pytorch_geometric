from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot
from torch_geometric.typing import (Adj, Optional, OptPairTensor, OptTensor,
                                    Size)
from torch_geometric.utils import softmax


class GeneralConv(MessagePassing):
    r"""A general GNN layer adapted from the `"Design Space for Graph Neural
    Networks" <https://arxiv.org/abs/2011.08843>`_ paper.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        in_edge_channels (int, optional): Size of each input edge.
            (default: :obj:`None`)
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        skip_linear (bool, optional): Whether apply linear function in skip
            connection. (default: :obj:`False`)
        directed_msg (bool, optional): If message passing is directed;
            otherwise, message passing is bi-directed. (default: :obj:`True`)
        heads (int, optional): Number of message passing ensembles.
            If :obj:`heads > 1`, the GNN layer will output an ensemble of
            multiple messages.
            If attention is used (:obj:`attention=True`), this corresponds to
            multi-head attention. (default: :obj:`1`)
        attention (bool, optional): Whether to add attention to message
            computation. (default: :obj:`False`)
        attention_type (str, optional): Type of attention: :obj:`"additive"`,
            :obj:`"dot_product"`. (default: :obj:`"additive"`)
        l2_normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge attributes :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: Optional[int],
        in_edge_channels: int = None,
        aggr: str = "add",
        skip_linear: str = False,
        directed_msg: bool = True,
        heads: int = 1,
        attention: bool = False,
        attention_type: str = "additive",
        l2_normalize: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_edge_channels = in_edge_channels
        self.aggr = aggr
        self.skip_linear = skip_linear
        self.directed_msg = directed_msg
        self.heads = heads
        self.attention = attention
        self.attention_type = attention_type
        self.normalize_l2 = l2_normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if self.directed_msg:
            self.lin_msg = Linear(in_channels[0], out_channels * self.heads,
                                  bias=bias)
        else:
            self.lin_msg = Linear(in_channels[0], out_channels * self.heads,
                                  bias=bias)
            self.lin_msg_i = Linear(in_channels[0], out_channels * self.heads,
                                    bias=bias)

        if self.skip_linear or self.in_channels != self.out_channels:
            self.lin_self = Linear(in_channels[1], out_channels, bias=bias)
        else:
            self.lin_self = torch.nn.Identity()

        if self.in_edge_channels is not None:
            self.lin_edge = Linear(in_edge_channels, out_channels * self.heads,
                                   bias=bias)

        # TODO: A general torch_geometric.nn.AttentionLayer
        if self.attention:
            if self.attention_type == 'additive':
                self.att_msg = Parameter(
                    torch.Tensor(1, self.heads, self.out_channels))
            elif self.attention_type == 'dot_product':
                self.scaler = torch.sqrt(
                    torch.tensor(out_channels, dtype=torch.float))
            else:
                raise ValueError(
                    f"Attention type '{self.attention_type}' not supported")

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_msg.reset_parameters()
        if hasattr(self.lin_self, 'reset_parameters'):
            self.lin_self.reset_parameters()
        if self.in_edge_channels is not None:
            self.lin_edge.reset_parameters()
        if self.attention and self.attention_type == 'additive':
            glorot(self.att_msg)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: Tensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        x_self = x[1]
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_attr=edge_attr)
        out = out.mean(dim=1)  # todo: other approach to aggregate heads
        out += self.lin_self(x_self)
        if self.normalize_l2:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message_basic(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor):
        if self.directed_msg:
            x_j = self.lin_msg(x_j)
        else:
            x_j = self.lin_msg(x_j) + self.lin_msg_i(x_i)
        if edge_attr is not None:
            x_j = x_j + self.lin_edge(edge_attr)
        return x_j

    def message(self, x_i: Tensor, x_j: Tensor, edge_index_i: Tensor,
                size_i: Tensor, edge_attr: Tensor) -> Tensor:
        x_j_out = self.message_basic(x_i, x_j, edge_attr)
        x_j_out = x_j_out.view(-1, self.heads, self.out_channels)
        if self.attention:
            if self.attention_type == 'dot_product':
                x_i_out = self.message_basic(x_j, x_i, edge_attr)
                x_i_out = x_i_out.view(-1, self.heads, self.out_channels)
                alpha = (x_i_out * x_j_out).sum(dim=-1) / self.scaler
            else:
                alpha = (x_j_out * self.att_msg).sum(dim=-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
            alpha = alpha.view(-1, self.heads, 1)
            return x_j_out * alpha
        else:
            return x_j_out
