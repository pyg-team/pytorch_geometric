from typing import Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM

from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.utils import to_dense_batch


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggregator_type (str): ['mean', 'max', 'gcn', 'lstm'], mean by default
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
            For eg: to inherit the aggregation function implementation from
            `torch_geometric.nn.conv.MessagePassing`, set (aggr = 'func_name')
            where func_name is in ['mean', 'sum', 'add', 'min', 'max', 'mul'];
            additionally, set the flow direction of message passing by passing
            the flow argument as either (flow = 'source_to_target') or
            (flow = 'target_to_source')

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
                 out_channels: int, aggregator_type: str = 'mean', 
                 normalize: bool = False, root_weight: bool = True, 
                 bias: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.aggregator_type = aggregator_type

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
            
        if self.aggregator_type == 'gcn':
            # Convolutional aggregator does not concatenate the root node 
            # i.e it doesn't concatenate the nodes previous layer
            self.root_weight = False 

        if self.aggregator_type == 'lstm':
            self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias) # neighbours
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False) # root

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
        if self.aggregator_type == 'lstm':
            self.lstm.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        # propagate internally calls message_and_aggregate()
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        # updates node embeddings
        x_r = x[1]
        if self.root_weight and x_r is not None:
            # root weight does not get concatenated for the convolutional aggregator
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor, edge_index_j, edge_index_i) -> Tensor:
        """
            Performs both message passing and aggregation of messages from neighbours using the aggregator_type
        """
        adj_t = adj_t.set_value(None, layout=None)
        if self.aggregator_type == 'mean' or self.aggregator_type == 'gcn':
            return matmul(adj_t, x[0], reduce='mean')

        elif self.aggregator_type == 'max':
            return matmul(adj_t, x[0], reduce='max')
            
        elif self.aggregator_type == 'lstm':
            x_j = x[0][edge_index_j]
            x, mask = to_dense_batch(x_j, edge_index_i)
            return self.lstm(x)
        