import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class MixHopConv(MessagePassing):
    def __init__(self,
                 in_dim,
                 out_dim,
                 p=[0, 1, 2],
                 dropout=0,
                 activation=None,
                 batchnorm=False,kwargs):
      
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim * len(p))
        
        # define weight dict for each power j
        self.weights = nn.ModuleDict({
            str(j): nn.Linear(in_dim, out_dim, bias=False) for j in p
        })

        self.max_j =  max(self.p) + 1

    def forward(self, x, edge_index):

        outputs = []

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        feats = x


        for j in range(self.max_j):

            if j in self.p:
                output = self.weights[str(j)](feats)
                outputs.append(output)

            
            feats = self.propagate(edge_index, x=x, norm=norm)

        final = torch.cat(outputs, dim=1)

        if self.batchnorm:
            final = self.bn(final)
        
        if self.activation is not None:
            final = self.activation(final)
        
        final = self.dropout(final)

        return final

    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j




class MixHop(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim, 
                 out_dim,
                 num_layers=2,
                 p=[0, 1, 2],
                 input_dropout=0.0,
                 layer_dropout=0.0,
                 activation=None,
                 batchnorm=False):
        super(MixHop, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.p = p
        self.input_dropout = input_dropout
        self.layer_dropout = layer_dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(self.input_dropout)

        # Input layer
        self.layers.append(MixHopConv(self.in_dim,
                                      self.hid_dim,
                                      p=self.p,
                                      dropout=self.input_dropout,
                                      activation=self.activation,
                                      batchnorm=self.batchnorm))
        
        # Hidden layers with n - 1 MixHopConv layers
        for i in range(self.num_layers - 2):
            self.layers.append(MixHopConv(self.hid_dim * len(p),
                                          self.hid_dim,
                                          p=self.p,
                                          dropout=self.layer_dropout,
                                          activation=self.activation,
                                          batchnorm=self.batchnorm))
        
        self.fc_layers = nn.Linear(self.hid_dim * len(p), self.out_dim, bias=False)

    def forward(self, x,edge_index):
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        
        x = self.fc_layers(x)

        return x

# import torch
# from torch.nn import Linear, Parameter
# from torch_geometric.nn import MessagePassing
# from torch_geometric.utils import add_self_loops, degree
# from typing import Optional
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
# import torch
# from torch import Tensor
# from torch.nn import Parameter

# from torch_geometric.nn.inits import zeros
# from torch_geometric.typing import (
#     Adj,
#     OptPairTensor,
#     OptTensor,
#     SparseTensor,
#     torch_sparse,
# )
# from torch_geometric.utils import (
#     add_remaining_self_loops,
#     is_torch_sparse_tensor,
#     scatter,
#     spmm,
#     to_edge_index,
#     to_torch_coo_tensor,
# )

# from torch_geometric.utils.num_nodes import maybe_num_nodes


# class MixHopConv(MessagePassing):
#     def __init__(self,
#                  in_dim,
#                  out_dim,
#                  p=[0, 1, 2],
#                  dropout=0,
#                  activation=None,
#                  batchnorm=False,**kwargs):
#         kwargs.setdefault('aggr', 'add')
#         super().__init__(**kwargs)
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.p = p
#         # self.activation = activation
#         self.batchnorm = batchnorm
#         # define dropout layer
#         self.dropout = nn.Dropout(dropout)
#         # define batch norm layer
#         if self.batchnorm:
#             self.bn = nn.BatchNorm1d(out_dim * len(p))
#         # define weight dict for each power j
#         self.weights = nn.ModuleDict({
#             str(j): nn.Linear(in_dim, out_dim, bias=False) for j in p
#         })
#         self.max_j =  max(self.p) + 1

#     def forward(self,x: Tensor, edge_index: Adj,edge_weight: OptTensor = None)-> Tensor:
        
#         if self.normalize:
#             if isinstance(edge_index, Tensor):
#                 cache = self._cached_edge_index
#                 if cache is None:
#                     edge_index, edge_weight = gcn_norm(  # yapf: disable
#                         edge_index, edge_weight, x.size(self.node_dim),
#                         self.improved, self.add_self_loops, self.flow, x.dtype)
#                     if self.cached:
#                         self._cached_edge_index = (edge_index, edge_weight)
#                 else:
#                     edge_index, edge_weight = cache[0], cache[1]

#             elif isinstance(edge_index, SparseTensor):
#                 cache = self._cached_adj_t
#                 if cache is None:
#                     edge_index = gcn_norm(  # yapf: disable
#                         edge_index, edge_weight, x.size(self.node_dim),
#                         self.improved, self.add_self_loops, self.flow, x.dtype)
#                     if self.cached:
#                         self._cached_adj_t = edge_index
#                 else:
#                     edge_index = cache
                    
                    
#         outputs = []
        
        
#         feats = x
#         for j in range(self.max_j):
#             if j in self.p:
#                 output = self.weights[str(j)](feats)
#                 outputs.append(output)
#             feats = self.propagate(edge_index, x=x, edge_weight=edge_weight,size=None)
            
#         final = torch.cat(outputs, dim=1)
#         if self.batchnorm:
#             final = self.bn(final)

#         final = self.dropout(final)
        
#         if self.bias is not None:
#             final = final + self.bias
            
#         return final
    
#     def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
#         return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
#     def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
#         return spmm(adj_t, x, reduce=self.aggr)
