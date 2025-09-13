# torch_geometric/nn/conv/spmd_gcn_conv.py
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.distributed.spmd_message_passing import SPMDMessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import add_remaining_self_loops, add_self_loops


class SPMDGCNConv(SPMDMessagePassing):
    """GCN layer with SPMD-style distributed message passing.
    
    This layer extends the standard GCN convolution to work with edge-level
    partitioning where each GPU processes a subset of edges while maintaining
    full node embeddings.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        zeros(self.bias)
    
    def forward(
        self, 
        x: Tensor, 
        edge_index: Adj, 
        edge_weight: OptTensor = None
    ) -> Tensor:
        """Forward pass with SPMD message passing."""
        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                           f"of node features as input while this layer "
                           f"does not support bipartite message passing.")
        
        # Linear transformation
        x = self.lin(x)
        
        # Add self-loops if requested
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = add_remaining_self_loops(
                    edge_index, edge_weight, 2. if self.improved else 1., x.size(0)
                )
        
        # SPMD message passing
        out = self.spmd_propagate(
            x, edge_index, edge_weight=edge_weight, size=(x.size(0), x.size(0))
        )
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        """Compute messages between nodes."""
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        """Update node embeddings after aggregation."""
        return aggr_out