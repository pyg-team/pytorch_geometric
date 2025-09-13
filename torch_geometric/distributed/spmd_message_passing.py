# torch_geometric/distributed/spmd_message_passing.py
import torch
import torch.distributed as dist
from typing import Any, Dict, Optional, Tuple, Union
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import scatter


class SPMDMessagePassing(MessagePassing):
    """SPMD-style message passing for distributed GNN reasoning.
    
    This class extends the base MessagePassing to support distributed
    message passing where each GPU has a full copy of node embeddings
    but only processes a subset of edges.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self._cached_messages = None
    
    def spmd_propagate(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> torch.Tensor:
        """SPMD-style message passing with edge-level partitioning.
        
        Args:
            x: Node feature matrix of shape [num_nodes, in_channels]
            edge_index: Edge connectivity of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            size: Optional size tuple (num_src_nodes, num_dst_nodes)
            **kwargs: Additional arguments passed to message function
            
        Returns:
            Updated node features of shape [num_nodes, out_channels]
        """
        if not dist.is_initialized() or self.world_size == 1:
            # Fall back to regular message passing for single GPU
            return self.propagate(edge_index, x=x, edge_weight=edge_weight, 
                                size=size, **kwargs)
        
        # Step 1: Compute messages locally on this GPU's edge partition
        local_messages = self._compute_local_messages(
            x, edge_index, edge_weight, **kwargs
        )
        
        # Step 2: All-reduce messages across all GPUs
        global_messages = self._all_reduce_messages(local_messages, x.size(0))
        
        # Step 3: Update node embeddings
        updated_x = self._update_embeddings(x, global_messages)
        
        return updated_x
    
    def _compute_local_messages(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute messages for local edge partition."""
        # Use the base class message computation
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index[0], edge_index[1]
            
            # Get source and target node features
            x_i = x[row]  # Source nodes
            x_j = x[col]  # Target nodes
            
            # Compute messages
            messages = self.message(x_i, x_j, edge_weight, **kwargs)
            
            # Aggregate messages by target nodes
            if self.aggr == 'add' or self.aggr == 'sum':
                aggregated = scatter(messages, col, dim=0, dim_size=x.size(0), reduce='sum')
            elif self.aggr == 'mean':
                aggregated = scatter(messages, col, dim=0, dim_size=x.size(0), reduce='mean')
            elif self.aggr == 'max':
                aggregated = scatter(messages, col, dim=0, dim_size=x.size(0), reduce='max')
            else:
                raise ValueError(f"Unsupported aggregation: {self.aggr}")
            
            return aggregated
        else:
            raise NotImplementedError("SparseTensor not yet supported in SPMD mode")
    
    def _all_reduce_messages(
        self, 
        local_messages: torch.Tensor, 
        num_nodes: int
    ) -> torch.Tensor:
        """All-reduce messages across all GPUs."""
        # Ensure all GPUs have the same tensor shape
        if local_messages.size(0) != num_nodes:
            # Pad with zeros if necessary
            padded_messages = torch.zeros(
                num_nodes, local_messages.size(1), 
                dtype=local_messages.dtype, device=local_messages.device
            )
            padded_messages[:local_messages.size(0)] = local_messages
            local_messages = padded_messages
        
        # All-reduce across all GPUs
        dist.all_reduce(local_messages, op=dist.ReduceOp.SUM)
        
        return local_messages
    
    def _update_embeddings(
        self, 
        x: torch.Tensor, 
        messages: torch.Tensor
    ) -> torch.Tensor:
        """Update node embeddings with aggregated messages."""
        # This is a simple update - subclasses can override for more complex updates
        return x + messages
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                edge_weight: OptTensor = None, **kwargs) -> torch.Tensor:
        """Compute messages between nodes.
        
        Override this method in subclasses to define custom message functions.
        """
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update node embeddings after aggregation.
        
        Override this method in subclasses to define custom update functions.
        """
        return aggr_out