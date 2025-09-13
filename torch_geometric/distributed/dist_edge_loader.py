# torch_geometric/distributed/dist_edge_loader.py
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch_geometric.distributed import DistContext, DistLoader
from torch_geometric.distributed.local_feature_store import LocalFeatureStore
from torch_geometric.distributed.local_graph_store import LocalGraphStore
from torch_geometric.data import Data, HeteroData


class DistEdgeLoader(DistLoader):
    """Distributed loader for edge-level partitioned graphs.
    
    This loader supports SPMD-style training where each GPU processes
    a subset of edges while maintaining full node embeddings.
    
    Args:
        data: Tuple of (LocalFeatureStore, LocalGraphStore) containing the graph data
        edge_partition_idx: Index of the edge partition to load
        current_ctx: Distributed context information
        master_addr: RPC address for distributed communication
        master_port: Port for RPC communication
        batch_size: Batch size for processing
        shuffle: Whether to shuffle the data
        **kwargs: Additional arguments passed to DistLoader
    """
    
    def __init__(
        self,
        data: Tuple[LocalFeatureStore, LocalGraphStore],
        edge_partition_idx: int,
        current_ctx: DistContext,
        master_addr: str,
        master_port: Union[int, str],
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs
    ):
        self.edge_partition_idx = edge_partition_idx
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load the edge partition data
        self.partition_data = self._load_partition_data(data, edge_partition_idx)
        
        # Initialize base class
        super().__init__(
            channel=None,  # No channel needed for edge-level loading
            master_addr=master_addr,
            master_port=master_port,
            current_ctx=current_ctx,
            dist_sampler=None,  # No sampler needed
            **kwargs
        )
    
    def _load_partition_data(
        self, 
        data: Tuple[LocalFeatureStore, LocalGraphStore], 
        partition_idx: int
    ) -> Union[Data, HeteroData]:
        """Load the edge partition data."""
        # This would typically load from disk, but for now we'll use the stores
        feature_store, graph_store = data
        
        if graph_store.meta['is_hetero']:
            # Create HeteroData from stores
            hetero_data = HeteroData()
            
            # Load node features
            for node_type in graph_store.meta['node_types']:
                node_attrs = feature_store.get_all_tensor_attrs()
                for attr in node_attrs:
                    if attr.group_name == node_type:
                        hetero_data[node_type][attr.attr_name] = feature_store.get_tensor(
                            attr.group_name, attr.attr_name, attr.index
                        )
            
            # Load edge information for this partition
            for edge_type in graph_store.meta['edge_types']:
                edge_attrs = graph_store.get_all_edge_attrs()
                for attr in edge_attrs:
                    if attr.edge_type == edge_type:
                        hetero_data[edge_type].edge_index = graph_store.get_edge_index(
                            attr.edge_type, attr.layout, attr.is_sorted
                        )
            
            return hetero_data
        else:
            # Create Data from stores
            data_obj = Data()
            
            # Load node features
            node_attrs = feature_store.get_all_tensor_attrs()
            for attr in node_attrs:
                data_obj[attr.attr_name] = feature_store.get_tensor(
                    attr.group_name, attr.attr_name, attr.index
                )
            
            # Load edge information for this partition
            edge_attrs = graph_store.get_all_edge_attrs()
            for attr in edge_attrs:
                if attr.attr_name == 'edge_index':
                    data_obj.edge_index = graph_store.get_edge_index(
                        attr.edge_type, attr.layout, attr.is_sorted
                    )
                else:
                    data_obj[attr.attr_name] = graph_store.get_edge_attr(
                        attr.edge_type, attr.attr_name, attr.index
                    )
            
            return data_obj
    
    def __iter__(self):
        """Iterate over the edge partition data."""
        # For SPMD training, we typically process the entire partition at once
        # since we have full node embeddings and only a subset of edges
        if self.shuffle:
            # Shuffle edge indices if needed
            if hasattr(self.partition_data, 'edge_index'):
                perm = torch.randperm(self.partition_data.edge_index.size(1))
                self.partition_data.edge_index = self.partition_data.edge_index[:, perm]
        
        yield self.partition_data
    
    def __len__(self):
        """Return the number of batches."""
        return 1  # For SPMD, we typically process the entire partition at once