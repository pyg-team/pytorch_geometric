# torch_geometric/distributed/edge_partition.py
import json
import logging
import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch_geometric.data import Data, HeteroData
from torch_geometric.io import fs
from torch_geometric.utils import to_undirected


class EdgePartitioner:
    r"""Edge-level partitioner for SPMD-style distributed GNN training.
    
    This partitioner keeps full copies of node embeddings on each GPU while
    partitioning only the edges. This is suitable for GNN reasoning tasks
    where messages need to be propagated to all nodes conditioned on queries.
    
    Args:
        data (Data or HeteroData): The data object.
        num_parts (int): The number of partitions.
        root (str): Root directory where the partitioned dataset should be saved.
        strategy (str): Partitioning strategy ('random', 'balanced', 'metis').
        undirected (bool): Whether to make the graph undirected before partitioning.
    """
    
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_parts: int,
        root: str,
        strategy: str = "random",
        undirected: bool = True,
    ):
        assert num_parts > 1
        
        self.data = data
        self.num_parts = num_parts
        self.root = root
        self.strategy = strategy
        self.undirected = undirected
        
        # Make graph undirected if requested
        if undirected and not isinstance(data, HeteroData):
            self.data = to_undirected(data)
        elif undirected and isinstance(data, HeteroData):
            # For heterogeneous graphs, make each edge type undirected
            for edge_type in data.edge_types:
                data[edge_type] = to_undirected(data[edge_type])
    
    @property
    def is_hetero(self) -> bool:
        return isinstance(self.data, HeteroData)
    
    def generate_edge_partition(self):
        """Generate edge-level partitions while keeping full node copies."""
        os.makedirs(self.root, exist_ok=True)
        
        if self.is_hetero:
            self._partition_hetero()
        else:
            self._partition_homo()
        
        self._save_metadata()
    
    def _partition_homo(self):
        """Partition homogeneous graph by edges."""
        data = self.data
        num_edges = data.num_edges
        num_nodes = data.num_nodes
        
        # Create edge partition mapping
        if self.strategy == "random":
            edge_partition = torch.randint(0, self.num_parts, (num_edges,))
        elif self.strategy == "balanced":
            # Balanced partitioning
            edges_per_part = num_edges // self.num_parts
            remainder = num_edges % self.num_parts
            
            edge_partition = torch.zeros(num_edges, dtype=torch.long)
            start_idx = 0
            for i in range(self.num_parts):
                end_idx = start_idx + edges_per_part + (1 if i < remainder else 0)
                edge_partition[start_idx:end_idx] = i
                start_idx = end_idx
        else:
            raise ValueError(f"Unknown partitioning strategy: {self.strategy}")
        
        # Save partitions
        for pid in range(self.num_parts):
            logging.info(f'Saving edge partition {pid}')
            path = osp.join(self.root, f'part_{pid}')
            os.makedirs(path, exist_ok=True)
            
            # Get edges for this partition
            edge_mask = edge_partition == pid
            part_edge_index = data.edge_index[:, edge_mask]
            part_edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else None
            
            # Create partition data with full node features
            part_data = Data(
                x=data.x.clone(),  # Full copy of node features
                edge_index=part_edge_index,
                edge_attr=part_edge_attr,
                y=data.y.clone() if data.y is not None else None,
                num_nodes=num_nodes,  # Keep original number of nodes
            )
            
            # Save partition
            torch.save(part_data, osp.join(path, 'data.pt'))
            
            # Save edge mapping
            torch.save(edge_mask, osp.join(path, 'edge_mask.pt'))
        
        # Save global edge partition mapping
        torch.save(edge_partition, osp.join(self.root, 'edge_partition.pt'))
    
    def _partition_hetero(self):
        """Partition heterogeneous graph by edges."""
        data = self.data
        
        # Create edge partition mapping for each edge type
        edge_partitions = {}
        for edge_type in data.edge_types:
            num_edges = data[edge_type].num_edges
            
            if self.strategy == "random":
                edge_partition = torch.randint(0, self.num_parts, (num_edges,))
            elif self.strategy == "balanced":
                edges_per_part = num_edges // self.num_parts
                remainder = num_edges % self.num_parts
                
                edge_partition = torch.zeros(num_edges, dtype=torch.long)
                start_idx = 0
                for i in range(self.num_parts):
                    end_idx = start_idx + edges_per_part + (1 if i < remainder else 0)
                    edge_partition[start_idx:end_idx] = i
                    start_idx = end_idx
            else:
                raise ValueError(f"Unknown partitioning strategy: {self.strategy}")
            
            edge_partitions[edge_type] = edge_partition
        
        # Save partitions
        for pid in range(self.num_parts):
            logging.info(f'Saving edge partition {pid}')
            path = osp.join(self.root, f'part_{pid}')
            os.makedirs(path, exist_ok=True)
            
            # Create partition data with full node features
            part_data = HeteroData()
            
            # Copy all node features (full copies)
            for node_type in data.node_types:
                part_data[node_type].x = data[node_type].x.clone()
                part_data[node_type].y = data[node_type].y.clone() if data[node_type].y is not None else None
                part_data[node_type].num_nodes = data[node_type].num_nodes
            
            # Copy edges for this partition
            for edge_type in data.edge_types:
                edge_mask = edge_partitions[edge_type] == pid
                part_data[edge_type].edge_index = data[edge_type].edge_index[:, edge_mask]
                if data[edge_type].edge_attr is not None:
                    part_data[edge_type].edge_attr = data[edge_type].edge_attr[edge_mask]
            
            # Save partition
            torch.save(part_data, osp.join(path, 'data.pt'))
            
            # Save edge masks for this partition
            edge_masks = {edge_type: edge_partitions[edge_type] == pid 
                         for edge_type in data.edge_types}
            torch.save(edge_masks, osp.join(path, 'edge_masks.pt'))
        
        # Save global edge partition mappings
        torch.save(edge_partitions, osp.join(self.root, 'edge_partitions.pt'))
    
    def _save_metadata(self):
        """Save metadata about the partitioning."""
        meta = {
            'num_parts': self.num_parts,
            'strategy': self.strategy,
            'undirected': self.undirected,
            'is_hetero': self.is_hetero,
            'num_nodes': self.data.num_nodes if not self.is_hetero else None,
            'num_edges': self.data.num_edges if not self.is_hetero else None,
        }
        
        if self.is_hetero:
            meta.update({
                'node_types': self.data.node_types,
                'edge_types': self.data.edge_types,
            })
        
        with open(osp.join(self.root, 'META.json'), 'w') as f:
            json.dump(meta, f)


def load_edge_partition_info(
    root_dir: str,
    partition_idx: int,
) -> Tuple[Dict, int, int, Union[torch.Tensor, Dict]]:
    """Load edge partition information."""
    with open(osp.join(root_dir, 'META.json'), 'rb') as infile:
        meta = json.load(infile)
    
    num_partitions = meta['num_parts']
    assert partition_idx >= 0
    assert partition_idx < num_partitions
    
    partition_dir = osp.join(root_dir, f'part_{partition_idx}')
    assert osp.exists(partition_dir)
    
    if meta['is_hetero']:
        edge_partitions = fs.torch_load(osp.join(root_dir, 'edge_partitions.pt'))
        return meta, num_partitions, partition_idx, edge_partitions
    else:
        edge_partition = fs.torch_load(osp.join(root_dir, 'edge_partition.pt'))
        return meta, num_partitions, partition_idx, edge_partition