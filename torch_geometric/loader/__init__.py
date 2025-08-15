from torch_geometric.deprecation import deprecated

from .dataloader import DataLoader
from .node_loader import NodeLoader
from .link_loader import LinkLoader
from .neighbor_loader import NeighborLoader
from .link_neighbor_loader import LinkNeighborLoader
from .hgt_loader import HGTLoader
from .cluster import ClusterData, ClusterLoader
from .graph_saint import (GraphSAINTSampler, GraphSAINTNodeSampler,
                          GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler)
from .shadow import ShaDowKHopSampler
from .random_node_loader import RandomNodeLoader
# from .ibmb_loader import IBMBBatchLoader, IBMBNodeLoader
from .zip_loader import ZipLoader
from .data_list_loader import DataListLoader
from .dense_data_loader import DenseDataLoader
from .temporal_dataloader import TemporalDataLoader
from .neighbor_sampler import NeighborSampler
from .imbalanced_sampler import ImbalancedSampler
from .dynamic_batch_sampler import DynamicBatchSampler
from .prefetch import PrefetchLoader
from .cache import CachedLoader
from .mixin import AffinityMixin
from .rag_loader import RAGQueryLoader, RAGFeatureStore, RAGGraphStore

__all__ = classes = [
    'DataLoader',
    'NodeLoader',
    'LinkLoader',
    'NeighborLoader',
    'LinkNeighborLoader',
    'HGTLoader',
    'ClusterData',
    'ClusterLoader',
    'GraphSAINTSampler',
    'GraphSAINTNodeSampler',
    'GraphSAINTEdgeSampler',
    'GraphSAINTRandomWalkSampler',
    'ShaDowKHopSampler',
    'RandomNodeLoader',
    # 'IBMBBatchLoader',
    # 'IBMBNodeLoader',
    'ZipLoader',
    'DataListLoader',
    'DenseDataLoader',
    'TemporalDataLoader',
    'NeighborSampler',
    'ImbalancedSampler',
    'DynamicBatchSampler',
    'PrefetchLoader',
    'CachedLoader',
    'AffinityMixin',
    'RAGQueryLoader',
    'RAGFeatureStore',
    'RAGGraphStore'
]

RandomNodeSampler = deprecated(
    details="use 'loader.RandomNodeLoader' instead",
    func_name='loader.RandomNodeSampler',
)(RandomNodeLoader)
